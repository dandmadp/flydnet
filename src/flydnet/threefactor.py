"""3요소 학습 규칙 (e-prop 방식, Bellec et al. 2020): 역전파 없이, 뇌에서 가능한 정보만으로 커넥톰 연결을 학습

  시냅스 변화 = 학습 신호(오차, 도파민처럼 넓게 퍼짐) x 시냅스 후 민감도(문턱 근처일수록 큼) x 시냅스 전 흔적(적격 흔적)

  tf = fd.ThreeFactor(layer, feedback="random")       # 또는 "connectome", "none"
  out = tf(x, seed=s)                                  # 순전파 (시간 역전파 경로 없음)
  loss = fd.surprise(readout(out), y)
  rule.clear(); loss.retrograde()                      # 리드아웃까지만 (출력 뉴런의 오차 = out.retro)
  tf.assign(out)                                       # 같은 seed로 한 번 더 순전파하며 흔적 x 신호 → layer 연결의 .retro
  rule.step()

역전파(BPTT)와 다른 점
  - 시간을 거슬러 가지 않는다: 메모리가 시뮬레이션 길이와 무관 (흔적은 뉴런마다 두 개)
  - 오차가 정확한 경로가 아니라 피드백으로 전달된다. 출력 뉴런은 자기 오차를 그대로, 나머지 뉴런은 feedback으로:
      "random"     고정된 무작위 피드백 (출력 뉴런 → 각 뉴런, 표준 e-prop)
      "connectome" 실제 커넥톰 연결을 따라 출력 뉴런에서 hops 단계 퍼진 오차 (실제로 있는 피드백 경로만)
      "none"       출력 뉴런으로 들어오는 연결만 학습
  - 적격 흔적은 학습하는 시냅스마다 (시냅스 후 뉴런의 리셋·불응기까지 반영). 그래서 출력 뉴런으로 들어오는 연결은
    역전파(대리 기울기, 리셋에서 끊음)와 같은 기울기
비용: 순전파 2번 + 스텝마다 학습 연결 수 x 배치만큼 계산. 메모리: 학습 연결 수 x 배치 x 2 (시뮬레이션 길이와 무관)
"""
from __future__ import annotations

import numpy as np

from .ganglion import backend as B
from .ganglion import kernels as K
from .ganglion.signal import Signal, quiescent


class ThreeFactor:
    """ConnectomeLayer(trainable=..., timing="brian", neuron="lif")의 연결을 3요소 규칙으로 학습"""

    def __init__(self, layer, feedback: str = "random", hops: int = 2, seed: int = 0):
        if feedback not in ("random", "connectome", "none"):
            raise ValueError(f"feedback은 'random', 'connectome', 'none': {feedback}")
        if not getattr(layer, "trainable", False):
            raise ValueError("학습하는 연결이 있는 ConnectomeLayer에서만 (trainable=True 또는 연결 종류 목록)")
        if layer.neuron != "lif" or layer.timing != "brian":
            raise ValueError("스파이킹 뉴런 (neuron='lif', timing='brian')에서만")
        if layer.neuron_params:
            raise ValueError("세포 유형별 뉴런 매개변수(bias·t_mbr)를 쓰는 층은 아직 지원 안 함")
        if hops < 1:
            raise ValueError("hops는 1 이상")
        self.layer, self.feedback, self.hops = layer, feedback, hops
        N, dev = layer.circuit.N, layer.device
        self.out_idx = B.numpy(layer.out_idx)
        is_out = np.zeros(N, bool); is_out[self.out_idx] = True
        self._hidden = B.to(~is_out, dev)
        if feedback == "random":
            F = np.random.default_rng(seed).standard_normal((N, len(self.out_idx))).astype(np.float32)
            F[is_out] = 0
            self._F = B.to(F, dev)
        elif feedback == "connectome":
            c = layer.circuit
            A = B.sparse("cpu").csr_matrix((c.weight.astype(np.float32), (c.post, c.pre)), shape=(N, N))
            self._A = B.sparse(dev).csr_matrix(A) if dev == "gpu" else A           # 받는 뉴런 x 주는 뉴런
        self._pending = None

    def __call__(self, rates=None, seed: int | None = None, batch: int = 1):
        """순전파 (역전파 경로 없이). 돌려주는 신호는 plastic 잎: 리드아웃 손실의 역행성 신호가 여기 쌓임"""
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        with quiescent():
            out = self.layer(rates, seed=seed, batch=batch)
        self._pending = (rates, seed, batch)
        return Signal(out.data, plastic=True)

    def _signal(self, delta, rate_c):
        """학습 신호 L (N, B): 출력 뉴런은 자기 오차, 나머지는 피드백. 피드백의 크기는 출력 오차와 같은 RMS로 맞춤"""
        xp = B.xp(self.layer.device)
        N, Bn = self.layer.circuit.N, delta.shape[0]
        L = xp.zeros((N, Bn), dtype=xp.float32)
        d = (delta.T * rate_c).astype(xp.float32)                            # (n_out, B) 스파이크 하나당 오차
        L[self.layer.out_idx] = d
        if self.feedback == "none":
            return L
        if self.feedback == "random":
            h = self._F @ d
        else:                                                                # 실제 연결을 따라 hops 단계
            src = xp.zeros((N, Bn), dtype=xp.float32); src[self.layer.out_idx] = d
            h, cur = xp.zeros_like(src), src
            for _ in range(self.hops):
                cur = self._A @ cur
                h += cur
        h = h * self._hidden[:, None]
        rms_h = float(xp.sqrt((h ** 2).mean())) if h.size else 0.0
        if rms_h > 0:
            h *= float(xp.sqrt((d ** 2).mean())) / rms_h
        return L + h

    def assign(self, out: Signal):
        """out.retro (리드아웃 손실의 기울기)로 학습 신호를 만들고, 같은 순전파를 다시 돌며 적격 흔적과 곱해
        layer의 학습 연결에 .retro를 쌓음. 반환: 연결마다 기울기 (E,)"""
        if self._pending is None:
            raise RuntimeError("먼저 tf(x, seed=...)로 순전파")
        if out.retro is None:
            raise RuntimeError("out.retro가 없음 - 리드아웃 손실에서 retrograde()를 먼저")
        rates, seed, batch = self._pending
        self._delta = B.to(B.numpy(out.retro), self.layer.device)
        self.layer._observer = self
        try:
            with quiescent():
                self.layer(rates, seed=seed, batch=batch)
        finally:
            self.layer._observer = None
        xp = B.xp(self.layer.device)
        g = xp.zeros(len(self.layer.w_base), dtype=xp.float32)
        g[self.layer.train_pos] = self._gt
        self.layer.values().retrograde(g)                                  # 연결 세기 → 학습 배율 (log_scale)
        self._pending = None
        return g

    # ─────────────── 관찰자: ConnectomeLayer 순전파가 부름 ───────────────
    def begin(self, info):
        xp = B.xp(self.layer.device)
        Bn = info["batch"]
        self._i = info
        self._L = self._signal(self._delta, info["rate_c"])
        w = self.layer.wiring
        pos = self.layer.train_pos                                           # 학습하는 연결만
        self._post, self._pre = w.post[pos], w.pre[pos]
        E = len(pos)
        self._eV = xp.zeros((E, Bn), dtype=xp.float32)                       # 적격 흔적: dV_post / d w (시냅스마다)
        self._eG = xp.zeros((E, Bn), dtype=xp.float32)                       # 적격 흔적: dG_post / d w
        self._ring = [None] * info["dly"]                                    # 지연 중인 시냅스 전 스파이크 (s - dly에 보낸 것)
        self._gt = xp.zeros(E, dtype=xp.float32)

    def step(self, s, u, act, sent):
        """lif_step_brian과 같은 순서: 적분(act면) → 발화 판정 → 도착(act면) → 리셋(발화면 흔적 0)"""
        i = self._i
        xp = B.xp(self.layer.device)
        a = act[self._post]                                                  # (E, B) 시냅스 후 뉴런이 적분 중인지
        self._eV = xp.where(a, i["e_v"] * self._eV + i["e_g"] * self._eG, self._eV)
        if s >= i["s_cnt"]:
            psi = (1.0 / i["scale"]) / (1 + i["slope"] * xp.abs(u)) ** 2       # 대리 기울기 (역전파와 같음)
            if i["gain"] is not None:
                psi = psi * i["gain"]
            self._gt += ((self._L * psi)[self._post] * self._eV).sum(axis=1)
        k = s % len(self._ring)
        arrived = self._ring[k]                                               # s - dly에 보낸 스파이크가 지금 도착
        self._ring[k] = sent
        decay = xp.where(a, i["gd"] * self._eG, self._eG)
        self._eG = decay + (xp.where(a, arrived[self._pre], 0) if arrived is not None else 0)
        fired = (u > 0)[self._post]                                          # 리셋: V = v_rst, G = 0 → 흔적도 0
        self._eV[fired] = 0
        self._eG[fired] = 0
