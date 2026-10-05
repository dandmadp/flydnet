"""관찰자 규격과 STDP: 순전파를 스텝마다 지켜보며 학습 신호를 만드는 규칙 (사용자 정의 학습 규칙의 틀)

관찰자 규격 (fd.ThreeFactor, fd.STDP가 따름 - 같은 방식으로 직접 만들 수 있음)
  begin(info)                    순전파 시작: info = dict(steps, s_cnt, batch, dt, dly, gain, rate_c, ...)
  step(s, spikes, **extra)       스텝마다: spikes = 이번 스텝에 시냅스로 나간 스파이크 (N, 배치) 배열
                                 extra: fired (실제 발화 - Shibire(block)로 전달이 막혀도 1), 내장 LIF면 u (문턱까지 거리),
                                 act (적분 중인지)
  층에 붙이기: layer._observer = 관찰자 (Monitor.run이 대신 해 줌)

  stdp = fd.STDP(layer, a_plus=0.01, a_minus=0.012)
  out = stdp(x, seed=s)              # 순전파하며 스파이크 시각 쌍을 셈 (역전파 경로 없음)
  stdp.assign()                       # 변화량을 layer 학습 값의 .retro로 (-Δ) → 아무 가소성 규칙으로 rule.step()
"""
from __future__ import annotations

from . import _check as _C
import numpy as np

from .ganglion import backend as B
from .ganglion.signal import quiescent
from .genetics import training


class Monitor:
    """관찰자 기본 클래스: run()이 층에 붙여 순전파를 돌리고 떼어냄"""

    def __init__(self, layer):
        if not hasattr(layer, "_observer"):
            raise TypeError("ConnectomeLayer에서만")
        self.layer = layer

    def run(self, rates=None, seed: int | None = None, batch: int = 1):
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.layer._observer = self
        try:
            with quiescent(), training():                          # mosaic은 켜 둠 (학습 중)
                return self.layer(rates, seed=seed, batch=batch)
        finally:
            self.layer._observer = None

    __call__ = run

    def begin(self, info):
        pass

    def step(self, s, spikes, **extra):
        pass


class STDP(Monitor):
    """쌍 기반 STDP (Song et al. 2000): 시냅스 전 스파이크 뒤에 시냅스 후 스파이크면 강화, 반대면 약화

      Δ = a_plus · (시냅스 전 흔적) · (시냅스 후 스파이크)  -  a_minus · (시냅스 후 흔적) · (시냅스 전 스파이크)
      시냅스 전은 실제로 전달된 스파이크, 시냅스 후는 실제 발화 (Shibire로 전달만 막힌 뉴런도 발화는 함)
      흔적은 스파이크마다 1 더하고 tau_plus / tau_minus ms로 감쇠. 배치 평균, 학습하는 연결만.
    변화량은 학습 배율(log_scale)에 대한 것 = 곱셈형 (세기의 부호·데일의 법칙 유지).
    근사: 시냅스 지연은 무시 (보낸 시각 기준)"""

    def __init__(self, layer, a_plus: float = 0.01, a_minus: float = 0.012, tau_plus: float = 20.0,
                 tau_minus: float = 20.0):
        _C.nonneg('a_plus', a_plus)
        _C.nonneg('a_minus', a_minus)
        _C.pos('tau_plus', tau_plus)
        _C.pos('tau_minus', tau_minus)
        super().__init__(layer)
        if not getattr(layer, "trainable", False):
            raise ValueError("학습하는 연결이 있는 ConnectomeLayer에서만 (trainable=...)")
        self.a_plus, self.a_minus, self.tau_plus, self.tau_minus = a_plus, a_minus, tau_plus, tau_minus
        self.delta = None

    def begin(self, info):
        xp = B.xp(self.layer.device)
        N, Bn, dt = self.layer.circuit.N, info["batch"], info.get("dt", self.layer.p["dt"])
        self._dp, self._dm = float(np.exp(-dt / self.tau_plus)), float(np.exp(-dt / self.tau_minus))
        self._x = xp.zeros((N, Bn), dtype=xp.float32)                  # 시냅스 전 흔적
        self._y = xp.zeros((N, Bn), dtype=xp.float32)                  # 시냅스 후 흔적
        w, pos = self.layer.wiring, self.layer.train_pos
        self._pre, self._post = w.pre[pos], w.post[pos]
        self._d = xp.zeros(len(pos), dtype=xp.float32)
        self._bn = Bn

    def step(self, s, spikes, **extra):
        S = spikes                                                     # 시냅스 전: 실제로 전달된 스파이크
        F = extra.get("fired", spikes)                                 # 시냅스 후: 실제 발화 (전달이 막혀도 발화는 함)
        self._x *= self._dp
        self._y *= self._dm
        pre, post = self._pre, self._post
        self._d += (self.a_plus * (self._x[pre] * F[post]).sum(axis=1)
                    - self.a_minus * (self._y[post] * S[pre]).sum(axis=1)) / self._bn
        self._x += S
        self._y += F

    def run(self, rates=None, seed=None, batch: int = 1):
        out = super().run(rates, seed, batch)
        self.delta = self._d
        return out

    __call__ = run

    def assign(self):
        """변화량을 layer.log_scale.retro에 (-Δ, 쌓임) → rule.step()이 Δ 방향으로 바꿈. 반환: 연결마다 Δ"""
        if self.delta is None:
            raise RuntimeError("먼저 stdp(x, seed=...)로 순전파")
        L = self.layer
        xp = B.xp(L.device)
        g = xp.zeros(L.log_scale.shape, dtype=xp.float32)
        B.scatter_add(g, L.train_which, -self.delta)                       # 연결 종류가 배율을 공유하면 합침
        L.log_scale.retro = g if L.log_scale.retro is None else L.log_scale.retro + g
        d, self.delta = self.delta, None
        return d
