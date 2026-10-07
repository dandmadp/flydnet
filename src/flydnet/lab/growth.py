"""flydnet.lab.growth - [실험적] 추가 연결 (구조적 가소성): 실제 배선(고정)은 그대로 두고, 학습으로 생기고 없어지는
시냅스를 따로 얹음. 엔진(ConnectomeLayer)에는 들어 있지 않고, layer.attach로 붙는 부품 - 인터페이스·기본값이 바뀌거나
없어질 수 있음, 효과는 조건부 (README "실험적 기능")

  from flydnet import lab
  layer = fd.Connectome(mb, "PN", "KC", trainable=["PN>KC"])
  g = lab.Growth(layer, allow=["PN>KC"], budget=5000)  # 층에 붙음 (layer.growth로도 - 이름 name="growth")
  g.grow(500)                                         # 무작위 (허용된 그룹 쌍 안에서)
  g.grow(500, rule="coactive", rates=X)               # 헤브: 대표 입력 X에서 함께 많이 발화한 뉴런 쌍
  g.grow(500, rule="homeostatic", rates=X)            # 항상성: 활동이 그룹 평균보다 낮은 뉴런이 모자란 만큼 새 입력을 받음
  g.grow(500, rule="gradient", loss=lambda L: fd.surprise(head(L(x)), y))   # 세기 0으로 넣었을 때 손실이 가장 줄 곳
  lab.grow_contrastive(g, 500, inputs=X_unlabeled, encoder=enc)              # 정답 없이: 시료끼리 구별되도록
  g.prune(0.1)                                        # 추가 연결 중 약한 10%만 (실제 배선은 절대 안 건드림)
  g.extra_edges()                                     # 표: pre, post, 그룹 쌍, 세기

생물학: 기본 회로는 정해져 있고(타고난 배선), 경험에 따라 시냅스가 덧붙고 없어짐 (구조적 가소성).
  - 세기 = 부호 x w_syn x init_syn x 그 그룹 쌍의 배율(gains) x exp(학습값 log). 처음 세기 init_syn은 기본
    "median" = 그 그룹 쌍 실제 연결의 시냅스 수 중앙값 (버섯체 PN>KC는 10개 - 새 연결이 실제 연결 하나만큼. 예전 기본 1개는
    실제 연결의 1/10이라 추가 연결 4,000개를 합쳐도 PN>KC 입력의 2.7%뿐 → 결과에 거의 영향이 없었음). 숫자면 모든 쌍에 그 수.
    추가 연결 시냅스 1개 = 같은 경로의 실제 시냅스 1개 (calibrate가 gains를 바꾸면 같이)
  - 부호 = 보내는 뉴런의 실제 출력 부호 (데일의 법칙). 출력이 없는 뉴런은 그 그룹의 다수 부호, 그것도 없으면 흥분
  - 허용된 그룹 쌍 안에서만, 이미 있는 연결(실제·추가)·자기 연결은 만들지 않음, 전체 상한 budget, 받는 뉴런마다 상한
    per_neuron (기본 "auto" = ceil(budget / 받을 수 있는 뉴런 수) - 소수 뉴런에 몰리지 않게, None이면 없음).
    최고 성능이 필요하고 데이터가 고르게 퍼진다면 per_neuron=None (냄새 손상 회복 5,400개: auto 0.894, None 0.901 -
    퍼지는지는 np.bincount(g.extra_edges().post)로 확인, 소수 뉴런에 몰리면 상한을 켤 것)
추가 연결은 고정 배선과 같은 지연·같은 효과기(silence·block·activate·mosaic)를 거침. 추가 연결이 없으면 계산이 전과 똑같음.
학습: 역전파(학습값 log, 다른 학습 값과 같은 규칙으로). ThreeFactor·STDP는 고정 배선의 학습 연결만 바꿈.
reach·explain(pathways)·calibrate의 경로 찾기는 고정 배선만 봄.
"""
from __future__ import annotations

import numpy as np

from .. import _check as _C
from ..ganglion import backend as B
from ..ganglion import kernels as K
from ..ganglion import physiology as P
from ..ganglion.signal import Signal, quiescent
from ..ganglion.tissue import Synapse, Tissue


class Growth(Tissue):
    """추가 연결 부품: 설정·목록 (CPU numpy가 원본, 장치에는 배선·세기 버퍼) + 학습값 log + grow·prune. 목록은 배선 순서
    ((post, pre) 정렬). 만들면 layer.attach(name, self)로 층에 붙음 - 엔진은 순전파마다 paths()의 추가 경로를 고정 배선과
    함께 전달하고, 배율이 바뀌면 on_build(), 저장하면 extension_config()로 다시 붙임"""

    def __init__(self, layer, allow=None, budget: int = 10000, per_neuron="auto", init_syn="median", name: str = "growth"):
        super().__init__()
        object.__setattr__(self, "_layer", layer)                      # 층은 자식 조직으로 등록하지 않음 (고리 방지)
        _C.integer("growth budget", budget, lo=0)
        if per_neuron != "auto":
            _C.optional(_C.integer, "growth per_neuron", per_neuron)
        if init_syn != "median":
            _C.pos("growth init_syn", init_syn)
        c = layer.circuit
        self.N = c.N
        names = layer._group_names                                     # 마지막은 "?" (어느 그룹에도 없는 뉴런)
        gid = np.full(c.N, len(names) - 1, np.int64)
        for i, nm in enumerate(names[:-1]):
            gid[c.groups[nm]] = i
        self.gid, self.n_groups = gid, len(names)
        if allow is None:                                              # 실제 배선에 이미 있는 그룹 쌍
            codes = np.unique(layer._edge_code)
        else:
            allow = [allow] if isinstance(allow, str) else list(allow)
            codes = []
            for k in allow:
                cd = layer._pair_code(k)
                if cd is None:
                    raise ValueError(f"growth allow: 회로에 없는 그룹 이름 {k!r} (형식 '보내는>받는', 그룹: {names[:-1][:20]})")
                codes.append(cd)
            codes = np.unique(np.asarray(codes, np.int64))
        self.allow = codes
        self.allow_names = layer._edge_names(codes)
        self.budget, self.per_neuron_arg = int(budget), per_neuron
        # 받는 뉴런마다 추가 연결 상한. "auto" = ceil(budget / 받을 수 있는 뉴런 수) - 입력 수를 고르게 (SRigL의 일정한 fan-in,
        # Lasby et al. 2024). 상한이 없으면 기울기·헤브 규칙이 소수 뉴런에 연결을 몰아 (MNIST에서 KC 상위 10%가 51%) 그
        # 뉴런들이 반응을 독차지 → 희소 부호의 다양성이 무너짐 (MNIST 손상 회복: 대조 학습 0.825 → 상한 3이면 0.851)
        recv = np.unique(np.concatenate([np.nonzero(gid == b)[0] for b in np.unique(codes % self.n_groups)]))             if len(codes) else np.zeros(0, np.int64)
        self.per_neuron = (max(1, int(np.ceil(self.budget / max(len(recv), 1)))) if per_neuron == "auto" else per_neuron)
        self.init_syn = init_syn if init_syn == "median" else float(init_syn)
        # 처음 세기 (시냅스 수) - 그룹 쌍 번호마다. "median": 그 쌍 실제 연결의 |시냅스 수| 중앙값 (실제 연결이 없으면 1)
        self.syn_of = np.full(len(self.allow), 1.0 if init_syn == "median" else self.init_syn, np.float32)   # allow 순서
        if init_syn == "median":
            code = layer._edge_code                                     # 층의 연결 순서 (같은 연결은 합친 시냅스 수)
            w = np.abs(B.numpy(layer.w_syn)).astype(np.float64)          # (예전: 회로 원래 순서의 weight를 이 순서로 읽어 엉뚱한 값)
            for i, cd in enumerate(self.allow):
                m = code == cd
                if m.any():
                    self.syn_of[i] = float(np.median(w[m]))
        # 고정 배선의 연결 (post * N + pre, 정렬) - 겹치지 않게
        self.fixed_keys = np.sort(c.post.astype(np.int64) * c.N + c.pre)
        # 부호: 보내는 뉴런의 실제 출력 시냅스 부호 합 → 없으면 그룹 다수 → 없으면 +1
        s = np.bincount(c.pre, weights=np.sign(c.weight).astype(np.float64), minlength=c.N)
        g = np.bincount(gid[c.pre], weights=np.sign(c.weight).astype(np.float64), minlength=len(names))
        sign = np.where(s != 0, np.sign(s), np.where(g[gid] != 0, np.sign(g[gid]), 1.0))
        self.sign = sign.astype(np.float32)
        self.pre = np.zeros(0, np.int64)
        self.post = np.zeros(0, np.int64)
        self._probe = None                                             # grow(rule="gradient"): 후보 연결 (배선, 값 0)
        self.name = name
        layer.attach(name, self)
        self._set_extra(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))

    # ─────────────── 엔진이 부르는 것 (layer.attach의 규격) ───────────────
    def config(self) -> dict:
        return dict(allow=list(self.allow_names), budget=self.budget, per_neuron=self.per_neuron_arg, init_syn=self.init_syn)

    def extension_config(self) -> dict:
        """저장 파일에 남길 인자 - ConnectomeLayer.load가 같은 인자로 다시 만들어 붙임"""
        return self.config()

    def paths(self) -> list:
        """순전파마다: [(배선, 값 Signal, M, MT)] - 추가 연결, grow(rule="gradient")의 후보. 없으면 []"""
        out = []
        if self.n:
            v = self._values()
            out.append((self.wiring_x, v, *K.matrices(self.wiring_x, v.data)))
        if self._probe is not None:
            wc, vc = self._probe
            out.append((wc, vc, *K.matrices(wc, vc.data)))
        return out

    def on_build(self):
        """층의 배율(gains)이 바뀜 → 추가 연결에도 같은 배율"""
        self._gains()

    def needs_retro(self) -> bool:
        return self.n > 0 or self._probe is not None

    def describe(self) -> str:
        return f"추가 연결 {self.n:,}/{self.budget:,}개"

    def forward(self, *a, **k):
        raise TypeError("Growth는 층에 붙는 부품 - 층을 부를 것 (layer(x))")

    @property
    def n(self) -> int:
        return len(self.pre)

    def keys(self):
        return self.post * self.N + self.pre

    @property
    def syn0(self) -> np.ndarray:
        """지금 추가 연결마다 처음 세기 (시냅스 수, 학습값 0일 때)"""
        return self.syn_of[np.searchsorted(self.allow, self.gid[self.pre] * self.n_groups + self.gid[self.post])]

    # ─────────────── 목록·세기 ───────────────
    def _set_extra(self, pre, post, log):
        """추가 연결 목록을 바꿈 (배선 순서로 정렬, 학습값 log도 같은 순서로). 같은 Synapse 객체를 유지하고
        log.remap에 새 자리마다 예전 번호를 남김 → 가소성 규칙이 살아남은 연결의 관성·적응 상태를 옮기고 새 연결만
        처음부터 (torch 연결 장치는 옵티마이저를 다시 만들 것)"""
        L = self._layer
        old_keys = self.keys()                                                    # 정렬돼 있음 ((post, pre) 순)
        order = self.set(pre, post)
        log = np.asarray(log, np.float32)[order]
        new_keys = self.keys()
        idx = np.full(len(new_keys), -1, np.int64)                              # 새 자리마다 예전 번호 (없으면 -1)
        if len(old_keys):
            pos = np.minimum(np.searchsorted(old_keys, new_keys), len(old_keys) - 1)
            idx = np.where(old_keys[pos] == new_keys, pos, -1)
        dev = L.device
        wx, _ = P.wiring(self.post, self.pre, L.circuit.N, L.circuit.N, device=dev)      # 이미 정렬 → 순서 그대로
        self.buffer("wiring_x", wx, persistent=False)
        self.buffer("sign_x", B.to(self.sign[self.pre] * self.syn0, dev), persistent=False)   # 부호 x 처음 세기(시냅스 수)
        self.buffer("edges_pre", B.to(self.pre.copy(), dev), optional=True)          # 저장용
        self.buffer("edges_post", B.to(self.post.copy(), dev), optional=True)
        if "log" in self._synapses:
            prev = getattr(self.log, "remap", None)                             # 아직 학습 규칙이 안 쓴 앞 변경과 이어 붙임
            if prev is not None and len(prev[1]) == len(old_keys):
                idx = np.where(idx >= 0, prev[1][np.maximum(idx, 0)], -1)
                n_old = prev[0]
            else:
                n_old = len(old_keys)
            self.log.remap = (n_old, idx.astype(np.int64))                       # 가소성 규칙: 살아남은 연결의 관성·적응 상태를
            self.log.data = B.to(log, dev)                                       # 새 자리로 옮김 (새 연결만 처음부터)
            self.log.retro = None
        else:
            self.log = Synapse(log, device=dev)
        self._gains()

    def _gains(self):
        """추가 연결에도 그 그룹 쌍의 배율(gains)을 곱함 - 추가 연결 시냅스 1개 = 같은 경로의 실제 시냅스 1개.
        calibrate가 gains를 바꾸면 추가 연결도 같이 (안 그러면 실제 연결만 커져 추가 연결의 비중이 저절로 바뀜)"""
        L = self._layer
        g = np.ones(self.n, np.float32)
        if self.n and L.gains:
            code = self.gid[self.pre] * self.n_groups + self.gid[self.post]
            for k, v in L.gains.items():
                c = L._pair_code(k)
                g[code == c] *= v
        self.buffer("gain_x", B.to(g, L.device), persistent=False)

    def _values(self):
        """추가 연결의 세기 (mV/스파이크, 부호 포함) Signal - 부호 x 처음 세기(init_syn) x w_syn x exp(log)"""
        base = self.sign_x * self.gain_x * np.float32(self._layer.p["w_syn"])
        return Signal(base) * self.log.clip(-20.0, 20.0).exp()

    def _before_load(self, state: dict, prefix: str):
        """load_state가 모양을 확인하기 전: 파일의 추가 연결 수에 맞춰 목록을 바꿔 둠"""
        if prefix + "edges_pre" not in state:
            return
        L = self._layer
        pre, post = np.asarray(state[prefix + "edges_pre"]), np.asarray(state[prefix + "edges_post"])
        log = np.asarray(state.get(prefix + "log", np.zeros(len(pre), np.float32)))
        if not (len(pre) == len(post) == len(log)):
            raise ValueError(f"추가 연결 상태의 길이가 다름: pre {len(pre)}, post {len(post)}, log {len(log)}")
        if len(pre) and (pre.min() < 0 or post.min() < 0 or max(pre.max(), post.max()) >= L.circuit.N):
            raise ValueError("추가 연결 번호가 회로 밖")
        code = self.gid[pre.astype(np.int64)] * self.n_groups + self.gid[post.astype(np.int64)]
        bad = ~np.isin(code, self.allow)                                  # 다른 allow로 만든 층의 파일 (예전: 처음 세기를
        if bad.any():                                                     # 엉뚱한 그룹 쌍에서 읽어 조용히 다른 세기)
            raise ValueError(f"파일의 추가 연결 {int(bad.sum())}개가 이 층의 허용 쌍({self.allow_names}) 밖: "
                             f"{sorted(set(L._edge_names(np.unique(code[bad]))))} - 만들 때와 같은 allow로")
        key = post.astype(np.int64) * self.N + pre
        if len(np.unique(key)) != len(key):
            raise ValueError("파일의 추가 연결에 같은 (pre, post)가 두 번 있음")
        self._set_extra(pre, post, log)

    # ─────────────── 만들기·없애기 ───────────────
    def grow(self, n: int, rule: str = "random", rates=None, loss=None, candidates: int = 10, seed=None) -> int:
        """추가 연결을 최대 n개 만듦 (허용 쌍·상한 안에서). 반환: 실제로 만든 수
        rule "random":   허용된 그룹 쌍 안에서 고르게
             "coactive": 헤브 - 대표 입력 rates (B, n_in) Hz로 한 번 돌려, 함께 많이 발화한 (pre, post) 쌍
             "homeostatic": 항상성 - 대표 입력 rates로 한 번 돌려, 그룹 평균보다 덜 발화하는 받는 뉴런에 모자란 만큼
                         (보내는 뉴런은 무작위). 정답(라벨)이 필요 없음 - 입력을 잃어 조용해진 뉴런이 시냅스를 새로 만드는 방식
             "gradient": 후보 n x candidates개를 세기 0으로 잠시 넣고 loss(layer) → 값 하나 Signal (줄일 것)의 기울기로,
                         그 부호 방향으로 키우면 손실이 가장 많이 주는 곳 (RigL과 같은 생각, 데일의 법칙 안에서).
                         층의 학습 기울기(.retro)는 건드리지 않음. 정답 없는 대조 학습은 lab.grow_contrastive
        seed: 정수 또는 np.random.Generator (그 흐름을 이어 씀)
        새 연결의 세기는 시냅스 init_syn개 (기본: 그 경로 실제 연결의 중앙값, log = 0)"""
        from ..ganglion.circuitry import _batched
        L = self._layer
        _C.integer("n", n, lo=0)
        if rule not in ("random", "coactive", "homeostatic", "gradient"):
            raise ValueError(f"rule은 'random', 'coactive', 'homeostatic', 'gradient': {rule!r}" +
                             (" - 대조 학습은 lab.grow_contrastive" if rule == "contrastive" else ""))
        need = {"coactive": ("rates", rates), "homeostatic": ("rates", rates), "gradient": ("loss", loss)}.get(rule)
        if need is not None and need[1] is None:                            # 상한이 차서 아무것도 안 만들 때도 알림
            raise ValueError(f"rule='{rule}'는 {need[0]}가 필요" +            # (예전: 0개를 돌려주며 조용히 넘어감)
                             {"rates": " (대표 입력 - 뉴런마다 발화를 재려고)", "loss": ": loss(layer) → 값 하나인 Signal"}[need[0]])
        rng = np.random.default_rng(seed)
        if n == 0 or self.n >= self.budget:
            return 0
        if rule == "random":
            p, q = self.random_candidates(n, rng)
        elif rule in ("coactive", "homeostatic"):
            with quiescent():
                Rr = B.numpy(L(_batched(rates), seed=0 if seed is None else seed, return_all=True).data)
            p, q = self.coactive_candidates(Rr, n) if rule == "coactive" else self.homeostatic_candidates(Rr, n, rng)
        else:
            _C.integer("candidates", candidates)
            cp, cq = self.random_candidates(n * candidates, rng)
            if not len(cp):
                return 0
            wc, order = P.wiring(cq, cp, L.circuit.N, L.circuit.N, device=L.device)
            cp, cq = cp[order], cq[order]
            probe = Signal(B.xp(L.device).zeros(len(cp), dtype=np.float32), plastic=True)
            from ..ganglion.tissue import preserved_retro
            self._probe = (wc, probe)
            try:
                with preserved_retro(L):
                    val = loss(L)
                    if not isinstance(val, Signal) or val.data.size != 1:
                        raise TypeError("loss(layer)는 값 하나인 Signal (줄일 손실)")
                    val.retrograde()
            finally:
                self._probe = None
            g = B.numpy(probe.retro) if probe.retro is not None else np.zeros(len(cp), np.float32)
            gain = -g * self.sign[cp]                                     # 부호 방향으로 키울 때 손실이 주는 정도
            idx = np.argsort(-gain, kind="stable")
            idx = idx[gain[idx] > 0]
            p, q = cp[idx], cq[idx]
        p, q = self._cap(p, q, n)
        if len(p):
            old = B.numpy(self.log.data)
            self._set_extra(np.concatenate([self.pre, p]), np.concatenate([self.post, q]),
                            np.concatenate([old, np.zeros(len(p), np.float32)]))
        return int(len(p))

    def prune(self, frac: float | None = None, below: float | None = None) -> int:
        """추가 연결 중 약한 것을 없앰 (실제 배선은 그대로). frac: 약한 순으로 이 비율 / below: 세기가 시냅스 below개
        미만인 것. 반환: 없앤 수"""
        if (frac is None) == (below is None):
            raise ValueError("frac와 below 중 하나만")
        if frac is not None:                                                 # 값 확인을 먼저 (추가 연결이 없어도 알림)
            _C.unit("frac", frac)
        else:
            _C.nonneg("below", below)
        if self.n == 0:
            return 0
        syn = self.syn0 * np.exp(np.clip(B.numpy(self.log.data), -20, 20))
        if frac is not None:
            k = int(round(frac * self.n))
            drop = np.argsort(syn, kind="stable")[:k]
        else:
            drop = np.nonzero(syn < below)[0]
        keep = np.ones(self.n, bool); keep[drop] = False
        self._set_extra(self.pre[keep], self.post[keep], B.numpy(self.log.data)[keep])
        return int((~keep).sum())

    def extra_edges(self):
        """추가 연결 표: pre, post (회로 번호), pre_id, post_id (root_id), pathway (그룹 쌍), synapses (세기 = 시냅스 수 환산), sign"""
        import pandas as pd
        L = self._layer
        log = B.numpy(self.log.data) if self.n else np.zeros(0, np.float32)
        code = self.gid[self.pre] * self.n_groups + self.gid[self.post]
        return pd.DataFrame({"pre": self.pre, "post": self.post, "pre_id": L.circuit.root_ids[self.pre],
                             "post_id": L.circuit.root_ids[self.post], "pathway": L._edge_names(code) if self.n else [],
                             "synapses": self.syn0 * np.exp(np.clip(log, -20, 20)), "sign": self.sign[self.pre]})

    # ─────────────── 후보 ───────────────
    def _valid(self, pre, post):
        """허용 쌍·자기 연결·이미 있는 연결(고정·추가)·받는 뉴런 상한을 거른 (pre, post), 중복 없이"""
        pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        code = self.gid[pre] * self.n_groups + self.gid[post]
        key = post * self.N + pre
        ok = np.isin(code, self.allow) & (pre != post) & ~_member(key, self.fixed_keys) & ~np.isin(key, self.keys())
        pre, post, key = pre[ok], post[ok], key[ok]
        _, first = np.unique(key, return_index=True)
        first = np.sort(first)
        return pre[first], post[first]

    def _cap(self, pre, post, n):
        """받는 뉴런 상한·전체 상한 안에서 앞에서부터 n개까지 (순서 = 우선순위)"""
        room = self.budget - self.n
        n = min(n, max(room, 0))
        if self.per_neuron is not None:
            have = np.bincount(self.post, minlength=self.N)
            keep, used = [], {}
            for i, q in enumerate(post):
                c = have[q] + used.get(q, 0)
                if c < self.per_neuron:
                    keep.append(i)
                    used[q] = used.get(q, 0) + 1
                    if len(keep) >= n:
                        break
            keep = np.asarray(keep, np.int64)
            return pre[keep], post[keep]
        return pre[:n], post[:n]

    def random_candidates(self, n, rng):
        """허용된 그룹 쌍 안에서 고르게 (쌍마다 가능한 연결 수에 비례) n개까지"""
        members = [np.nonzero(self.gid == g)[0] for g in range(self.n_groups)]
        a, b = self.allow // self.n_groups, self.allow % self.n_groups
        size = np.array([len(members[i]) * len(members[j]) for i, j in zip(a, b)], np.float64)
        if size.sum() == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        pre_all, post_all = [], []
        want, p, q = n, np.zeros(0, np.int64), np.zeros(0, np.int64)
        for _ in range(20):                                           # 거절된 만큼 다시 뽑기
            k = int(want * 2 + 16)
            counts = rng.multinomial(k, size / size.sum())             # 쌍마다 몇 개
            for w in np.nonzero(counts)[0]:
                ma, mb = members[a[w]], members[b[w]]
                pre_all.append(ma[rng.integers(len(ma), size=counts[w])])
                post_all.append(mb[rng.integers(len(mb), size=counts[w])])
            pp, qq = np.concatenate(pre_all), np.concatenate(post_all)
            mix = rng.permutation(len(pp))                              # 쌍 순서가 우선순위가 되지 않게
            p, q = self._valid(pp[mix], qq[mix])
            if len(p) >= n:
                return p[:n], q[:n]
            want = n - len(p)
        return p, q

    def coactive_candidates(self, rates, n):
        """헤브 규칙: 대표 활동 rates (B, N) Hz에서 함께 많이 발화한 (pre, post) 쌍. 점수 = 시료 평균 r_pre·r_post"""
        R = np.asarray(rates, np.float64)
        act = R.mean(0)
        members = [np.nonzero(self.gid == g)[0] for g in range(self.n_groups)]
        m = int(max(32, np.ceil(np.sqrt(8 * max(n, 1)))))
        cand_p, cand_q, cand_s = [], [], []
        for code in self.allow:
            a, b = members[code // self.n_groups], members[code % self.n_groups]
            if not len(a) or not len(b):
                continue
            ta = a[np.argsort(-act[a], kind="stable")[:m]]               # 가장 활발한 보내는·받는 뉴런
            tb = b[np.argsort(-act[b], kind="stable")[:m]]
            S = R[:, ta].T @ R[:, tb] / len(R)                          # (m, m) 함께 발화
            ii, jj = np.nonzero(S > 0)
            cand_p.append(ta[ii]); cand_q.append(tb[jj]); cand_s.append(S[ii, jj])
        if not cand_p:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        p, q, s = np.concatenate(cand_p), np.concatenate(cand_q), np.concatenate(cand_s)
        order = np.lexsort((q, p, -s))                                  # 점수 큰 순 (같으면 번호순 - 결정론적)
        p, q = self._valid(p[order], q[order])                          # _valid는 순서를 지킴
        return p[:n], q[:n]

    def homeostatic_candidates(self, rates, n, rng):
        """항상성 구조 가소성 (정답 없이, 활동만): 받는 뉴런마다 모자란 활동 = max(0, 그 그룹 평균 발화율 - 자기 발화율).
        n개를 모자란 만큼 비례해 나눠 (큰 나머지 방식) 받는 뉴런에 주고, 보내는 뉴런은 허용 쌍의 보내는 그룹에서 무작위.
        입력을 잃어 조용해진 뉴런이 새 시냅스를 만드는 생물의 방식 (Butz & van Ooyen 2013). 순서: 뉴런마다 1번째 연결이
        모두 먼저, 그다음 2번째 … (상한에 걸려도 고르게)"""
        R = np.asarray(rates, np.float64)
        act = R.mean(0)
        members = [np.nonzero(self.gid == g)[0] for g in range(self.n_groups)]
        post_set = {}
        for code in self.allow:
            a, b = members[code // self.n_groups], members[code % self.n_groups]
            if len(a) and len(b):
                post_set.setdefault(int(code % self.n_groups), []).append(a)
        qs, need, pres = [], [], []
        for gb, pre_groups in post_set.items():
            b = members[gb]
            d = np.maximum(act[b].mean() - act[b], 0.0)
            pool = np.unique(np.concatenate(pre_groups))
            qs.append(b); need.append(d); pres += [pool] * len(b)
        if not qs:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        q, d = np.concatenate(qs), np.concatenate(need)
        if d.sum() <= 0:
            return self.random_candidates(n, rng)
        share = n * d / d.sum()
        k = np.floor(share).astype(np.int64)
        rest = n - int(k.sum())
        if rest > 0:
            k[np.argsort(-(share - k), kind="stable")[:rest]] += 1
        P, Q, rank = [], [], []
        for i in np.nonzero(k)[0]:
            pick = rng.choice(pres[i], size=min(len(pres[i]), 2 * k[i] + 4), replace=False)   # 겹치면 걸러지므로 넉넉히
            P.append(pick); Q.append(np.full(len(pick), q[i])); rank.append(np.arange(len(pick)))
        P, Q, rank = np.concatenate(P), np.concatenate(Q), np.concatenate(rank)
        order = np.lexsort((Q, rank))                                   # 뉴런 안 순번이 먼저 (모든 뉴런의 1번째 → 2번째 …)
        p, qq = self._valid(P[order], Q[order])
        want = dict(zip(q[k > 0].tolist(), k[k > 0].tolist()))
        used, keep = {}, []
        for j, t in enumerate(qq.tolist()):                              # 뉴런마다 몫까지만
            if used.get(t, 0) < want.get(t, 0):
                used[t] = used.get(t, 0) + 1
                keep.append(j)
        keep = np.asarray(keep, np.int64)
        p, qq = p[keep], qq[keep]
        r = np.zeros(len(qq), np.int64)                                  # 다시 뉴런 안 순번으로 정렬 (몫을 고르게)
        seen = {}
        for j, t in enumerate(qq.tolist()):
            r[j] = seen.get(t, 0); seen[t] = r[j] + 1
        o = np.argsort(r, kind="stable")
        return p[o], qq[o]

    # ─────────────── 바꾸기 ───────────────
    def set(self, pre, post):
        """목록을 (post, pre) 정렬해 저장. 반환: 정렬 순서 (학습값을 같이 옮기도록)"""
        pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
        order = np.lexsort((pre, post))
        self.pre, self.post = pre[order], post[order]
        return order



def _member(keys, sorted_keys):
    """keys 각각이 정렬된 sorted_keys에 있는지"""
    if not len(sorted_keys):
        return np.zeros(len(keys), bool)
    i = np.searchsorted(sorted_keys, keys)
    i = np.minimum(i, len(sorted_keys) - 1)
    return sorted_keys[i] == keys
