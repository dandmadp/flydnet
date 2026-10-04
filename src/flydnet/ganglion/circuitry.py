"""ConnectomeLayer: 회로 전체를 시간에 따라 시뮬레이션하는 층 (자체 엔진판, torch 없음)

flydnet 0.1의 torch판(flydnet.layers.ConnectomeLayer)과 같은 모델·같은 인자. 차이:
- 신호 배치가 (B, N) (torch판은 (N, B))
- 포아송 입력 난수는 (시드, 스텝, 칸)으로 정해지는 해시 난수 → 체크포인팅으로 다시 계산해도 같고 CPU·GPU 결과도 같음
  (torch판과 같은 시드여도 난수 자체는 다름)
- 저장은 np.savez 한 파일 (회로 배선 포함, FlyWire 데이터 없이 다시 만들 수 있음)

  LIF:    스파이킹 뉴런 (Shiu et al. 2024 매개변수), 대리 기울기로 역전파
  graded: 스파이크 없이 연속값을 전달하는 뉴런 (라미나·메둘라처럼)
"""
from __future__ import annotations

import json

import numpy as np

from . import backend as B
from . import kernels as K
from . import physiology as P
from .signal import Signal, as_signal, checkpoint, concat, learning_enabled, quiescent, where
from .tissue import Synapse, Tissue


def resolve_damp(damp, steps: int, dly: int) -> float:
    """surrogate_damp 값 ("auto"면 홉 수 규칙)"""
    if damp != "auto":
        return float(damp)
    hops = max(steps / max(dly, 1), 1e-9)
    return float(min(1.0, (11.0 / hops) ** 1.5))


def _damped(spk: Signal, d: float) -> Signal:
    """값은 spk 그대로, 역행성 신호만 d배"""
    if d == 1.0 or not spk.plastic:
        return spk
    return spk * d + Signal(spk.data * (1.0 - d))


def _cut(x):
    """역행성 신호 경로를 끊은 같은 값 (구간 절단)"""
    return Signal(x.data) if isinstance(x, Signal) else x


def _registry():
    from ..neurons import REGISTRY
    return REGISTRY


def genetics_effects(layer) -> dict:
    from ..genetics import effects
    return effects(layer)


def genetics_mosaic(layer, seed, batch):
    from ..genetics import mosaic_mask
    return mosaic_mask(layer, seed, batch)

DEFAULT_PARAMS = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,   # mV
    t_mbr=20.0, tau=5.0,                  # ms
    t_rfc=2.2, t_dly=1.8,                 # ms
    w_syn=0.275,                          # mV / synapse
    f_poi=250,                            # Poisson 입력 1회 = w_syn*f_poi mV
    dt=0.1,                               # ms
)


class ConnectomeLayer(Tissue):
    """입력 그룹 뉴런에 발화율 (B, n_in) Hz 또는 (B, T, n_in)을 넣고, 출력 그룹 뉴런의 발화율 (B, n_out) Hz를 돌려줌

    gains:      {"PN>KC": 1.5, ...} 연결 종류별 세기 배율
    input_mode: "poisson" = 무작위 스파이크 / "regular" = 같은 발화율의 일정 간격 스파이크 (같은 입력 → 같은 반응)
                두 모드 모두 입력 발화율로 미분 가능 (straight-through)
    trainable:  False / True (모든 연결) / ["KC>MBON", ...] (그 종류만). 세기 = 원래 × exp(log_scale), 부호 유지
    share:      "edge" = 연결마다 / "pair" = 연결 종류마다 배율 하나
    dt, slope:  시간 간격 ms / 대리 기울기의 날카로움
    checkpoint_every: n이면 n스텝 구간마다 중간 상태를 버리고 역전파 때 다시 계산 (메모리 절약)
    bias, t_mbr, train_neurons: 세포 유형(그룹)별 휴지 전위 더하기(mV)·막 시간 상수(ms)와 그 학습
    v_init:     "rest" / "random" (리셋~문턱 사이 고정된 무작위 값)
    count_from_ms: 이 시각부터 스파이크(또는 활동)를 셈
    neuron:     "lif" (기본) / "graded"
    inputs=None: 입력 그룹 없이 (fd.genetics.activate로만 자극할 때). 그때 layer(None, batch=시행 수)
    surrogate_damp: 스파이크의 대리 기울기에 곱하는 계수. 값은 그대로, 역전파만 줄임 → 되먹임 회로를 돌며 기울기가
                커지는 것을 막음. 기본 "auto" = min(1, (11 / 홉 수)^1.5), 홉 수 = 시뮬레이션 스텝 / 시냅스 지연 스텝
                (신호가 시냅스를 건널 수 있는 횟수). 짧으면 감쇠 없음, 길수록 강하게 - 초파리 버섯체·전체 뇌·합성
                그래프 5가지 조건에서 fd.gradcheck로 가장 잘 맞은 값을 맞추는 경험 규칙 (validation/gradients).
                timing="legacy"는 1 (0.1.15와 같음). 자기 손실로 확인·조정: fd.gradcheck, fd.tune_surrogate
    noise:      막전위 잡음 표준편차 mV/스텝 (내장 LIF, 적분 중인 뉴런만, 시드로 정해짐). 출력이 연결 세기에 대해
                매끄러워지고 대리 기울기가 '잡음 있는 뉴런의 발화 확률의 기울기'에 가까워짐 (Gygax & Zenke 2024)
    truncate:   n이면 n스텝마다 상태의 역전파 연결을 끊음 (구간 절단 역전파, truncated BPTT). 발화율 합은 끊지 않음
                → 각 스텝의 기울기가 최대 n스텝 과거까지만. 긴 시뮬레이션에서 기울기가 부푸는 것을 막음
    timing:     "brian" (기본, 0.1.16~) = Shiu et al. 2024 Brian2 모델과 같은 한 스텝 (정확한 선형 적분, 불응기 중 도착한
                시냅스 입력은 버림, 입력 스파이크는 발화 판정 뒤, 불응기 2.2 ms = 22스텝). 같은 입력이면 스파이크 시각까지 같음
                "legacy" = 0.1.15까지 (오일러 적분, 불응기 중 입력을 쌓아 둠, torch판과 같음). 0.1.15 저장 파일은 legacy로 읽힘
    """

    FORMAT = "flydnet.ganglion.ConnectomeLayer/1"

    def __init__(self, circuit, inputs="PN", outputs=("KC",), t_ms: float = 100.0, params: dict | None = None,
                 gains: dict | None = None, device: str | None = None, input_mode: str = "poisson",
                 trainable=False, dt: float | None = None, slope: float = 10.0, checkpoint_every: int | None = None,
                 share: str = "edge", bias=None, t_mbr=None, train_neurons: bool = False, v_init: str = "rest",
                 count_from_ms: float = 0.0, neuron: str = "lif", timing: str = "brian",
                 surrogate_damp: float | str | None = None, truncate: int | None = None, noise: float = 0.0):
        super().__init__()
        listify = lambda x: x if (x is None or isinstance(x, str)) else list(x)
        self.config = dict(inputs=listify(inputs), outputs=listify(outputs), t_ms=t_ms, params=dict(params or {}),
                           input_mode=input_mode,
                           trainable=trainable if isinstance(trainable, bool) else list(trainable),
                           dt=dt, slope=slope, checkpoint_every=checkpoint_every, share=share,
                           bias=dict(bias) if isinstance(bias, dict) else bias,
                           t_mbr=dict(t_mbr) if isinstance(t_mbr, dict) else t_mbr,
                           train_neurons=train_neurons, v_init=v_init, count_from_ms=count_from_ms, neuron=neuron,
                           timing=timing, surrogate_damp=surrogate_damp, truncate=truncate, noise=noise)
        from ..neurons import NeuronModel, from_config
        if isinstance(neuron, dict):                                     # 저장 파일에서 (사용자 정의 뉴런 모델)
            neuron = from_config(neuron)
        if isinstance(neuron, NeuronModel):
            if bias is not None or t_mbr is not None or train_neurons:
                raise ValueError("사용자 정의 뉴런 모델에서는 bias·t_mbr·train_neurons를 쓰지 않음 (모델 매개변수로)")
            self.config["neuron"] = neuron.config() if type(neuron).__name__ in _registry() else None
        elif neuron not in ("lif", "graded"):
            raise ValueError(f"neuron은 'lif', 'graded', 또는 fd.neurons.NeuronModel: {neuron}")
        if timing not in ("brian", "legacy"):
            raise ValueError(f"timing은 'brian' 또는 'legacy': {timing}")
        if surrogate_damp is None:                       # 기본: "auto" (홉 수 규칙), 예전 방식(legacy)은 1 (0.1.15와 같게)
            surrogate_damp = 1.0 if timing == "legacy" and isinstance(neuron, str) else "auto"
        if surrogate_damp != "auto":
            surrogate_damp = float(surrogate_damp)
            if not 0 < surrogate_damp <= 1:
                raise ValueError(f"surrogate_damp는 0 초과 1 이하 또는 'auto': {surrogate_damp}")
        self.config["surrogate_damp"] = surrogate_damp
        if truncate is not None and truncate < 1:
            raise ValueError(f"truncate는 1 이상의 스텝 수: {truncate}")
        self.surrogate_damp, self.truncate = surrogate_damp, truncate
        if noise < 0:
            raise ValueError(f"noise는 0 이상 (mV): {noise}")
        self.noise = float(noise)
        if truncate is not None:
            _dt = dt if dt is not None else DEFAULT_PARAMS["dt"]
            _dly = max(int(round(dict(DEFAULT_PARAMS, **(params or {}))["t_dly"] / _dt)), 1)
            if truncate <= _dly:
                raise ValueError(f"truncate({truncate})는 시냅스 지연 {_dly}스텝보다 커야 함 - 아니면 이동 중인 스파이크가 "
                                 f"늘 끊겨 연결 세기의 기울기가 모두 사라짐 (예: truncate={_dly * 5})")
        self.timing = timing
        if input_mode not in ("poisson", "regular"):
            raise ValueError(f"input_mode는 'poisson' 또는 'regular': {input_mode!r}")
        if v_init not in ("rest", "random"):
            raise ValueError(f"v_init은 'rest' 또는 'random': {v_init!r}")
        if share not in ("edge", "pair"):
            raise ValueError(f"share는 'edge'(연결마다) 또는 'pair'(연결 종류마다): {share!r}")
        if not t_ms > 0:
            raise ValueError(f"t_ms는 양수 (ms): {t_ms}")
        if not 0 <= count_from_ms < t_ms:
            raise ValueError("count_from_ms는 0 이상 t_ms 미만")
        if t_ms < (dt if dt is not None else DEFAULT_PARAMS["dt"]):
            raise ValueError(f"t_ms({t_ms})가 시간 간격 dt보다 짧아 시뮬레이션이 한 스텝도 안 됨")
        dev = B.check(device) if device is not None else B.default_device()
        for g in ([inputs] if isinstance(inputs, str) else list(inputs or [])) + \
                 ([outputs] if isinstance(outputs, str) else list(outputs)):
            if g not in circuit.groups:
                raise KeyError(f"회로에 없는 그룹: {g!r} (있는 것: {list(circuit.groups)[:20]})")
        _ins = set([inputs] if isinstance(inputs, str) else list(inputs or []))
        _outs = set([outputs] if isinstance(outputs, str) else list(outputs))
        if _ins & _outs:
            import warnings
            warnings.warn(f"입력 그룹과 출력 그룹이 겹침 ({sorted(_ins & _outs)}): 그 뉴런의 출력은 넣은 입력 발화율 그대로임",
                          stacklevel=2)
        self.neuron, self.input_mode, self.v_init, self.share = neuron, input_mode, v_init, share
        self.circuit = circuit
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        if dt is not None:
            self.p["dt"] = dt
        self.t_ms, self.slope, self.checkpoint_every, self.count_from_ms = t_ms, slope, checkpoint_every, count_from_ms
        inputs = [] if inputs is None else [inputs] if isinstance(inputs, str) else list(inputs)
        outputs = [outputs] if isinstance(outputs, str) else list(outputs)
        self.in_names, self.out_names = inputs, outputs
        N = circuit.N
        in_idx = np.concatenate([circuit.groups[i] for i in inputs]) if inputs else np.array([], np.int64)
        out_idx = np.concatenate([circuit.groups[o] for o in outputs])
        self.n_in, self.n_out = len(in_idx), len(out_idx)

        # 연결: (post, pre) 정렬, 같은 연결은 시냅스 수 합치기 (torch판과 같은 순서)
        key = circuit.post.astype(np.int64) * N + circuit.pre
        uniq, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=circuit.weight.astype(np.float64), minlength=len(uniq))
        post, pre = uniq // N, uniq % N
        wr, _ = P.wiring(post, pre, N, N, device=dev)              # 이미 정렬되어 순서 그대로
        names = list(circuit.groups)
        gid = np.full(N, len(names), np.int64)
        for i, nm in enumerate(names):
            gid[circuit.groups[nm]] = i
        self._group_names = names + ["?"]
        code = gid[pre] * (len(names) + 1) + gid[post]             # 연결 종류 번호 (문자열은 필요할 때만)
        self._edge_code = code

        if trainable is True:
            pos = np.arange(len(uniq))
        elif trainable:
            want = {self._pair_code(k) for k in trainable}
            if None in want or not np.isin(list(want), code).all():
                raise ValueError(f"회로에 없는 연결 종류: {sorted(set(trainable) - set(self._edge_names(np.unique(code))))}")
            pos = np.nonzero(np.isin(code, list(want)))[0]
        else:
            pos = np.array([], np.int64)
        if share == "pair":
            pair_codes, which = np.unique(code[pos], return_inverse=True)
            names_ = np.array(self._edge_names(pair_codes), dtype=object)
            order = np.argsort(names_, kind="stable")              # 이름순 (torch판·저장 파일과 같은 순서)
            rank = np.empty_like(order); rank[order] = np.arange(len(order))
            which = rank[which]
            self.train_pairs = list(names_[order])
        else:
            which = np.arange(len(pos))
        xp = B.xp(dev)
        self.buffer("wiring", wr, persistent=False)
        self.buffer("w_syn", B.to(w.astype(np.float32), dev), persistent=False)
        self.buffer("in_idx", B.to(in_idx, dev), persistent=False)
        self.buffer("out_idx", B.to(out_idx, dev), persistent=False)
        self.buffer("train_pos", B.to(pos, dev), persistent=False)
        self.buffer("train_which", B.to(which.astype(np.int64), dev), persistent=False)
        self.buffer("phase0", B.to(np.random.default_rng(0).random(self.n_in).astype(np.float32), dev), persistent=False)
        self.buffer("v_frac", B.to(np.random.default_rng(1).random(N).astype(np.float32), dev), persistent=False)
        self.buffer("wiring_id", np.array([N, len(uniq), int(post.sum() % (2 ** 61 - 1)),
                                           int((post * 31 + pre * 17).sum() % (2 ** 61 - 1))], np.int64))
        self.log_scale = Synapse(np.zeros(int(which.max()) + 1 if len(pos) else 0, np.float32), device=dev) \
            if len(pos) else None
        self.gains = dict(gains or {})
        self._effects = []                                         # fd.genetics 효과기 (저장 안 됨)
        self._probe = {}                                           # fd.explain 탐침: "neuron" (N, 1), "edge" (E,) Signal
        self._observer = None                                      # fd.ThreeFactor: 스텝마다 상태를 받는 관찰자

        self.neuron_params = bias is not None or t_mbr is not None or train_neurons
        if self.neuron_params:
            self.group_names = names
            self.buffer("group_idx", B.to(np.minimum(gid, len(names)), dev), persistent=False)
            per = lambda v, default: np.array([float(v.get(n, default) if isinstance(v, dict) else
                                                     (default if v is None else v)) for n in names + ["_"]], np.float32)
            b0, t0 = per(bias, 0.0), np.log(per(t_mbr, self.p["t_mbr"]))
            if train_neurons:
                self.bias, self.log_t_mbr = Synapse(b0, device=dev), Synapse(t0, device=dev)
            else:
                self.buffer("bias", B.to(b0, dev)); self.buffer("log_t_mbr", B.to(t0, dev))
        self._build()

    # ─────────────── 연결 종류 ───────────────
    def _pair_code(self, name: str):
        a, _, b = name.partition(">")
        if a not in self._group_names or b not in self._group_names:
            return None
        return self._group_names.index(a) * len(self._group_names) + self._group_names.index(b)

    def _edge_names(self, codes) -> list:
        n = len(self._group_names)
        return [f"{self._group_names[c // n]}>{self._group_names[c % n]}" for c in np.asarray(codes)]

    @property
    def edge_key(self) -> np.ndarray:
        """연결마다 종류 이름 (예: "KC>MBON"), 연결 순서"""
        uniq, inv = np.unique(self._edge_code, return_inverse=True)
        return np.array(self._edge_names(uniq), dtype=object)[inv]

    @property
    def trainable(self) -> bool:
        return self.log_scale is not None

    @property
    def device(self) -> str:
        return B.device_of(self.w_syn)

    def _build(self):
        """고정 부분: 원래 세기 × 배율 × mV/시냅스 (± 시냅스 수)"""
        g = np.ones(len(self._edge_code), np.float32)
        for k, v in self.gains.items():
            c = self._pair_code(k)
            if c is None or not (self._edge_code == c).any():
                raise ValueError(f"회로에 없는 연결 종류: {k}")
            g[self._edge_code == c] *= v
        self.w_base = self.w_syn * B.to(g, self.device) * self.p["w_syn"]

    def _moved(self, device):
        self._build()

    def set_gain(self, key: str, value: float):
        self.gains[key] = value
        self._build()

    def values(self) -> Signal:
        """현재 연결별 세기 (mV/스파이크, 부호 포함), 연결 순서"""
        base = Signal(self.w_base)
        if not self.trainable:
            return base
        ones = B.xp(self.device).ones(len(self.w_base), dtype=self.w_base.dtype)
        scale = P.put(ones, self.train_pos, self.log_scale.exp()[self.train_which])
        return base * scale

    def weights(self) -> np.ndarray:
        with quiescent():
            return self.values().numpy()

    def scale_of(self) -> dict:
        if self.share != "pair" or not self.trainable:
            raise ValueError("share='pair'로 학습하는 층에서만")
        return dict(zip(self.train_pairs, np.exp(self.log_scale.numpy()).tolist()))

    def neuron_table(self):
        import pandas as pd
        if not self.neuron_params:
            raise ValueError("뉴런 매개변수(bias/t_mbr/train_neurons)를 켠 층에서만")
        b = self.bias.numpy() if isinstance(self.bias, Signal) else B.numpy(self.bias)
        t = self.log_t_mbr.numpy() if isinstance(self.log_t_mbr, Signal) else B.numpy(self.log_t_mbr)
        return pd.DataFrame({"bias_mV": b[:-1], "t_mbr_ms": np.exp(t[:-1])}, index=self.group_names)

    def _neuron_terms(self, dt):
        """그룹별 휴지 전위 더하기 (N,)와 적분 비율 a = dt/τ (N,)"""
        bias = self.bias if isinstance(self.bias, Signal) else Signal(self.bias)
        log_t = self.log_t_mbr if isinstance(self.log_t_mbr, Signal) else Signal(self.log_t_mbr)
        b = bias[self.group_idx]
        a = (dt / log_t.exp()[self.group_idx]).clip(hi=1.0)
        return b, a

    def _needs_retro(self, x: Signal) -> bool:
        return learning_enabled() and (self.trainable or x.plastic or bool(self._probe)
                                       or (self.neuron_params and isinstance(self.bias, Synapse)))

    @staticmethod
    def _frames(x: Signal):
        """입력 (B, n_in) 또는 (B, T, n_in) → 스텝 s의 (n_in, B) 신호를 주는 함수"""
        if x.ndim == 2:
            xt = x.T
            return lambda s, steps: xt
        if x.ndim == 3:
            T = x.shape[1]
            xt = x.transpose(1, 2, 0)                                      # (T, n_in, B)
            return lambda s, steps: xt[s * T // steps]
        raise ValueError("입력은 (B, n_in) 또는 (B, T, n_in)")

    # ─────────────── 순전파 ───────────────
    # 내부 신호 배치는 (뉴런 N, 배치 B). 한 스텝은 kernels.lif_step / graded_step 한 번 (필요한 값만 저장)
    def forward(self, rates=None, seed: int | None = None, return_all: bool = False, record=None, batch: int = 1):
        """record: 회로 뉴런 번호 목록이면 스텝별 활동도 → (출력, (B, steps, k) numpy). 체크포인팅과 같이 쓰지 않음
        rates=None: 입력 없음 (모두 0), batch개 시행 - fd.genetics.activate로만 자극할 때"""
        if rates is None:
            rates = np.zeros((batch, self.n_in), np.float32)
        if seed is not None and not (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)):
            raise TypeError(f"seed는 정수: {seed!r}")
        if np.ndim(rates.data if isinstance(rates, Signal) else rates) == 1:   # 시료 하나 (n_in,) → 출력도 (n_out,)
            one = rates[None] if isinstance(rates, Signal) else np.asarray(rates)[None]
            res = self.forward(one, seed=seed, return_all=return_all, record=record)
            return (res[0][0], res[1][0]) if isinstance(res, tuple) else res[0]
        if record is not None:
            r = np.asarray(record)
            if r.size and (r.min() < 0 or r.max() >= self.circuit.N):
                raise IndexError(f"record의 뉴런 번호는 0 ~ {self.circuit.N - 1}: 범위 밖 {r[(r < 0) | (r >= self.circuit.N)][:5].tolist()}")
        x = as_signal(rates, self.device)
        if x.data.dtype != np.float32 and x.data.dtype.kind == "f" and not x.plastic:
            x = Signal(x.data.astype(np.float32))
        if x.shape[-1] != self.n_in:
            raise ValueError(f"입력 마지막 차원 {x.shape[-1]} ≠ 입력 뉴런 {self.n_in}")
        if x.data.size and not bool(B.xp(self.device).isfinite(x.data).all()):
            raise ValueError("입력에 NaN·무한대가 있음 (그대로 두면 스파이크가 안 생겨 출력이 조용히 0이 됨)")
        if self.neuron != "graded" and x.data.size and bool((x.data < 0).any()):
            raise ValueError(f"입력 발화율은 0 이상 (Hz): 최소 {float(x.data.min()):.3g} - 음수는 스파이크가 안 생겨 조용히 0이 됨")
        if self.neuron == "graded":
            return self._forward_graded(x, return_all, record, seed)
        if not isinstance(self.neuron, str):
            return self._forward_custom(x, return_all, record, seed)
        p, N, xp = self.p, self.circuit.N, B.xp(self.device)
        Bn = x.shape[0]
        dt = p["dt"]; steps = int(round(self.t_ms / dt))
        dly = max(int(round(p["t_dly"] / dt)), 1); R = dly + 1
        rfc = int(round(p["t_rfc"] / dt))
        gd = float(np.exp(-dt / p["tau"]))
        if self.neuron_params:
            b, a = self._neuron_terms(dt)
            v_eq, a = (b + p["v_0"]).reshape(-1, 1), a.reshape(-1, 1)
        else:
            v_eq, a = p["v_0"], dt / p["t_mbr"]
        brian = self.timing == "brian"
        if brian:                                   # 정확한 선형 적분 계수 (a = dt/t_mbr). t_mbr = tau면 극한값
            tau = p["tau"]
            if isinstance(a, Signal):
                e_v = (-a).exp()
                den = a * tau - dt
                den = where(den.data >= 0, den + 1e-6, den - 1e-6)
                e_g = (a * tau / den) * (e_v * -1.0 + gd)
            else:
                e_v = float(np.exp(-a))
                e_g = float(a * np.exp(-a)) if abs(a * tau - dt) < 1e-9 else float(a * tau / (a * tau - dt) * (gd - e_v))
        frame = self._frames(x * (dt / 1000.0))                            # 스텝당 입력 스파이크 확률 (n_in, B)
        s_cnt = int(round(self.count_from_ms / dt))
        rfc_vec = xp.full((N, 1), float(rfc), dtype=xp.float32); rfc_vec[self.in_idx] = 0
        values = self.values()
        if "edge" in self._probe:                                          # fd.explain: 연결마다 배율 탐침 (값 1)
            values = values * self._probe["edge"]
        probe_n = self._probe.get("neuron")
        M, MT = K.matrices(self.wiring, values.data)                       # 순전파당 한 번만
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        in_idx, regular = self.in_idx, self.input_mode == "regular"
        consts = dict(gd=gd, poi_w=p["w_syn"] * p["f_poi"], v_th=p["v_th"], v_rst=p["v_rst"],
                      scale=p["v_th"] - p["v_rst"], slope=self.slope)
        rec = [] if record is not None else None
        rec_idx = B.to(np.asarray(record), self.device) if record is not None else None
        fx = genetics_effects(self)
        quiet, blocked, act_idx = fx.get("silence"), fx.get("block"), fx.get("act_idx")
        n_act = 0 if act_idx is None else len(act_idx)
        drop = genetics_mosaic(self, seed, Bn)                             # 세포 유형 드롭아웃 (학습 중에만)
        if n_act:                                   # 활성화 = 포아송 자극을 입력처럼 전위에 직접, 불응기 없음 (Shiu et al.과 같음)
            idx_all = xp.concatenate([in_idx, act_idx])
            rfc_vec[act_idx] = 0
            p_act = Signal(xp.broadcast_to(fx["act_hz"] * np.float32(dt / 1000.0), (n_act, Bn)).copy())
            act_seed = (int(seed) * 0x2545F4914F6CDD1D + 0x5EED) % (1 << 63)   # 입력 난수와 따로

        # 발화한 스텝 뒤 적분이 멈추는 스텝 수: brian = rfc (Brian2: t - 마지막 발화 >= 2.2 ms면 다시 적분), legacy = rfc + 1
        rfc_set = rfc_vec - 1 if brian else rfc_vec
        obs = self._observer
        if obs is not None:
            if not brian:
                raise ValueError("관찰자(fd.ThreeFactor)는 timing='brian'에서만")
            gain = None                                                    # 발화에 곱해지는 것들 (끄기·탐침·드롭아웃)
            for m in (quiet, probe_n.data if probe_n is not None else None, drop):
                if m is not None:
                    gain = m if gain is None else gain * m
            obs.begin(dict(e_v=e_v, e_g=e_g, gd=gd, dly=dly, steps=steps, s_cnt=s_cnt, scale=consts["scale"],
                           slope=self.slope, damp=resolve_damp(self.surrogate_damp, steps, dly), batch=Bn, gain=gain,
                           blocked=blocked,
                           rate_c=1000.0 / (self.t_ms - s_cnt * dt)))

        damp, trunc, sigma = resolve_damp(self.surrogate_damp, steps, dly), self.truncate, self.noise
        noise_seed = (int(seed) * 0x9E3779B97F4A7C15 + 0x0015E) % (1 << 63)

        def run(s0, s1, V, G, refr, counts, phase, *buf):
            buf = list(buf)
            for s in range(s0, s1):
                if trunc and s and s % trunc == 0:                           # 구간 절단: 상태의 과거 경로를 끊음
                    V, G, buf = _cut(V), _cut(G), [_cut(b) for b in buf]
                ps = frame(s, steps)
                act = refr.data <= 0
                if sigma:                                                    # 막전위 잡음 (값만, 기울기는 그대로 통과)
                    z = P.hash_uniform(xp, noise_seed, s, (Bn, N, 2))
                    gauss = xp.sqrt(-2 * xp.log(1 - z[..., 0])) * xp.cos(2 * np.pi * z[..., 1])
                    Vd = V.data + (gauss.T * sigma * act).astype(xp.float32)
                    V = Signal(Vd)._link((V,), lambda g: (g,))
                if regular:
                    ph = phase.data + ps.data
                    spikes = (ph >= 1).astype(ph.dtype)
                    phase = Signal(ph - spikes)
                else:
                    spikes = (P.hash_uniform(xp, seed, s, ps.shape[::-1]).T < ps.data).astype(ps.data.dtype)
                idx = in_idx
                if n_act:
                    kick = P.hash_uniform(xp, act_seed, s, (Bn, n_act)).T < p_act.data
                    spikes = xp.concatenate([spikes, kick.astype(spikes.dtype)])
                    ps, idx = concat([ps, p_act]), idx_all
                if brian:
                    uu = [] if obs is not None else None
                    V, G, spk = K.lif_step_brian(V, G, buf[s % R], ps, spikes, act, idx, v_eq, e_v, e_g, out_u=uu,
                                                 **consts)
                else:
                    V, G, spk = K.lif_step(V, G, buf[s % R], ps, spikes, act, idx, v_eq, a, **consts)
                spk = _damped(spk, damp)
                if quiet is not None:                                      # Kir2.1: 발화 없음
                    spk = spk * quiet
                if probe_n is not None:                                    # fd.explain: 뉴런마다 배율 탐침 (값 1)
                    spk = spk * probe_n
                if drop is not None:
                    spk = spk * drop
                fired = spk.data > 0
                if s >= s_cnt:
                    counts = counts + spk
                if rec is not None:
                    rec.append(spk.data[rec_idx].T.copy())
                refr = Signal(xp.where(fired, rfc_set, refr.data - 1))
                sent = spk if blocked is None else spk * blocked           # Shibire: 발화는 하지만 전달 없음
                buf[(s + dly) % R] = K.propagate(sent, values, M, MT, self.wiring)
                if obs is not None:
                    obs.step(s, sent.data, u=uu[0], act=act)
            return (V, G, refr, counts, phase, *buf)

        z = lambda: Signal(xp.zeros((N, Bn), dtype=xp.float32))
        if self.v_init == "random":
            V0 = Signal(xp.broadcast_to((p["v_rst"] + self.v_frac * (p["v_th"] - p["v_rst"]))[:, None], (N, Bn)).copy())
        else:
            V0 = Signal(xp.full((N, Bn), p["v_0"], dtype=xp.float32))
        phase0 = Signal(xp.broadcast_to(self.phase0[:, None], (self.n_in, Bn)).copy())
        state = (V0, z(), z(), z(), phase0, *[z() for _ in range(R)])
        state = self._run(run, state, steps, x, record)
        rate = state[3].T * (1000.0 / (self.t_ms - s_cnt * dt))            # (B, N) Hz
        out = rate if return_all else rate[:, self.out_idx]
        return (out, self._trace(rec)) if record is not None else out

    def _run(self, run, state, steps, x, record):
        """구간 나눠 실행 (학습 중이고 checkpoint_every면 구간마다 다시 계산)"""
        ce = self.checkpoint_every
        if ce and self._needs_retro(x):
            if record is not None:
                raise ValueError("record는 체크포인팅과 같이 쓸 수 없음 (quiescent()에서 쓰기)")
            for s0 in range(0, steps, ce):
                state = checkpoint(lambda *st, s0=s0: run(s0, min(s0 + ce, steps), *st), *state)
            return state
        return run(0, steps, *state)

    def _trace(self, rec):
        return B.numpy(B.xp(self.device).stack(rec, axis=1))              # (B, steps, k)

    def _forward_custom(self, x: Signal, return_all: bool, record, seed):
        """사용자 정의 뉴런 모델 (fd.neurons): LIF 순전파와 같은 틀 (입력·지연·genetics·탐침·드롭아웃·기록·관찰자),
        한 스텝만 model.step - Signal 연산이라 역전파는 자동 미분으로"""
        from ..neurons import Context
        model, p, N, xp = self.neuron, self.p, self.circuit.N, B.xp(self.device)
        Bn = x.shape[0]
        dt = p["dt"]; steps = int(round(self.t_ms / dt))
        dly = max(int(round(p["t_dly"] / dt)), 1); R = dly + 1
        poi_w = p["w_syn"] * p["f_poi"]
        frame = self._frames(x * (dt / 1000.0))
        s_cnt = int(round(self.count_from_ms / dt))
        values = self.values()
        if "edge" in self._probe:
            values = values * self._probe["edge"]
        probe_n = self._probe.get("neuron")
        M, MT = K.matrices(self.wiring, values.data)
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        in_idx, regular = self.in_idx, self.input_mode == "regular"
        rec = [] if record is not None else None
        rec_idx = B.to(np.asarray(record), self.device) if record is not None else None
        fx = genetics_effects(self)
        quiet, blocked, act_idx = fx.get("silence"), fx.get("block"), fx.get("act_idx")
        n_act = 0 if act_idx is None else len(act_idx)
        drop = genetics_mosaic(self, seed, Bn)
        idx_all = in_idx
        if n_act:
            idx_all = xp.concatenate([in_idx, act_idx])
            p_act = Signal(xp.broadcast_to(fx["act_hz"] * np.float32(dt / 1000.0), (n_act, Bn)).copy())
            act_seed = (int(seed) * 0x2545F4914F6CDD1D + 0x5EED) % (1 << 63)
        ctx = Context(xp, N, Bn, dt, idx_all, p)
        init = model.init(ctx)
        names = tuple(model.state)
        if set(init) != set(names):
            raise ValueError(f"{type(model).__name__}.init이 돌려준 상태 {sorted(init)} ≠ state {sorted(names)}")
        obs = self._observer
        if obs is not None:
            gain = None
            for m in (quiet, probe_n.data if probe_n is not None else None, drop):
                if m is not None:
                    gain = m if gain is None else gain * m
            obs.begin(dict(dly=dly, steps=steps, s_cnt=s_cnt, batch=Bn, gain=gain, blocked=blocked, dt=dt,
                           rate_c=1000.0 / (self.t_ms - s_cnt * dt), model=model))
        zeros = xp.zeros((N, Bn), dtype=xp.float32)

        damp, trunc = resolve_damp(self.surrogate_damp, steps, dly), self.truncate

        def run(s0, s1, counts, phase, *rest):
            st = dict(zip(names, rest[:len(names)]))
            buf = list(rest[len(names):])
            for s in range(s0, s1):
                if trunc and s and s % trunc == 0:
                    st, buf = {k: _cut(v) for k, v in st.items()}, [_cut(b) for b in buf]
                ps = frame(s, steps)
                if regular:
                    ph = phase.data + ps.data
                    spikes = (ph >= 1).astype(ph.dtype)
                    phase = Signal(ph - spikes)
                else:
                    spikes = (P.hash_uniform(xp, seed, s, ps.shape[::-1]).T < ps.data).astype(ps.data.dtype)
                if n_act:
                    kick = P.hash_uniform(xp, act_seed, s, (Bn, n_act)).T < p_act.data
                    spikes = xp.concatenate([spikes, kick.astype(spikes.dtype)])
                    ps = concat([ps, p_act])
                # 외부 자극: 스파이크 하나 = poi_w mV, 역전파는 확률로 (straight-through, 내장 LIF와 같음)
                I_ext = P.put(zeros, idx_all, ps * poi_w + Signal((spikes - ps.data) * poi_w)) if len(idx_all) \
                    else Signal(zeros)
                st, spk = model.step(st, buf[s % R], I_ext, ctx)
                spk = _damped(spk, damp)
                for m in (quiet, probe_n, drop):
                    if m is not None:
                        spk = spk * m
                if s >= s_cnt:
                    counts = counts + spk
                if rec is not None:
                    rec.append(spk.data[rec_idx].T.copy())
                sent = spk if blocked is None else spk * blocked
                buf[(s + dly) % R] = K.propagate(sent, values, M, MT, self.wiring)
                if obs is not None:
                    obs.step(s, sent.data)
            return (counts, phase, *[st[k] for k in names], *buf)

        z = lambda: Signal(xp.zeros((N, Bn), dtype=xp.float32))
        phase0 = Signal(xp.broadcast_to(self.phase0[:, None], (self.n_in, Bn)).copy())
        state = (z(), phase0, *[init[k] for k in names], *[z() for _ in range(R)])
        state = self._run(run, state, steps, x, record)
        rate = state[0].T * (1000.0 / (self.t_ms - s_cnt * dt))
        out = rate if return_all else rate[:, self.out_idx]
        return (out, self._trace(rec)) if record is not None else out

    def _forward_graded(self, x: Signal, return_all: bool, record, seed=None):
        p, N, xp = self.p, self.circuit.N, B.xp(self.device)
        Bn = x.shape[0]
        dt = p["dt"]; steps = int(round(self.t_ms / dt))
        r_max = p.get("r_max", 10.0)
        frame = self._frames(x)
        s_cnt = int(round(self.count_from_ms / dt))
        if self.neuron_params:
            b, a = self._neuron_terms(dt)
            b, a = b.reshape(-1, 1), a.reshape(-1, 1)
        else:
            b, a = 0.0, dt / p["t_mbr"]
        values = self.values()
        if "edge" in self._probe:                                          # fd.explain: 연결마다 배율 탐침 (값 1)
            values = values * self._probe["edge"]
        probe_n = self._probe.get("neuron")
        M, MT = K.matrices(self.wiring, values.data)
        in_idx = self.in_idx
        rec = [] if record is not None else None
        rec_idx = B.to(np.asarray(record), self.device) if record is not None else None
        fx = genetics_effects(self)
        quiet, blocked, act_idx = fx.get("silence"), fx.get("block"), fx.get("act_idx")
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        drop = genetics_mosaic(self, seed, Bn)
        if act_idx is not None:                                            # 활성화 = 활동을 level로 고정 (입력 뉴런처럼)
            in_idx = xp.concatenate([in_idx, act_idx])
            level = Signal(xp.broadcast_to(fx["act_level"], (len(act_idx), Bn)).copy())

        trunc = self.truncate

        def run(s0, s1, V, r, acc):
            for s in range(s0, s1):
                if trunc and s and s % trunc == 0:
                    V, r = _cut(V), _cut(r)
                I = K.propagate(r if blocked is None else r * blocked, values, M, MT, self.wiring)
                x_in = frame(s, steps)
                if act_idx is not None:
                    x_in = concat([x_in, level])
                V, r = K.graded_step(V, I, x_in, in_idx, b, a, r_max)
                if quiet is not None:
                    r = r * quiet
                if probe_n is not None:
                    r = r * probe_n
                if drop is not None:
                    r = r * drop
                if s >= s_cnt:
                    acc = acc + r
                if rec is not None:
                    rec.append(r.data[rec_idx].T.copy())
            return V, r, acc

        z = lambda: Signal(xp.zeros((N, Bn), dtype=xp.float32))
        state = self._run(run, (z(), z(), z()), steps, x, record)
        mean = state[2].T * (1.0 / (steps - s_cnt))
        out = mean if return_all else mean[:, self.out_idx]
        return (out, self._trace(rec)) if record is not None else out

    # ─────────────── 저장 / 불러오기 ───────────────
    def save(self, path):
        """회로 배선 + 설정 + 학습한 값을 파일 하나에 (np.savez). FlyWire 데이터 없이 load()로 다시 만듦"""
        from .. import __version__
        if not isinstance(self.neuron, str) and self.config.get("neuron") is None:
            raise ValueError(f"{type(self.neuron).__name__}이 등록되지 않아 저장할 수 없음 - @fd.neurons.register")
        cfg = dict(self.config, gains=dict(self.gains))
        arrays = {"format": np.array(self.FORMAT), "version": np.array(__version__),
                  "config": np.array(json.dumps(cfg, ensure_ascii=False))}
        arrays.update(self.circuit.to_arrays("circuit."))
        arrays.update({"state." + k: v for k, v in self.state().items()})
        from .._archive import write
        return write(path, "ConnectomeLayer", arrays)

    @classmethod
    def load(cls, path, device: str | None = None) -> "ConnectomeLayer":
        from ..circuit import Circuit
        from .._archive import read
        d, _ = read(path, "ConnectomeLayer")
        if str(d.get("format")) != cls.FORMAT:
            raise ValueError(f"flydnet ganglion ConnectomeLayer 파일이 아님 (format={d.get('format')})")
        cfg = json.loads(str(d["config"]))
        cfg.setdefault("timing", "legacy")                                 # 0.1.15 이전 파일은 그때 방식으로
        cfg.setdefault("surrogate_damp", 1.0)
        layer = cls(Circuit.from_arrays(d, "circuit."), device=device, **cfg)
        layer.load_state({k[6:]: v for k, v in d.items() if k.startswith("state.")})
        layer._build()
        return layer

    def damp_value(self, t_ms: float | None = None) -> float:
        """이 층이 쓰는 대리 기울기 감쇠 값 ("auto"면 t_ms 길이에서 계산된 값)"""
        dt = self.p["dt"]
        steps = int(round((t_ms or self.t_ms) / dt))
        return resolve_damp(self.surrogate_damp, steps, max(int(round(self.p["t_dly"] / dt)), 1))

    def extra_repr(self):
        tr = f", 학습 연결 {len(self.train_pos):,}개" if self.trainable else ""
        if self.neuron == "graded":
            tr = ", 연속값 뉴런" + tr
        if self.trainable and self.share == "pair":
            tr += f" (종류 {len(self.log_scale.data):,}개가 배율 공유)"
        if self.neuron_params:
            tr += f", 뉴런 매개변수 그룹 {len(self.group_names)}개{' 학습' if isinstance(self.bias, Synapse) else ''}"
        if self._effects:
            tr += f", 켜진 효과기: {', '.join(map(repr, self._effects))}"
        if self.neuron == "lif" and self.timing != "brian":
            tr += f", timing {self.timing}"
        if not isinstance(self.neuron, str):
            tr += f", 뉴런 {self.neuron!r}"
        return (f"{self.circuit.name}: in {self.n_in} ({'+'.join(self.in_names)}) → out {self.n_out} "
                f"({'+'.join(self.out_names)}), {self.t_ms} ms, dt {self.p['dt']} ms, 입력 {self.input_mode}, "
                f"장치 {self.device}{tr}")
