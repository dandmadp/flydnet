"""조직: 매개변수를 가진 신경 구조물 (torch.nn에 해당)

  Synapse             학습되는 연결 세기                           ≈ nn.Parameter
  Tissue              구조물의 기본 클래스                         ≈ nn.Module
  Pathway(*조직)      차례로 이어진 신경 경로                       ≈ nn.Sequential
  Projection          축삭 투사: 모든 입력이 모든 출력에 연결       ≈ nn.Linear
  Neuropil            신경망 영역: 실제 커넥톰 배선으로만 연결      ≈ nn.Linear (희소)
  LateralInhibition   측억제 (APL): 가장 강한 k개만
  AxonHillock         축삭 둔덕: 스파이크 (대리 기울기)
  Activation          단순 활성화 (relu, tanh, sigmoid)
  MushroomBodyOutput  버섯체 출력 구역: 도파민 연합 학습 분류기 (역전파 없음)
"""
from __future__ import annotations

import numpy as np

from . import backend as B
from . import physiology as P
from .signal import Signal, as_signal, quiescent


def _is_array(v) -> bool:
    """numpy·cupy 배열인지 (Wiring처럼 shape만 있는 구조는 아님)"""
    return hasattr(v, "dtype") and hasattr(v, "shape")


class Synapse(Signal):
    """학습되는 연결 세기 (plastic=True인 잎 신호, nn.Parameter)"""

    def __init__(self, data, device: str | None = None):
        super().__init__(data, plastic=True, device=device)

    def __repr__(self):
        return f"Synapse(shape={tuple(self.shape)}, device={self.device})"


class Tissue:
    """구조물 기본 클래스 (nn.Module). 속성으로 넣은 Synapse·Tissue·배열 버퍼를 자동으로 관리"""

    def __init__(self):
        object.__setattr__(self, "_synapses", {})
        object.__setattr__(self, "_tissues", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "learning", True)

    def __setattr__(self, name, value):
        if "_synapses" not in self.__dict__:
            raise RuntimeError("Tissue.__init__()을 먼저 불러야 함 (super().__init__())")
        for d in (self._synapses, self._tissues, self._buffers):
            d.pop(name, None)
        if isinstance(value, Synapse):
            self._synapses[name] = value
        elif isinstance(value, Tissue):
            self._tissues[name] = value
        object.__setattr__(self, name, value)

    def buffer(self, name: str, value, persistent: bool = True):
        """학습되지 않지만 장치 이동·저장에 따라가는 배열 (register_buffer)"""
        object.__setattr__(self, name, value)
        self._buffers[name] = persistent

    # ─────────────── 순회 ───────────────
    def named_synapses(self, prefix: str = ""):
        for n, s in self._synapses.items():
            yield prefix + n, s
        for n, t in self._tissues.items():
            yield from t.named_synapses(prefix + n + ".")

    def synapses(self):
        """학습되는 Synapse 전부 (parameters())"""
        return [s for _, s in self.named_synapses()]

    def tissues(self):
        yield self
        for t in self._tissues.values():
            yield from t.tissues()

    def n_synapses(self) -> int:
        return int(sum(s.data.size for s in self.synapses()))

    # ─────────────── 장치 · 상태 ───────────────
    @property
    def device(self) -> str:
        for s in self.synapses():
            return s.device
        for t in self.tissues():
            for n in t._buffers:
                v = getattr(t, n)
                if _is_array(v):
                    return B.device_of(v)
        return "cpu"

    def to(self, device: str):
        """모든 Synapse와 버퍼를 장치로 (제자리)"""
        B.check(device)
        for t in self.tissues():
            for n, s in t._synapses.items():
                s.data = B.to(s.data, device)
                s.retro = None if s.retro is None else B.to(s.retro, device)
            for n in t._buffers:
                v = getattr(t, n)
                if _is_array(v):
                    object.__setattr__(t, n, B.to(v, device))
                elif hasattr(v, "to"):                                    # Wiring 같은 구조
                    object.__setattr__(t, n, v.to(device))
            t._moved(device)
        return self

    def _moved(self, device: str):
        """장치를 옮긴 뒤 각 조직이 따로 할 일 (예: 캐시 다시 만들기)"""

    def clear_retro(self):
        """쌓인 역행성 신호(기울기) 지우기 (zero_grad)"""
        for s in self.synapses():
            s.retro = None

    def state(self) -> dict:
        """저장용 상태: 이름 → numpy 배열 (state_dict)"""
        out = {n: s.numpy().copy() for n, s in self.named_synapses()}

        def walk(t, prefix):
            for n, keep in t._buffers.items():
                v = getattr(t, n)
                if keep and _is_array(v):
                    out[prefix + n] = B.numpy(v).copy()
            for n, c in t._tissues.items():
                walk(c, prefix + n + ".")
        walk(self, "")
        return out

    def load_state(self, state: dict, strict: bool = True):
        """state()로 저장한 것을 불러옴 (load_state_dict)"""
        mine = self.state()
        if strict and set(state) != set(mine):
            raise KeyError(f"상태 이름이 다름: 없음 {sorted(set(mine) - set(state))}, 남음 {sorted(set(state) - set(mine))}")
        dev = self.device
        syn = dict(self.named_synapses())

        def check(t, prefix):                                   # 배선 지문이 다르면 아무것도 바꾸기 전에 멈춤
            key = prefix + "wiring_id"
            if "wiring_id" in t._buffers and key in state and \
                    not np.array_equal(np.asarray(state[key]), B.numpy(t.wiring_id)):
                raise RuntimeError(f"{type(t).__name__} 배선이 다름 ({key}): 다른 회로(또는 그룹)로 만든 것의 상태")
            for n, c in t._tissues.items():
                check(c, prefix + n + ".")
        check(self, "")

        def walk(t, prefix):
            for n, keep in t._buffers.items():
                if keep and prefix + n in state:
                    object.__setattr__(t, n, B.to(np.asarray(state[prefix + n]), dev))
            for n, c in t._tissues.items():
                walk(c, prefix + n + ".")
        for n, arr in state.items():
            if n in syn:
                if tuple(arr.shape) != tuple(syn[n].shape):
                    raise ValueError(f"{n}: 모양 {arr.shape} ≠ {syn[n].shape}")
                syn[n].data = B.to(np.asarray(arr, dtype=syn[n].data.dtype), dev)
        walk(self, "")
        return self

    def save(self, path):
        """state()를 파일 하나에 (np.savez, 형식 버전 포함, 저장 도중 멈춰도 예전 파일 유지)"""
        from .._archive import write
        return write(path, type(self).__name__, self.state())

    def load(self, path):
        from .._archive import read
        return self.load_state(read(path, type(self).__name__)[0])

    # ─────────────── 호출 ───────────────
    def __call__(self, *args, **kwargs):
        with B.oom_hint(f"{type(self).__name__} 순전파"):
            return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def __repr__(self):
        head = f"{type(self).__name__}({self.extra_repr()}"
        if not self._tissues:
            return head + ")"
        body = "\n".join(f"  ({n}): " + repr(t).replace("\n", "\n  ") for n, t in self._tissues.items())
        return head + "\n" + body + "\n)"


class Pathway(Tissue):
    """차례로 이어진 신경 경로 (nn.Sequential)"""

    def __init__(self, *tissues):
        super().__init__()
        for i, t in enumerate(tissues):
            setattr(self, str(i), t)
        self._order = list(tissues)

    def forward(self, x):
        for t in self._order:
            x = t(x)
        return x

    def __getitem__(self, i):
        return self._order[i]

    def __len__(self):
        return len(self._order)


def _device(device):
    return B.check(device) if device is not None else B.default_device()


_labels = B.labels


class Projection(Tissue):
    """축삭 투사: 모든 입력이 모든 출력에 연결 (nn.Linear). 초기값은 ±1/√n_in 균등 (torch와 같은 방식)"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True, seed: int | None = None, device: str | None = None):
        super().__init__()
        dev = _device(device)
        rng = np.random.default_rng(seed)
        bound = 1 / np.sqrt(n_in)
        self.weight = Synapse(rng.uniform(-bound, bound, (n_out, n_in)).astype(np.float32), device=dev)
        self.bias = Synapse(rng.uniform(-bound, bound, n_out).astype(np.float32), device=dev) if bias else None
        self.n_in, self.n_out = n_in, n_out

    def forward(self, x):
        x = as_signal(x, self.weight.device)
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y

    def extra_repr(self):
        return f"{self.n_in} → {self.n_out}, bias={self.bias is not None}"


class Neuropil(Tissue):
    """신경망 영역: 회로의 pre 그룹 → post 그룹을 실제 커넥톰 시냅스로만 연결. (B, n_pre) → (B, n_post)

    train:  "edge" = 연결마다 배율 (부호 유지, 기본) / "pair" = 연결 종류마다 배율 / "free" = 값 자유 / None = 고정
    init:   "fan_in" = 받는 뉴런마다 √(Σ w²)로 나눔 (입력 분산 ≈ 출력 분산) / "counts" = 시냅스 수 그대로
    다른 회로로 만든 Neuropil의 상태를 불러오면 배선 지문이 달라 오류.
    """

    def __init__(self, circuit, pre, post, train: str | None = "edge", init: str = "fan_in", bias: bool = False,
                 device: str | None = None):
        super().__init__()
        if train not in ("pair", "edge", "free", None):
            raise ValueError(f"train은 'pair', 'edge', 'free', None 중 하나: {train}")
        if init not in ("fan_in", "counts"):
            raise ValueError(init)
        dev = _device(device)
        pre = [pre] if isinstance(pre, str) else list(pre)
        post = [post] if isinstance(post, str) else list(post)
        for g in pre + post:
            if g not in circuit.groups:
                raise ValueError(f"회로에 없는 그룹: {g}")
        pre_idx = np.concatenate([circuit.groups[g] for g in pre])
        post_idx = np.concatenate([circuit.groups[g] for g in post])
        lp = np.full(circuit.N, -1, np.int64); lp[pre_idx] = np.arange(len(pre_idx))
        lq = np.full(circuit.N, -1, np.int64); lq[post_idx] = np.arange(len(post_idx))
        m = (lp[circuit.pre] >= 0) & (lq[circuit.post] >= 0)
        q, p, w = lq[circuit.post[m]], lp[circuit.pre[m]], circuit.weight[m].astype(np.float64)
        key, inv = np.unique(q * len(pre_idx) + p, return_inverse=True)       # 같은 연결은 합치기
        w = np.bincount(inv, weights=w, minlength=len(key))
        q, p = key // len(pre_idx), key % len(pre_idx)
        nz = w != 0                                                            # ±가 상쇄된 연결은 빼기
        q, p, w = q[nz], p[nz], w[nz]
        if len(q) == 0:
            raise ValueError(f"{pre} → {post} 연결이 없음")
        wr, order = P.wiring(q, p, len(post_idx), len(pre_idx), device=dev)
        q, p, w = q[order], p[order], w[order]
        if init == "fan_in":
            w = w / np.sqrt(np.bincount(q, weights=w ** 2, minlength=len(post_idx)))[q]
        names = list(circuit.groups)
        gid = np.full(circuit.N, -1, np.int64)
        for i, nm in enumerate(names):
            gid[circuit.groups[nm]] = i
        code = gid[pre_idx][p] * len(names) + gid[post_idx][q]
        uniq, which = np.unique(code, return_inverse=True)
        self.pairs = [f"{names[c // len(names)]}>{names[c % len(names)]}" for c in uniq]
        fp = np.array([len(pre_idx), len(post_idx), len(q), int(q.sum()), int(p.sum()),
                       int((q * 31 + p * 17).sum() % (2 ** 61 - 1))], dtype=np.int64)

        self.buffer("wiring", wr, persistent=False)
        self.buffer("base", B.to(w.astype(np.float32), dev))
        self.buffer("pair_of", B.to(which.astype(np.int64), dev), persistent=False)
        self.buffer("wiring_id", fp)
        self.train_mode = train
        if train == "pair":
            self.log_scale = Synapse(np.zeros(len(uniq), np.float32), device=dev)
        elif train == "edge":
            self.log_scale = Synapse(np.zeros(len(w), np.float32), device=dev)
        elif train == "free":
            self.values_ = Synapse(w.astype(np.float32), device=dev)
        self.bias = Synapse(np.zeros(len(post_idx), np.float32), device=dev) if bias else None
        self.in_features, self.out_features = len(pre_idx), len(post_idx)
        self.pre_names, self.post_names = pre, post

    @property
    def n_connections(self) -> int:
        return int(len(self.base))

    def values(self) -> Signal:
        """현재 연결 값 (배선 순서: post 순, 그 안에서 pre 순)"""
        base = Signal(self.base)
        if self.train_mode == "pair":
            return base * self.log_scale.exp()[self.pair_of]
        if self.train_mode == "edge":
            return base * self.log_scale.exp()
        if self.train_mode == "free":
            return self.values_
        return base

    def forward(self, x):
        x = as_signal(x, B.device_of(self.base))
        out = P.transmit(x, self.values(), self.wiring)
        return out + self.bias if self.bias is not None else out

    def dense(self) -> np.ndarray:
        """밀집 가중치 (n_post, n_pre), numpy - 확인용"""
        if self.out_features * self.in_features > 2e8:
            raise MemoryError(f"밀집 행렬 {self.out_features}×{self.in_features}는 너무 큼")
        with quiescent():
            v = self.values().numpy()
        W = np.zeros((self.out_features, self.in_features), np.float32)
        W[B.numpy(self.wiring.post), B.numpy(self.wiring.pre)] = v
        return W

    def scale_of(self) -> dict:
        if self.train_mode != "pair":
            raise ValueError("train='pair'에서만")
        return dict(zip(self.pairs, np.exp(self.log_scale.numpy()).tolist()))

    def extra_repr(self):
        dens = self.n_connections / (self.in_features * self.out_features)
        return (f"{'+'.join(self.pre_names)} {self.in_features} → {'+'.join(self.post_names)} {self.out_features}, "
                f"연결 {self.n_connections:,}개 (밀도 {dens:.2%}), 학습 {self.train_mode}, 장치 {B.device_of(self.base)}")


class LateralInhibition(Tissue):
    """측억제 (APL 뉴런처럼): 가장 강한 k개(또는 비율 frac)만 남기고 0"""

    def __init__(self, k: int | None = None, frac: float | None = None):
        super().__init__()
        if (k is None) == (frac is None):
            raise ValueError("k와 frac 중 하나만")
        self.k, self.frac = k, frac

    def forward(self, x):
        return P.inhibit(x, k=self.k, frac=self.frac)

    def extra_repr(self):
        return f"k={self.k}" if self.k is not None else f"frac={self.frac}"


class AxonHillock(Tissue):
    """축삭 둔덕: 문턱을 넘으면 스파이크 1 (대리 기울기)"""

    def __init__(self, threshold: float = 0.0, slope: float = 10.0):
        super().__init__()
        self.threshold, self.slope = threshold, slope

    def forward(self, v):
        return P.fire(v, self.threshold, self.slope)

    def extra_repr(self):
        return f"threshold={self.threshold}, slope={self.slope}"


class Activation(Tissue):
    """단순 활성화: "relu" | "tanh" | "sigmoid" """

    def __init__(self, kind: str = "relu"):
        super().__init__()
        if kind not in ("relu", "tanh", "sigmoid"):
            raise ValueError(kind)
        self.kind = kind

    def forward(self, x):
        return getattr(as_signal(x), self.kind)()

    def extra_repr(self):
        return self.kind


class MushroomBodyOutput(Tissue):
    """버섯체 출력 구역: 도파민 연합 학습 분류기 (역전파 없음, 연속 학습에 강함)
    learn(x, y): 정답 클래스의 가장 잘 맞는 원형 하나만 강화 (누적 평균). forward(x) = 클래스 점수 (코사인 최대)"""

    def __init__(self, n_in: int, n_classes: int, per_class: int = 1, binary: bool = False, device: str | None = None):
        super().__init__()
        dev = _device(device)
        xp = B.xp(dev)
        self.n_in, self.n_classes, self.per_class, self.binary = n_in, n_classes, per_class, binary
        self.buffer("prototypes", xp.zeros((n_classes * per_class, n_in), dtype=xp.float32))
        self.buffer("count", xp.zeros(n_classes * per_class, dtype=xp.float32))

    def activity(self, x):
        a = as_signal(x, B.device_of(self.prototypes)).data
        a = a.reshape(len(a), -1).astype(self.prototypes.dtype, copy=False)
        if self.binary:
            return (a > 0).astype(a.dtype)
        xp = B.xp(B.device_of(a))
        return a / xp.maximum(a.max(axis=1, keepdims=True), 1e-8)

    def _scores(self, a):
        s = P.recall(a, self.prototypes, self.count)
        return s.reshape(len(a), self.n_classes, self.per_class).max(axis=2)

    def forward(self, x) -> Signal:
        return Signal(self._scores(self.activity(x)))

    def learn(self, x, y, batch: int = 256):
        """도파민 강화 (physiology.reinforce_ - AssocReadout과 같은 규칙)"""
        a_all = self.activity(x)
        y_all = B.labels(y)
        for s in range(0, len(a_all), batch):
            P.reinforce_(self.prototypes, self.count, a_all[s:s + batch], y_all[s:s + batch], self.per_class)
        return self

    def predict(self, x, classes=None) -> np.ndarray:
        """예측 클래스 (numpy, 장치와 상관없이 - 정답 라벨과 바로 비교하도록)"""
        s = self._scores(self.activity(x))
        xp = B.xp(B.device_of(s))
        if classes is not None:
            mask = xp.full(self.n_classes, -xp.inf, dtype=s.dtype)
            mask[xp.asarray(list(classes))] = 0
            s = s + mask
        return B.numpy(s.argmax(1))

    def extra_repr(self):
        n = int((B.numpy(self.count).reshape(self.n_classes, self.per_class).sum(1) > 0).sum())
        return f"in {self.n_in} → 클래스 {self.n_classes} × 원형 {self.per_class}, 배운 클래스 {n}개"
