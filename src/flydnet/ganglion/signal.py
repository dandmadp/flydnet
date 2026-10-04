"""Signal: 신경 신호 = 배열 + 역행성 신호(기울기) 기록 (torch.Tensor에 해당)

    s = Signal([[1., 2.]], plastic=True)     # plastic=True: 학습으로 바뀌는 신호 (requires_grad)
    loss = (s * s).sum()
    loss.retrograde()                         # 역행성 신호를 거슬러 보냄 (backward)
    s.retro                                   # 도착한 역행성 신호 = d loss / d s (grad)
    with quiescent(): ...                     # 휴지 상태: 학습 흔적을 남기지 않음 (no_grad)

역행성 신호(retrograde signal): 실제 시냅스에서도 받는 쪽 뉴런이 보내는 쪽으로 거꾸로 보내는 신호가 있다
(예: 내인성 카나비노이드). 여기서는 출력 오차를 입력 쪽으로 되돌려 보내는 기울기 계산을 그렇게 부른다.
"""
from __future__ import annotations

import contextlib

import numpy as np

from . import backend as B

_LEARNING = [True]                                     # quiescent()로 끌 수 있음


@contextlib.contextmanager
def quiescent():
    """휴지 상태: 이 안의 연산은 역행성 신호 경로를 기록하지 않음 (torch.no_grad)"""
    prev = _LEARNING[0]
    _LEARNING[0] = False
    try:
        yield
    finally:
        _LEARNING[0] = prev


def learning_enabled() -> bool:
    return _LEARNING[0]


def _unbroadcast(g, shape):
    """브로드캐스트로 늘어난 차원을 원래 모양으로 합침"""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Signal:
    """신경 신호. data = 배열 (CPU: numpy, GPU: cupy), plastic = 학습으로 바뀌는가, retro = 역행성 신호"""

    __array_priority__ = 1000                          # numpy 배열 @ Signal 같은 연산을 Signal 쪽이 처리

    def __init__(self, data, plastic: bool = False, device: str | None = None, dtype=None):
        if isinstance(data, Signal):
            data = data.data
        elif hasattr(data, "detach") and hasattr(data, "cpu"):    # torch 텐서 → numpy (torch 연동)
            data = data.detach().cpu().numpy()
        if device is None:
            device = B.device_of(data) if not isinstance(data, (list, tuple, float, int)) else "cpu"
        xp = B.xp(device)
        arr = B.to(data, device) if hasattr(data, "shape") else xp.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        elif arr.dtype.kind in "fc" and arr.dtype != xp.float32 and arr.dtype != xp.float64:
            arr = arr.astype(xp.float32)
        elif arr.dtype.kind not in "fcbiu":
            arr = arr.astype(xp.float32)
        self.data = arr
        self.plastic = bool(plastic)
        self.retro = None
        self._parents: tuple = ()
        self._back = None                                  # g → 부모마다의 역행성 신호

    # ─────────────── 기본 정보 ───────────────
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self) -> str:
        return B.device_of(self.data)

    @property
    def xp(self):
        return B.xp(self.device)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        tag = ", plastic" if self.plastic else ""
        return f"Signal({B.numpy(self.data)!r}, device={self.device}{tag})"

    def numpy(self) -> np.ndarray:
        return B.numpy(self.data)

    def to_torch(self):
        """torch 텐서로 (torch 연동, 값만 복사 - 역행성 신호 경로는 이어지지 않음)"""
        import torch
        if self.device == "gpu":
            return torch.from_dlpack(self.data).clone()
        return torch.from_numpy(np.ascontiguousarray(self.data).copy())

    def item(self):
        return self.data.item()

    def to(self, device: str) -> "Signal":
        """다른 장치로 옮긴 새 신호 (역행성 신호 경로는 이어짐)"""
        if device == self.device:
            return self
        out = Signal(B.to(self.data, device))
        src = self.device
        return out._link((self,), lambda g: (B.to(g, src),))

    def detach(self) -> "Signal":
        return Signal(self.data)

    # ─────────────── 연산 그래프 ───────────────
    def _link(self, parents, back):
        """parents에서 만들어진 신호로 기록 (학습 중이고 부모 중 하나라도 plastic이면).
        plastic이 아닌 부모(상수)는 자리표만 남김 - 역전파에 쓰이지 않으므로 그 배열을 붙잡아 둘 필요가 없음
        (back이 직접 쓰는 값은 back 쪽이 따로 들고 있음)"""
        if learning_enabled() and any(p.plastic for p in parents):
            self.plastic = True
            self._parents = tuple(p if p.plastic else _CONST for p in parents)
            self._back = back
        return self

    def retrograde(self, retro=None, keep: bool = False):
        """역행성 신호 보내기 (backward): 이 신호에서 plastic 잎(Synapse 등)까지 기울기를 계산해 .retro에 더함.
        keep=True면 경로를 풀지 않음 (같은 경로로 다시 보낼 때, retain_graph)"""
        if not self.plastic:
            raise RuntimeError("plastic이 아닌 신호에서는 역행성 신호를 보낼 수 없음")
        xp = self.xp
        if retro is None:
            if self.data.size != 1:
                raise RuntimeError("값이 하나가 아닌 신호는 retro를 직접 줘야 함")
            if not bool(xp.isfinite(self.data).all()):
                raise FloatingPointError(
                    f"손실이 {float(self.data.reshape(-1)[0])} - 역행성 신호를 보내지 않음. 학습이 발산했거나 입력에 "
                    "NaN·무한대가 있음: 학습률을 낮추거나 가소성 규칙에 clip=1.0, 입력 확인")
            retro = xp.ones_like(self.data)
        retro = retro.data if isinstance(retro, Signal) else xp.asarray(retro, dtype=self.data.dtype)

        order, seen, stack = [], set(), [(self, False)]      # 위상 정렬 (재귀 없이)
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.plastic and id(p) not in seen:
                    stack.append((p, False))
        with B.oom_hint("역행성 신호(역전파)"):
            self._send(order, retro)
        if not keep:
            for node in order:                               # 경로 풀기 (메모리 해제)
                if node._back is not None:
                    node._parents, node._back = (), None

    def _send(self, order, retro):
        grads = {id(self): retro}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._back is None:                          # 잎: 기울기 쌓기
                node.retro = g if node.retro is None else node.retro + g
                continue
            for p, pg in zip(node._parents, node._back(g), strict=True):   # 연산이 기울기를 덜 돌려주면 바로 오류
                if pg is None or not p.plastic:
                    continue
                grads[id(p)] = pg if id(p) not in grads else grads[id(p)] + pg

    # ─────────────── 산술 ───────────────
    def _wrap(self, other) -> "Signal":
        if isinstance(other, Signal):
            if other.device != self.device:
                raise RuntimeError(f"장치가 다름: {self.device} 와 {other.device}")
            return other
        xp = self.xp
        if isinstance(other, (bool, int, float)) and not isinstance(other, np.ndarray):
            # 파이썬 숫자: 실수 신호면 그 자료형 그대로 (float32 유지), 정수 신호에 실수를 곱하면 float32로 (잘림 방지)
            if self.data.dtype.kind == "f":
                dt = self.data.dtype
            elif isinstance(other, float):
                dt = xp.float32
            else:
                dt = self.data.dtype
            return Signal(xp.asarray(other, dtype=dt), device=self.device)
        return Signal(B.to(np.asarray(other) if not hasattr(other, "shape") else other, self.device))

    def __add__(self, o):
        o = self._wrap(o)
        a, b = self.shape, o.shape
        return Signal(self.data + o.data)._link((self, o), lambda g: (_unbroadcast(g, a), _unbroadcast(g, b)))

    __radd__ = __add__

    def __sub__(self, o):
        o = self._wrap(o)
        a, b = self.shape, o.shape
        return Signal(self.data - o.data)._link((self, o), lambda g: (_unbroadcast(g, a), _unbroadcast(-g, b)))

    def __rsub__(self, o):
        return self._wrap(o) - self

    def __neg__(self):
        return Signal(-self.data)._link((self,), lambda g: (-g,))

    def __mul__(self, o):
        o = self._wrap(o)
        x, y = self.data, o.data
        return Signal(x * y)._link((self, o), lambda g: (_unbroadcast(g * y, x.shape), _unbroadcast(g * x, y.shape)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._wrap(o)
        x, y = self.data, o.data
        return Signal(x / y)._link((self, o), lambda g: (_unbroadcast(g / y, x.shape),
                                                        _unbroadcast(-g * x / (y * y), y.shape)))

    def __rtruediv__(self, o):
        return self._wrap(o) / self

    def __pow__(self, p: float):
        if isinstance(p, Signal):
            raise TypeError("지수는 숫자만")
        x = self.data
        return Signal(x ** p)._link((self,), lambda g: (g * p * x ** (p - 1),))

    def __matmul__(self, o):
        o = self._wrap(o)
        x, y = self.data, o.data
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("@는 2차원끼리만")
        return Signal(x @ y)._link((self, o), lambda g: (g @ y.T, x.T @ g))

    def __rmatmul__(self, o):
        return self._wrap(o) @ self

    # ─────────────── 비교 (기울기 없음) ───────────────
    def __gt__(self, o):
        return Signal(self.data > (o.data if isinstance(o, Signal) else o))

    def __lt__(self, o):
        return Signal(self.data < (o.data if isinstance(o, Signal) else o))

    def __ge__(self, o):
        return Signal(self.data >= (o.data if isinstance(o, Signal) else o))

    def __le__(self, o):
        return Signal(self.data <= (o.data if isinstance(o, Signal) else o))

    # ─────────────── 원소별 함수 ───────────────
    def exp(self):
        out = self.xp.exp(self.data)
        return Signal(out)._link((self,), lambda g: (g * out,))

    def log(self):
        x = self.data
        return Signal(self.xp.log(x))._link((self,), lambda g: (g / x,))

    def relu(self):
        m = self.data > 0
        return Signal(self.data * m)._link((self,), lambda g: (g * m,))

    def tanh(self):
        out = self.xp.tanh(self.data)
        return Signal(out)._link((self,), lambda g: (g * (1 - out * out),))

    def sigmoid(self):
        out = 1 / (1 + self.xp.exp(-self.data))
        return Signal(out)._link((self,), lambda g: (g * out * (1 - out),))

    def abs(self):
        s = self.xp.sign(self.data)
        return Signal(self.xp.abs(self.data))._link((self,), lambda g: (g * s,))

    def clip(self, lo=None, hi=None):
        x = self.data
        m = self.xp.ones_like(x, dtype=bool)
        if lo is not None:
            m &= x >= lo
        if hi is not None:
            m &= x <= hi
        return Signal(self.xp.clip(x, lo, hi))._link((self,), lambda g: (g * m,))

    # ─────────────── 모으기 ───────────────
    def sum(self, axis=None, keepdims: bool = False):
        shape = self.shape
        xp = self.xp

        def back(g):
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)
            return (xp.broadcast_to(g, shape).copy(),)
        return Signal(self.data.sum(axis=axis, keepdims=keepdims))._link((self,), back)

    def mean(self, axis=None, keepdims: bool = False, dtype=None, out=None):
        if out is not None:
            raise TypeError("Signal.mean은 out=을 받지 않음")
        n = self.data.size if axis is None else int(np.prod([self.shape[a] for a in np.atleast_1d(axis)]))
        return self.sum(axis, keepdims) * (1.0 / n)

    def min(self, axis=None, keepdims: bool = False):
        """최솟값 (같은 값이 여럿이면 기울기를 나눠 가짐)"""
        return -((-self).max(axis=axis, keepdims=keepdims))

    def var(self, axis=None, keepdims: bool = False, ddof: int = 0):
        """분산 (ddof=1이면 표본 분산)"""
        n = self.data.size if axis is None else int(np.prod([self.shape[a] for a in np.atleast_1d(axis)]))
        d = self - self.mean(axis=axis, keepdims=True)
        return (d * d).sum(axis=axis, keepdims=keepdims) * (1.0 / max(n - ddof, 1))

    def std(self, axis=None, keepdims: bool = False, ddof: int = 0):
        return self.var(axis=axis, keepdims=keepdims, ddof=ddof) ** 0.5

    def sqrt(self):
        return self ** 0.5

    def square(self):
        return self * self

    def softmax(self, axis: int = -1):
        from .physiology import log_softmax
        return log_softmax(self, axis=axis).exp()

    def argmax(self, axis=None) -> np.ndarray:
        """가장 큰 값의 위치 (numpy, 역전파 없음)"""
        return B.numpy(self.data.argmax(axis=axis))

    def argmin(self, axis=None) -> np.ndarray:
        return B.numpy(self.data.argmin(axis=axis))

    def squeeze(self, axis=None):
        return self.reshape(*self.data.squeeze(axis=axis).shape)

    def astype(self, dtype):
        """자료형 바꾸기 (역행성 신호는 원래 자료형으로 돌아감)"""
        src = self.data.dtype
        return Signal(self.data.astype(dtype))._link((self,), lambda g: (g.astype(src),))

    def copy(self):
        """값을 복사한 새 신호 (역행성 신호 경로는 이어짐, torch의 clone)"""
        return Signal(self.data.copy())._link((self,), lambda g: (g,))

    def tolist(self):
        return self.numpy().tolist()

    def any(self) -> bool:
        return bool(self.data.any())

    def all(self) -> bool:
        return bool(self.data.all())

    @property
    def size(self) -> int:
        """원소 개수 (numpy와 같음)"""
        return int(self.data.size)

    def _scalar(self, what):
        if self.data.size != 1:
            raise ValueError(f"값이 {self.data.size}개인 신호를 {what}로 바꿀 수 없음 - 값 하나인 신호에서만 "
                             "(.sum(), .mean() 등) 또는 .numpy()")
        return self.data.reshape(-1)[0]

    def __float__(self):
        return float(self._scalar("float"))

    def __int__(self):
        return int(self._scalar("int"))

    def __bool__(self):
        return bool(self._scalar("참·거짓 (if 등)"))

    def __abs__(self):
        return self.abs()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        from .hints import SIGNAL, missing
        raise missing(type(self).__name__, name, SIGNAL, dir(type(self)))

    def max(self, axis=None, keepdims: bool = False):
        """최댓값 (같은 값이 여럿이면 기울기를 나눠 가짐)"""
        x = self.data
        out = x.max(axis=axis, keepdims=True)
        m = (x == out)
        m = m / m.sum(axis=axis, keepdims=True)
        xp = self.xp
        res = out if keepdims else (out.squeeze(axis) if axis is not None else out.reshape(()))

        def back(g):
            if not keepdims:
                g = xp.expand_dims(g, axis) if axis is not None else g.reshape((1,) * x.ndim)
            return (g * m,)
        return Signal(res)._link((self,), back)

    # ─────────────── 모양 ───────────────
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        old = self.shape
        return Signal(self.data.reshape(shape))._link((self,), lambda g: (g.reshape(old),))

    def flatten(self, start: int = 1):
        return self.reshape(*self.shape[:start], -1)

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        inv = np.argsort(axes)
        return Signal(self.data.transpose(axes))._link((self,), lambda g: (g.transpose(inv),))

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        if isinstance(idx, Signal):
            idx = idx.data
        x = self.data
        xp = self.xp

        parts = idx if isinstance(idx, tuple) else (idx,)
        advanced = any(hasattr(p, "shape") or isinstance(p, list) for p in parts)

        def back(g):
            out = xp.zeros_like(x)
            if advanced:
                B.scatter_add(out, idx, g)                     # 같은 칸을 여러 번 고른 경우도 더함
            else:
                out[idx] += g                                  # 슬라이스·정수: 칸이 겹치지 않음
            return (out,)
        return Signal(x[idx])._link((self,), back)


class _Constant:
    """역전파 경로의 상수 자리표 (plastic 아님, 데이터 없음)"""
    plastic = False
    _parents = ()
    _back = None


_CONST = _Constant()


class _Packed:
    """여러 출력의 역행성 신호를 한 묶음으로 (checkpoint의 허브 노드용). 더하면 칸끼리 더함"""

    def __init__(self, n, k=None, g=None):
        self.parts = [None] * n
        if k is not None:
            self.parts[k] = g

    def __add__(self, other):
        out = _Packed(len(self.parts))
        out.parts = [b if a is None else a if b is None else a + b for a, b in zip(self.parts, other.parts)]
        return out


def checkpoint(fn, *inputs):
    """구간 다시 계산 (torch.utils.checkpoint): 순전파 때는 fn의 중간 신호를 버리고 출력만 남김.
    역전파 때 fn을 다시 계산해서 역행성 신호를 보냄 → 메모리는 구간 하나만큼.
    fn(*Signal) → Signal 튜플. fn은 같은 입력에 같은 결과여야 함 (난수는 시드·스텝으로 정할 것).
    fn 밖에서 만든 plastic 신호(예: 학습되는 연결 세기)를 fn이 써도 그쪽으로 역행성 신호가 감."""
    inputs = [as_signal(i) for i in inputs]
    if not learning_enabled():
        return fn(*inputs)
    with quiescent():
        outs = fn(*[Signal(i.data) for i in inputs])
    n = len(outs)

    def back(packed):
        fresh = [Signal(i.data, plastic=i.plastic) for i in inputs]
        outs2 = fn(*fresh)
        total = None
        for o, g in zip(outs2, packed.parts, strict=True):
            if g is not None and o.plastic:
                term = (o * Signal(g)).sum()
                total = term if total is None else total + term
        if total is not None:
            total.retrograde(keep=True)                      # fn 밖에서 온 신호의 경로는 다음 구간도 써야 함
        return tuple(f.retro if f.plastic else None for f in fresh)

    return multi_output(inputs, [o.data for o in outs], back_packed=back, force=True)


def multi_output(parents, outs, back=None, back_packed=None, force: bool = False):
    """출력이 여러 개인 연산을 만듦. outs: 출력 배열 목록, back(grads 목록 - 없는 것은 None) → 부모마다 역행성 신호.
    force=True면 부모가 plastic이 아니어도 경로를 만듦 (안에서 바깥 Synapse를 쓰는 경우, checkpoint)"""
    parents = tuple(parents)
    if not learning_enabled() or not (force or any(p.plastic for p in parents)):
        return tuple(Signal(o) for o in outs)
    n = len(outs)
    hub = Signal(B.xp(B.device_of(outs[0])).zeros(()))
    hub.plastic = True
    hub._parents = parents
    hub._back = back_packed if back_packed is not None else (lambda packed: back(packed.parts))
    result = []
    for k, o in enumerate(outs):
        sig = Signal(o)
        sig.plastic, sig._parents = True, (hub,)
        sig._back = (lambda k: (lambda g: (_Packed(n, k, g),)))(k)
        result.append(sig)
    return tuple(result)


def as_signal(x, device: str | None = None) -> Signal:
    if isinstance(x, Signal):
        return x if device is None else x.to(device)
    return Signal(x, device=device)


def concat(signals, axis: int = 0) -> Signal:
    xp = signals[0].xp
    sizes = [s.shape[axis] for s in signals]
    cuts = np.cumsum(sizes)[:-1]
    out = Signal(xp.concatenate([s.data for s in signals], axis=axis))
    return out._link(tuple(signals), lambda g: tuple(xp.split(g, cuts, axis=axis)))


def where(cond, a, b) -> Signal:
    """cond가 참인 칸은 a, 아니면 b. a·b 중 하나는 숫자여도 됨 (상대 신호의 장치·자료형을 따름)"""
    c = cond.data if isinstance(cond, Signal) else cond
    ref = a if isinstance(a, Signal) else b if isinstance(b, Signal) else None
    if ref is None:
        ref = Signal(B.xp(B.device_of(c)).zeros((), dtype=np.float32))
    cast = lambda v: v if isinstance(v, Signal) else Signal(ref.xp.asarray(v, dtype=ref.dtype))
    a, b = cast(a), cast(b)
    xp = ref.xp
    return Signal(xp.where(c, a.data, b.data))._link(
        (a, b), lambda g: (_unbroadcast(xp.where(c, g, 0), a.shape), _unbroadcast(xp.where(c, 0, g), b.shape)))
