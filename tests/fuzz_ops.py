"""연산 퍼징: 무작위 모양·축·브로드캐스팅으로 값(numpy float64 기준)과 기울기(수치 미분)를 반복 대조

  python tests/fuzz_ops.py              # 연산마다 40가지 경우 (약 1분)
  python tests/fuzz_ops.py 200          # 더 많이
tests/test_ops_fuzz.py가 같은 함수를 고정 seed로 적은 수만큼 돌림 (회귀 테스트)
"""
from __future__ import annotations

import sys

import numpy as np

import flydnet as fd
from flydnet.ganglion import physiology as P
from flydnet.ganglion.signal import concat, where


def _shape(r, nd=None):
    nd = nd or int(r.integers(1, 4))
    return tuple(int(r.integers(1, 5)) for _ in range(nd))


def _bshape(r, shape):
    """shape과 브로드캐스팅되는 모양 (일부 축을 1로, 앞 축을 뺌)"""
    s = [1 if r.random() < 0.4 else d for d in shape]
    k = int(r.integers(0, len(s)))
    return tuple(s[k:]) if r.random() < 0.3 else tuple(s)


def _axis(r, nd):
    c = r.random()
    if c < 0.25:
        return None
    if c < 0.5 and nd > 1:
        return tuple(sorted(r.choice(nd, size=int(r.integers(1, nd + 1)), replace=False).tolist()))
    a = int(r.integers(0, nd))
    return a - nd if r.random() < 0.3 else a


# 연산: 이름 → (입력 만들기, flydnet 함수, numpy 기준 함수). 입력은 float64 배열 목록
def _cases():
    C = {}

    def un(name, f_fd, f_np, pos=False, sep=None):
        def make(r):
            x = r.standard_normal(_shape(r))
            if pos:
                x = np.abs(x) + 0.5
            if sep is not None:                               # 꺾이는 점에서 떨어뜨림 (수치 미분이 정의되게)
                x = np.where(np.abs(x - sep) < 0.05, x + 0.2, x)
            return [x]
        C[name] = (make, lambda xs, r=None: f_fd(xs[0]), lambda xs: f_np(xs[0]))

    un("exp", lambda s: s.exp(), np.exp)
    un("log", lambda s: s.log(), np.log, pos=True)
    un("relu", lambda s: s.relu(), lambda x: np.maximum(x, 0), sep=0.0)
    un("tanh", lambda s: s.tanh(), np.tanh)
    un("sigmoid", lambda s: s.sigmoid(), lambda x: 1 / (1 + np.exp(-x)))
    un("abs", lambda s: s.abs(), np.abs, sep=0.0)
    un("sqrt", lambda s: s.sqrt(), np.sqrt, pos=True)
    un("square", lambda s: s.square(), np.square)
    un("neg", lambda s: -s, lambda x: -x)
    un("pow 1.7", lambda s: s ** 1.7, lambda x: x ** 1.7, pos=True)
    un("pow -0.5", lambda s: s ** -0.5, lambda x: x ** -0.5, pos=True)
    un("clip", lambda s: s.clip(-0.5, 0.7), lambda x: np.clip(x, -0.5, 0.7), sep=0.7)
    un("clip lo만", lambda s: s.clip(lo=-0.3), lambda x: np.maximum(x, -0.3), sep=-0.3)
    un("rsub", lambda s: 2.0 - s, lambda x: 2.0 - x)
    un("rdiv", lambda s: 2.0 / s, lambda x: 2.0 / x, pos=True)

    def bin_(name, f_fd, f_np, pos_b=False):
        def make(r):
            a = r.standard_normal(_shape(r))
            b = r.standard_normal(_bshape(r, a.shape))
            if r.random() < 0.5:
                a, b = b, a
            if pos_b:
                b = np.abs(b) + 0.5
            return [a, b]
        C[name] = (make, lambda xs: f_fd(xs[0], xs[1]), lambda xs: f_np(xs[0], xs[1]))

    bin_("add (브로드캐스팅)", lambda a, b: a + b, np.add)
    bin_("sub (브로드캐스팅)", lambda a, b: a - b, np.subtract)
    bin_("mul (브로드캐스팅)", lambda a, b: a * b, np.multiply)
    bin_("div (브로드캐스팅)", lambda a, b: a / b, np.divide, pos_b=True)

    def red(name, f_fd, f_np):
        def make(r):
            x = r.standard_normal(_shape(r))
            return [x], dict(axis=_axis(r, x.ndim), keepdims=bool(r.random() < 0.5))
        C[name] = (make, f_fd, f_np)

    red("sum", lambda s, k: s.sum(**k), lambda x, k: x.sum(**k))
    red("mean", lambda s, k: s.mean(**k), lambda x, k: x.mean(**k))
    red("max", lambda s, k: s.max(**k), lambda x, k: x.max(**k))
    red("min", lambda s, k: s.min(**k), lambda x, k: x.min(**k))
    red("var", lambda s, k: s.var(**k), lambda x, k: x.var(**k))
    red("std ddof1", lambda s, k: s.std(ddof=1, **k), lambda x, k: x.std(ddof=1, **k) if _n(x, k) > 1 else None)

    def mm(r):
        n, k, m = (int(r.integers(1, 5)) for _ in range(3))
        return [r.standard_normal((n, k)), r.standard_normal((k, m))]
    C["matmul"] = (mm, lambda xs: xs[0] @ xs[1], lambda xs: xs[0] @ xs[1])

    def tr(r):
        x = r.standard_normal(_shape(r, int(r.integers(2, 4))))
        ax = [a - x.ndim if r.random() < 0.4 else a for a in r.permutation(x.ndim).tolist()]   # 음수 축 섞음
        return [x], dict(axes=tuple(ax))
    C["transpose(axes)"] = (tr, lambda s, k: s.transpose(*k["axes"]), lambda x, k: x.transpose(k["axes"]))

    def mm1(r):
        n, k = int(r.integers(1, 5)), int(r.integers(1, 5))
        kind = int(r.integers(3))
        shapes = [((k,), (k, n)), ((n, k), (k,)), ((k,), (k,))][kind]
        return [r.standard_normal(shapes[0]), r.standard_normal(shapes[1])]
    C["matmul 1차원"] = (mm1, lambda xs: xs[0] @ xs[1], lambda xs: xs[0] @ xs[1])

    def pw(r):
        x = np.abs(r.standard_normal(_shape(r))) + 0.3
        return [x], dict(p=float(r.choice([0, 1, 2, 3, -1, 0.5, 2.5])))
    C["pow"] = (pw, lambda s, k: s ** k["p"], lambda x, k: x ** k["p"])

    def rp(r):
        return [r.standard_normal(_shape(r))], dict(base=float(r.choice([0.5, 2.0, np.e, 10.0])))
    C["숫자 ** 신호"] = (rp, lambda s, k: k["base"] ** s, lambda x, k: k["base"] ** x)

    def sq(r):
        x = r.standard_normal(_shape(r, int(r.integers(1, 4))))
        ones = [i for i, d in enumerate(x.shape) if d == 1]
        ax = None if not ones or r.random() < 0.3 else (ones[0] - x.ndim if r.random() < 0.5 else ones[0])
        return [x], dict(axis=ax)
    C["squeeze"] = (sq, lambda s, k: s.squeeze(k["axis"]), lambda x, k: x.squeeze(k["axis"]))

    def fl(r):
        x = r.standard_normal(_shape(r, int(r.integers(2, 4))))
        return [x], dict(start=int(r.integers(0, x.ndim)))
    C["flatten"] = (fl, lambda s, k: s.flatten(k["start"]), lambda x, k: x.reshape(*x.shape[:k["start"]], -1))

    def cl(r):
        x = r.standard_normal(_shape(r, 2)) * 2
        lo = -np.abs(r.standard_normal(x.shape[-1])) - 0.05
        hi = np.abs(r.standard_normal(x.shape[-1])) + 0.05
        x = np.where(np.minimum(np.abs(x - lo), np.abs(x - hi)) < 0.02, x + 0.1, x)   # 경계에서 떨어뜨림
        return [x], dict(lo=lo, hi=hi)
    C["clip 배열 경계"] = (cl, lambda s, k: s.clip(k["lo"], k["hi"]), lambda x, k: np.clip(x, k["lo"], k["hi"]))

    def rs(r):
        x = r.standard_normal(_shape(r))
        return [x], dict(shape=(int(np.prod(x.shape)),) if r.random() < 0.5 else (-1, 1))
    C["reshape"] = (rs, lambda s, k: s.reshape(*k["shape"]), lambda x, k: x.reshape(k["shape"]))

    def sm(r):
        x = r.standard_normal(_shape(r, int(r.integers(1, 4)))) * float(r.choice([1, 10]))
        return [x], dict(axis=int(r.integers(-x.ndim, x.ndim)))
    C["softmax"] = (sm, lambda s, k: s.softmax(k["axis"]), lambda x, k: _np_softmax(x, k["axis"]))
    C["log_softmax"] = (sm, lambda s, k: P.log_softmax(s, axis=k["axis"]), lambda x, k: np.log(_np_softmax(x, k["axis"])))

    def gi(r):
        x = r.standard_normal(_shape(r, int(r.integers(1, 4))))
        c = r.random()
        if c < 0.25:
            idx = tuple(slice(int(r.integers(0, d)), None, int(r.integers(1, 3))) for d in x.shape)
        elif c < 0.5:
            idx = (r.integers(-x.shape[0], x.shape[0], size=int(r.integers(1, 6))),)          # 중복·음수
        elif c < 0.75:
            idx = x > float(r.standard_normal())                                         # 불리언 마스크
        else:
            idx = int(r.integers(-x.shape[0], x.shape[0]))
        return [x], dict(idx=idx)
    C["getitem"] = (gi, lambda s, k: s[k["idx"]], lambda x, k: x[k["idx"]])

    def cc(r):
        sh = list(_shape(r, int(r.integers(1, 4))))
        ax = int(r.integers(0, len(sh)))
        parts = []
        for _ in range(int(r.integers(2, 4))):
            s2 = list(sh); s2[ax] = int(r.integers(1, 4))
            parts.append(r.standard_normal(s2))
        return parts, dict(axis=ax)
    C["concat"] = (cc, lambda ss, k: concat(ss, axis=k["axis"]), lambda xs, k: np.concatenate(xs, axis=k["axis"]))

    def wh(r):
        a = r.standard_normal(_shape(r)); b = r.standard_normal(a.shape)
        return [a, b], dict(cond=r.random(a.shape) < 0.5)
    C["where"] = (wh, lambda ss, k: where(k["cond"], ss[0], ss[1]), lambda xs, k: np.where(k["cond"], xs[0], xs[1]))

    def ast(r):
        return [r.standard_normal(_shape(r))], {}
    C["astype·copy·squeeze"] = (ast, lambda s, k: s.reshape(1, *s.shape).squeeze(0).astype(np.float64).copy(),
                               lambda x, k: x.copy())

    def hm(r):
        return [r.standard_normal((int(r.integers(1, 4)), int(r.integers(2, 6)))) * float(r.choice([1, 100]))], {}
    C["Homeostasis"] = (hm, lambda s, k: fd.Homeostasis(eps=0.0)(s),
                        lambda x, k: (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True))
    return C


def _n(x, k):
    a = k["axis"]
    return x.size if a is None else int(np.prod([x.shape[i] for i in np.atleast_1d(a)]))


def _np_softmax(x, axis):
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def _num_grad(f, xs, w, h=1e-6):
    grads = []
    for i, x in enumerate(xs):
        g = np.zeros_like(x)
        for j in np.ndindex(x.shape):
            a = [y.copy() for y in xs]; b = [y.copy() for y in xs]
            a[i][j] += h; b[i][j] -= h
            g[j] = ((f(a) * w).sum() - (f(b) * w).sum()) / (2 * h)
        grads.append(g)
    return grads


def check_case(name, case, r, device="cpu"):
    """하나의 무작위 경우. 문제 없으면 None, 있으면 설명 문자열"""
    make, f_fd, f_np = case
    made = make(r)
    xs, kw = (made if isinstance(made, tuple) else (made, None))
    call_fd = (lambda ss: f_fd(ss if name in ("concat", "where") else ss[0], kw)) if kw is not None else \
        (lambda ss: f_fd(ss))
    call_np = (lambda arrs: f_np(arrs if name in ("concat", "where") else arrs[0], kw)) if kw is not None else \
        (lambda arrs: f_np(arrs))
    ref = call_np(xs)
    if ref is None:
        return None
    ref = np.asarray(ref, dtype=np.float64)
    sig = [fd.Signal(x.copy(), plastic=True, device=device) for x in xs]
    out = call_fd(sig)
    val = fd.ganglion.backend.numpy(out.data).astype(np.float64)
    if val.shape != ref.shape:
        return f"모양 {val.shape} ≠ {ref.shape} (입력 {[x.shape for x in xs]}, {kw})"
    if not np.allclose(val, ref, rtol=1e-9, atol=1e-9):
        return f"값 최대 차 {np.abs(val - ref).max():.2e} (입력 {[x.shape for x in xs]}, {kw})"
    w = r.standard_normal(ref.shape)
    (out * fd.Signal(w, device=device)).sum().retrograde()
    num = _num_grad(lambda arrs: np.asarray(call_np(arrs), dtype=np.float64), xs, w)
    for i, (s, g) in enumerate(zip(sig, num)):
        got = np.zeros_like(g) if s.retro is None else fd.ganglion.backend.numpy(s.retro)
        if not np.allclose(got, g, rtol=1e-5, atol=1e-6):
            return f"입력 {i} 기울기 최대 차 {np.abs(got - g).max():.2e} (입력 {[x.shape for x in xs]}, {kw})"
    return None


def run(n: int = 40, seed: int = 0, device: str = "cpu", verbose: bool = True) -> dict:
    fails = {}
    for k, (name, case) in enumerate(_cases().items()):
        r = np.random.default_rng([seed, k])
        for t in range(n):
            try:
                msg = check_case(name, case, r, device)
            except Exception as e:                                  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:150]}"
            if msg:
                fails.setdefault(name, []).append(msg)
        if verbose:
            st = "ok  " if name not in fails else f"FAIL {len(fails[name])}/{n}"
            print(f"  {st} {name}" + (f" - {fails[name][0]}" if name in fails else ""), flush=True)
    return fails


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    f = run(n)
    print(f"\n문제 있는 연산 {len(f)}개")
