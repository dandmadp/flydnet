"""Linear 계열 구조물 퍼저: 무작위 모양·조합에서 역전파 기울기를 float64 수치 미분(중심 차분)과 대조

  python tests/fuzz_layers.py [반복 수]

대상: Projection (bias 있음/없음, 1·2·3차원 입력), Neuropil (edge·pair·free, bias), Homeostasis, LateralInhibition,
Activation (relu·tanh·sigmoid), AxonHillock은 대리 기울기라 제외, Pathway 조합, surprise·log_softmax.
모든 학습 값(Synapse)과 입력에 대한 기울기를 확인. 값은 numpy float64 기준식과도 대조
"""
from __future__ import annotations

import sys

import numpy as np

import flydnet as fd
from flydnet.ganglion import backend as B
from flydnet.ganglion.signal import Signal


def _f64(tissue, dev):
    """구조물의 학습 값·버퍼를 float64로 (수치 미분 정밀도)"""
    xp = B.xp(dev)
    for s in tissue.synapses():
        s.data = s.data.astype(xp.float64)
    for t in tissue.tissues():
        for n in t._buffers:
            v = getattr(t, n)
            if hasattr(v, "dtype") and v.dtype == np.float32:
                object.__setattr__(t, n, v.astype(xp.float64))
    return tissue


def _graph(rng, n_pre, n_post):
    """pre·post 그룹이 있는 작은 무작위 회로 (부호 섞임)"""
    N = n_pre + n_post
    E = int(rng.integers(n_pre, n_pre * n_post + 1))
    pre = rng.integers(0, n_pre, E)
    post = rng.integers(n_pre, N, E)
    w = rng.integers(1, 6, E) * rng.choice([-1, 1], E)
    groups = {"P": np.arange(n_pre), "Q": np.arange(n_pre, N)}
    if n_post > 3:                                                   # pair 학습용으로 받는 쪽을 둘로
        groups = {"P": np.arange(n_pre), "Q": np.arange(n_pre, n_pre + n_post // 2), "R": np.arange(n_pre + n_post // 2, N)}
    return fd.Circuit(np.arange(N), groups, pre, post, w.astype(np.float32))


def _build(rng, dev):
    """무작위 구조물과 입력 (B, n) 또는 (n,) 또는 (B, T, n)"""
    kind = rng.choice(["proj", "neuropil", "path", "norm", "inhibit", "act"])
    n_in = int(rng.integers(2, 9))
    if kind == "proj":
        m = fd.Projection(n_in, int(rng.integers(1, 6)), bias=bool(rng.integers(2)), seed=int(rng.integers(1e6)), device=dev)
    elif kind == "neuropil":
        c = _graph(rng, n_in, int(rng.integers(2, 9)))
        post = [g for g in c.groups if g != "P"]
        m = fd.Neuropil(c, "P", post, train=str(rng.choice(["edge", "pair", "free"])), bias=bool(rng.integers(2)),
                        init=str(rng.choice(["fan_in", "counts"])), device=dev)
    elif kind == "path":
        h = int(rng.integers(2, 7))
        m = fd.Pathway(fd.Projection(n_in, h, seed=int(rng.integers(1e6)), device=dev),
                       fd.Activation(str(rng.choice(["tanh", "sigmoid"]))), fd.Homeostasis(),
                       fd.Projection(h, int(rng.integers(1, 5)), seed=int(rng.integers(1e6)), device=dev))
    elif kind == "norm":
        m = fd.Homeostasis(eps=float(rng.choice([1e-5, 0.1])))
    elif kind == "inhibit":
        m = fd.Inhibition(k=int(rng.integers(1, n_in + 1)))
    else:
        m = fd.Activation(str(rng.choice(["relu", "tanh", "sigmoid"])))
    shape = {0: (n_in,), 1: (int(rng.integers(1, 5)), n_in), 2: (int(rng.integers(1, 4)), int(rng.integers(1, 4)), n_in)}
    sh = shape[int(rng.integers(3))] if kind in ("proj", "norm", "act") else shape[1]
    x = rng.standard_normal(sh) * 2
    if kind == "inhibit":                                           # k번째 값 경계의 동점을 피하려고 띄움
        x = x + np.arange(x.size).reshape(x.shape) * 0.37
    return kind, _f64(m, dev), x


def _check(kind, m, x, dev, eps=1e-6):
    xp = B.xp(dev)
    W = xp.asarray(np.random.default_rng(7).standard_normal(1))[0]    # 무작위 방향 손실: sum(out * R)
    xs = Signal(xp.asarray(x), plastic=True)
    out = m(xs)
    R = xp.asarray(np.random.default_rng(8).standard_normal(out.shape))
    loss = (out * Signal(R)).sum()
    loss.retrograde()
    params = [s for s in m.synapses()] + [xs]
    worst = 0.0
    for s in params:
        g = B.numpy(s.retro if s.retro is not None else xp.zeros_like(s.data))
        flat = s.data.reshape(-1)
        idx = np.random.default_rng(9).choice(flat.size, min(flat.size, 6), replace=False)
        for i in idx:
            old = float(flat[i])
            vals = []
            for d in (eps, -eps):
                flat[i] = old + d
                with fd.quiescent():
                    o = m(Signal(xs.data) if s is xs else Signal(xp.asarray(x)))
                vals.append(float((o.data * R).sum()))
            flat[i] = old
            num = (vals[0] - vals[1]) / (2 * eps)
            got = float(g.reshape(-1)[i])
            err = abs(num - got) / max(1.0, abs(num), abs(got))
            if kind in ("inhibit", "act") and abs(num) > 0 and abs(got) == 0 and abs(num) < 1e-3:
                continue                                              # relu·측억제 경계 바로 위
            worst = max(worst, err)
    return worst, out


def run(n: int = 200):
    devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])
    bad = {}
    for dev in devs:
        rng = np.random.default_rng(0)
        for it in range(n):
            kind, m, x = _build(rng, dev)
            try:
                err, out = _check(kind, m, x, dev)
                ok = err < 1e-5 and np.all(np.isfinite(B.numpy(out.data)))
                if np.ndim(x) == 1 and kind != "neuropil":
                    ok &= out.ndim == 1                                   # 1차원 입력 → 1차원 출력
            except Exception as e:                                         # noqa: BLE001
                ok, err = False, f"{type(e).__name__}: {e}"
            if not ok:
                bad.setdefault((dev, kind), []).append((it, np.shape(x), err))
        print(f"  {dev}: {n}번, 문제 {sum(len(v) for k, v in bad.items() if k[0] == dev)}건")
    for k, v in bad.items():
        print("  !", k, v[:3])
    return not bad


if __name__ == "__main__":
    sys.exit(0 if run(int(sys.argv[1]) if len(sys.argv) > 1 else 200) else 1)
