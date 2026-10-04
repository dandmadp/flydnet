"""희소 배선·커널 퍼징: wiring(CSR) / transmit / propagate (spmm·edge_dot, GPU 전용 커널)를 밀집 행렬 기준과 대조

  python tests/fuzz_sparse.py            # 경우 60가지 (CPU, GPU 있으면 GPU도)
"""
from __future__ import annotations

import sys

import numpy as np

import flydnet as fd
from flydnet.ganglion import backend as B
from flydnet.ganglion import kernels as K
from flydnet.ganglion import physiology as P


def _graph(r):
    n_post, n_pre = int(r.integers(1, 40)), int(r.integers(1, 40))
    dens = float(r.choice([0.0, 0.05, 0.3, 1.0]))
    m = r.random((n_post, n_pre)) < dens
    if r.random() < 0.3:                                    # 빈 행·빈 열
        m[r.integers(0, n_post)] = False
        m[:, r.integers(0, n_pre)] = False
    post, pre = np.nonzero(m)
    shuf = r.permutation(len(post))                         # 순서 섞어 넣기 (wiring이 정렬해야 함)
    return n_post, n_pre, post[shuf], pre[shuf]


def check(r, device):
    n_post, n_pre, post, pre = _graph(r)
    nb = int(r.choice([1, 2, 3, 15, 16, 33, 128]))
    w, order = P.wiring(post, pre, n_post, n_pre, device=device)
    vals = r.standard_normal(len(post)).astype(np.float32)            # 원래 순서의 값
    dense = np.zeros((n_post, n_pre)); dense[post, pre] = vals
    x = r.standard_normal((nb, n_pre)).astype(np.float32)
    # transmit: (B, n_pre) → (B, n_post), 값은 원래 순서[order]
    v = fd.Signal(B.to(vals[order], device), plastic=True)
    xs = fd.Signal(B.to(x, device), plastic=True)
    out = P.transmit(xs, v, w)
    ref = x.astype(np.float64) @ dense.T
    got = B.numpy(out.data)
    if not np.allclose(got, ref, rtol=1e-4, atol=1e-4):
        return f"transmit 값 차 {np.abs(got - ref).max():.2e} (n {n_post}x{n_pre}, 연결 {len(post)}, B {nb})"
    g = r.standard_normal(ref.shape).astype(np.float32)
    (out * fd.Signal(B.to(g, device))).sum().retrograde()
    gx_ref = g.astype(np.float64) @ dense
    gv_ref = (g.T.astype(np.float64) @ x.astype(np.float64))[post, pre][order]   # dL/dw_e = Σ_b g[b, post] x[b, pre]
    if not np.allclose(B.numpy(xs.retro), gx_ref, rtol=1e-4, atol=1e-4):
        return f"transmit 입력 기울기 차 {np.abs(B.numpy(xs.retro) - gx_ref).max():.2e}"
    if len(post) and not np.allclose(B.numpy(v.retro), gv_ref, rtol=1e-4, atol=1e-4):
        return f"transmit 세기 기울기 차 {np.abs(B.numpy(v.retro) - gv_ref).max():.2e}"
    # propagate (시간 시뮬레이션용, (N, B) 배치): 값은 wiring 순서
    M, MT = K.matrices(w, B.to(vals[order], device))
    xn = fd.Signal(B.to(x.T.copy(), device), plastic=True)
    vn = fd.Signal(B.to(vals[order], device), plastic=True)
    o2 = K.propagate(xn, vn, M, MT, w)
    if not np.allclose(B.numpy(o2.data), ref.T, rtol=1e-4, atol=1e-4):
        return f"propagate 값 차 {np.abs(B.numpy(o2.data) - ref.T).max():.2e} (B {nb})"
    (o2 * fd.Signal(B.to(g.T.copy(), device))).sum().retrograde()
    if not np.allclose(B.numpy(xn.retro), gx_ref.T, rtol=1e-4, atol=1e-4):
        return f"propagate 입력 기울기 차 {np.abs(B.numpy(xn.retro) - gx_ref.T).max():.2e} (B {nb})"
    if len(post) and not np.allclose(B.numpy(vn.retro), gv_ref, rtol=1e-4, atol=1e-4):
        return f"propagate 세기 기울기 차 {np.abs(B.numpy(vn.retro) - gv_ref).max():.2e} (B {nb})"
    return None


def run(n: int = 60, seed: int = 0, verbose: bool = True) -> dict:
    devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])
    fails = {}
    for dev in devs:
        r = np.random.default_rng([seed, len(dev)])
        for _ in range(n):
            try:
                msg = check(r, dev)
            except Exception as e:                                  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:160]}"
            if msg:
                fails.setdefault(dev, []).append(msg)
        if verbose:
            print(f"  {dev}: " + ("문제 없음" if dev not in fails else f"{len(fails[dev])}/{n} 실패 - {fails[dev][0]}"), flush=True)
    return fails


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
