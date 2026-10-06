"""생리 함수 대 참조: 시냅스 전달(희소) 대 밀집 torch, put·put_columns·add_columns, 상태 없는 난수의 균등성·CPU=GPU,
포아송 입력의 실제 발화율 대 목표 (이항 검정), 정규 입력(regular)의 스파이크 수, KC 부호화 대 직접 계산

  python tests/ref_physiology.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from scipy import stats

import flydnet as fd
from flydnet.ganglion import backend as B, kernels as K, physiology as P

torch.set_default_dtype(torch.float64)
bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


r = np.random.default_rng(0)
devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])

# 1) transmit: 무작위 희소 연결 (중복 없음) 대 밀집 행렬곱, 값·입력 기울기·연결 기울기
for dev in devs:
    for it in range(20):
        n_pre, n_post = int(r.integers(1, 40)), int(r.integers(1, 40))
        E = int(r.integers(1, n_pre * n_post + 1))
        flat = r.choice(n_pre * n_post, E, replace=False)
        post, pre = flat // n_pre, flat % n_pre
        w, order = P.wiring(post, pre, n_post, n_pre, device=dev)
        vals = r.standard_normal(E)
        Bn = int(r.integers(1, 6))
        x = r.standard_normal((Bn, n_pre))
        xs = fd.Signal(x, plastic=True, device=dev)
        vs = fd.Signal(vals[order], plastic=True, device=dev)
        y = P.transmit(xs, vs, w)
        g = r.standard_normal((Bn, n_post))
        y.retrograde(g)
        W = torch.zeros(n_post, n_pre)
        vt = torch.tensor(vals, requires_grad=True)
        W = W.index_put((torch.tensor(post), torch.tensor(pre)), vt)
        xt = torch.tensor(x, requires_grad=True)
        yt = xt @ W.T
        (yt * torch.tensor(g)).sum().backward()
        ok(f"[{dev}] transmit 값", np.allclose(y.numpy(), yt.detach().numpy(), atol=1e-10))
        ok(f"[{dev}] transmit 입력 기울기", np.allclose(B.numpy(xs.retro), xt.grad.numpy(), atol=1e-10))
        ok(f"[{dev}] transmit 연결 기울기", np.allclose(B.numpy(vs.retro), vt.grad.numpy()[order], atol=1e-10))
    try:
        P.wiring([0, 0], [1, 1], 2, 2)
        ok("같은 연결 두 번 → 오류", False)
    except ValueError:
        pass

# 2) put / put_columns / add_columns 기울기
x = r.standard_normal((3, 5))
idx = np.array([1, 3])
v = fd.Signal(r.standard_normal(2), plastic=True)
out = P.put(np.zeros(5), idx, v)
out.retrograde(np.arange(5.0))
ok("put 기울기 = 그 자리", np.allclose(v.retro, [1.0, 3.0]))
xs, vs = fd.Signal(x, plastic=True), fd.Signal(r.standard_normal((3, 2)), plastic=True)
P.put_columns(xs, idx, vs).retrograde(np.ones((3, 5)))
ok("put_columns: 바뀐 열의 x 기울기 0", np.allclose(xs.retro[:, idx], 0) and np.allclose(vs.retro, 1))
xs, vs = fd.Signal(x, plastic=True), fd.Signal(r.standard_normal((3, 2)), plastic=True)
P.add_columns(xs, idx, vs).retrograde(np.ones((3, 5)))
ok("add_columns", np.allclose(xs.retro, 1) and np.allclose(vs.retro, 1))

# 3) 상태 없는 난수: 균등 (KS 검정), seed·스텝마다 다름, CPU = GPU 비트 단위
u = P.hash_uniform(np, 7, 3, (200000,))
ok("hash_uniform 균등 (KS p > 0.001)", stats.kstest(u, "uniform").pvalue > 1e-3, stats.kstest(u, "uniform").pvalue)
ok("hash_uniform 범위 [0, 1)", u.min() >= 0 and u.max() < 1)
u2 = P.hash_uniform(np, 7, 4, (200000,))
ok("스텝마다 다름 (상관 거의 0)", abs(np.corrcoef(u, u2)[0, 1]) < 0.01)
lag = np.corrcoef(u[:-1], u[1:])[0, 1]
ok("이웃 칸 상관 거의 0", abs(lag) < 0.01, lag)
z = P.hash_uniform(np, 0, 0, (1,))
ok("seed 0·스텝 0·칸 0이 0이 아님", z[0] > 1e-6, z)
if "gpu" in devs:
    ug = B.numpy(P.hash_uniform(B.xp("gpu"), 7, 3, (200000,)))
    ok("hash_uniform CPU = GPU", np.array_equal(u, ug))

# 4) 포아송 입력 스파이크: 칸마다 확률 p로 1 (이항 검정), CPU = GPU
for dev in devs:
    xp = B.xp(dev)
    p = xp.asarray(np.linspace(0.001, 0.3, 50)[:, None] * np.ones((1, 4)), dtype=xp.float32)    # (n_in, B)
    cnt = sum(B.numpy(K.poisson_spikes(xp, 11, s, p)) for s in range(4000))
    pv = np.asarray(B.numpy(p))
    z = (cnt - 4000 * pv) / np.sqrt(4000 * pv * (1 - pv))
    ok(f"[{dev}] 포아송 스파이크 수 (|z| < 4.5)", np.abs(z).max() < 4.5, np.abs(z).max())
if "gpu" in devs:
    p = np.random.default_rng(1).random((30, 5)).astype(np.float32) * 0.5
    a = [K.poisson_spikes(np, 3, s, p) for s in range(50)]
    b = [B.numpy(K.poisson_spikes(B.xp("gpu"), 3, s, B.to(p, "gpu"))) for s in range(50)]
    ok("포아송 CPU = GPU", all(np.array_equal(x_, y_) for x_, y_ in zip(a, b)))

# 5) 층 수준: 입력 뉴런의 실제 발화율 대 목표 (poisson·regular), 불응기 없는 입력 뉴런
g = fd.graphs.layered([20, 30, 5], 0.2, weight=30, seed=0)
rates = np.linspace(5, 200, 20).astype(np.float32)[None].repeat(64, 0)
for mode in ("poisson", "regular"):
    L = fd.Connectome(g, "in", "out", t_ms=500, dt=0.1, input_mode=mode, device="cpu")
    with fd.quiescent():
        out = L(rates, seed=0, return_all=True).numpy()[:, g.groups["in"]].mean(0)
    # 입력 뉴런 발화율 = 자극 발화율. poisson: 전체 스파이크 수가 포아송 잡음 안 (|z| < 4, 시행 64 x 0.5 s).
    # regular: 일정 간격이라 시행당 정수로 끊김 (5 Hz x 0.5 s = 2.5개 → 2 또는 3) - 1개 이내
    if mode == "poisson":
        n = rates[0] * 0.5 * 64
        z = (out * 0.5 * 64 - n) / np.sqrt(n)
        ok("입력 뉴런 발화율 = 자극 (poisson)", np.abs(z).max() < 4, z.round(2))
    else:
        err = np.abs(out - rates[0]) * 0.5
        ok("입력 뉴런 발화율 = 자극 (regular)", err.max() <= 1.0, (rates[0].round(), out.round(1)))

# 6) KC 부호화: 직접 계산과 같음 (상위 k, 음수 0)
W = np.abs(r.standard_normal((50, 12))).astype(np.float32)
xk = r.random((4, 12)).astype(np.float32)
code = P.kenyon_code(xk, W, k=5, center=True)
a = xk - xk.mean(1, keepdims=True)
d = a @ W.T
ref = np.zeros_like(d)
for i in range(4):
    top = np.argsort(-d[i])[:5]
    ref[i, top] = np.maximum(d[i, top], 0)
ok("kenyon_code", np.allclose(code, ref, atol=1e-5))

print(f"생리 함수: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
