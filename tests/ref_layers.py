"""층 대 torch: Projection = nn.Linear, Neuropil = 연결 칸만 남긴 밀집 행렬 (train edge·pair·free·None, fan_in·counts),
Inhibition·AxonHillock·Activation = torch 함수, 인코더·KCExpansion = 수식, MushroomBodyOutput = AssocReadout

  python tests/ref_layers.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import flydnet as fd
from flydnet.ganglion import backend as B

torch.set_default_dtype(torch.float32)
bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


def close(a, b, tol=1e-5):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and np.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(np.abs(b).max(initial=0))))


r = np.random.default_rng(0)
devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])

# 1) Projection = nn.Linear (값·입력 기울기·가중치 기울기·편향), 1D·3D 입력
for dev in devs:
    for shape in ((5, 7), (7,), (2, 3, 7)):
        p = fd.Projection(7, 4, seed=1, device=dev)
        lin = torch.nn.Linear(7, 4)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(p.weight.numpy())); lin.bias.copy_(torch.tensor(p.bias.numpy()))
        x = r.standard_normal(shape).astype(np.float32)
        xs = fd.Signal(x, plastic=True, device=dev)
        y = p(xs)
        g = r.standard_normal(y.shape).astype(np.float32)
        y.retrograde(g)
        xt = torch.tensor(x, requires_grad=True)
        yt = lin(xt)
        (yt * torch.tensor(g)).sum().backward()
        ok(f"[{dev}] Projection {shape} 값", close(y.numpy(), yt.detach().numpy()))
        ok(f"[{dev}] Projection {shape} 입력 기울기", close(B.numpy(xs.retro), xt.grad.numpy()))
        ok(f"[{dev}] Projection {shape} 가중치 기울기", close(B.numpy(p.weight.retro), lin.weight.grad.numpy()))
        ok(f"[{dev}] Projection {shape} 편향 기울기", close(B.numpy(p.bias.retro), lin.bias.grad.numpy()))
    w = fd.Projection(400, 300, seed=0, device=dev).weight.numpy()
    ok(f"[{dev}] Projection 초기값 ±1/√n_in 균등", abs(w.max() - 0.05) < 0.002 and abs(w.min() + 0.05) < 0.002
       and abs(w.std() - 0.05 / np.sqrt(3)) < 0.001)

# 2) Neuropil = 연결 칸만 남긴 밀집 행렬
g = fd.graphs.stochastic_block({"A": 30, "B": 20, "C": 10}, {("A", "B"): 0.3, ("C", "B"): 0.5, ("A", "A"): 0.1},
                               weight=1.0, inhibitory=0.3, seed=2)
for dev in devs:
    for train in ("edge", "pair", "free", None):
        for init in ("fan_in", "counts"):
            n = fd.Neuropil(g, ["A", "C"], "B", train=train, init=init, bias=True, device=dev)
            if train in ("edge", "pair"):
                n.log_scale.data = B.to(r.standard_normal(n.log_scale.shape).astype(np.float32) * 0.3, dev)
            if n.bias is not None:
                n.bias.data = B.to(r.standard_normal(n.bias.shape).astype(np.float32), dev)
            Wd = n.dense()
            # 밀집 기준: 원래 회로에서 직접 (A·C → B 연결의 시냅스 수, fan_in이면 받는 뉴런마다 √Σw²로 나눔)
            pre_idx = np.concatenate([g.groups["A"], g.groups["C"]])
            post_idx = g.groups["B"]
            ref = np.zeros((len(post_idx), len(pre_idx)))
            lp = {v: i for i, v in enumerate(pre_idx)}
            lq = {v: i for i, v in enumerate(post_idx)}
            for a_, b_, w_ in zip(g.pre, g.post, g.weight):
                if a_ in lp and b_ in lq:
                    ref[lq[b_], lp[a_]] += w_
            if init == "fan_in":
                ref = ref / np.sqrt((ref ** 2).sum(1, keepdims=True)).clip(1e-30)
            with fd.quiescent():
                base = Wd if train in (None, "free") else None
            if train in (None, "free"):
                ok(f"[{dev}] Neuropil {train}/{init} 밀집 = 회로", close(Wd, ref))
            else:                                                    # 배율 = exp(log_scale): 부호는 그대로
                ok(f"[{dev}] Neuropil {train}/{init} 부호 유지", np.array_equal(np.sign(Wd), np.sign(ref)))
            x = r.standard_normal((4, len(pre_idx))).astype(np.float32)
            xs = fd.Signal(x, plastic=True, device=dev)
            y = n(xs)
            gr = r.standard_normal(y.shape).astype(np.float32)
            y.retrograde(gr)
            Wt = torch.tensor(Wd, requires_grad=True)
            xt = torch.tensor(x, requires_grad=True)
            yt = xt @ Wt.T + torch.tensor(n.bias.numpy())
            (yt * torch.tensor(gr)).sum().backward()
            ok(f"[{dev}] Neuropil {train}/{init} 값", close(y.numpy(), yt.detach().numpy()))
            ok(f"[{dev}] Neuropil {train}/{init} 입력 기울기", close(B.numpy(xs.retro), xt.grad.numpy()))
            if train == "free":                                       # 연결 값 기울기 = 밀집 기울기의 연결 칸
                post_e, pre_e = B.numpy(n.wiring.post), B.numpy(n.wiring.pre)
                ok(f"[{dev}] Neuropil free 연결 기울기", close(B.numpy(n.values_.retro), Wt.grad.numpy()[post_e, pre_e]))
            if train == "edge":                                       # d/d log_scale = W 기울기 x 값
                post_e, pre_e = B.numpy(n.wiring.post), B.numpy(n.wiring.pre)
                ok(f"[{dev}] Neuropil edge 배율 기울기", close(B.numpy(n.log_scale.retro),
                                                         Wt.grad.numpy()[post_e, pre_e] * Wd[post_e, pre_e]))

# 3) 측억제·발화·활성화 = torch
x = np.round(r.standard_normal((4, 10)) * 2) / 2
for k in (1, 3, 10):
    a = fd.Inhibition(k=k)(x).numpy()
    kth = -np.sort(-x, 1)[:, k - 1:k]
    ok(f"Inhibition k={k} = k번째 이상 남김", np.array_equal(a, x * (x >= kth)))
for kind, f in (("relu", torch.relu), ("tanh", torch.tanh), ("sigmoid", torch.sigmoid)):
    ok(f"Activation {kind}", close(fd.Activation(kind)(x.astype(np.float32)).numpy(), f(torch.tensor(x, dtype=torch.float32)).numpy()))
ok("AxonHillock", np.array_equal(fd.AxonHillock(0.3)(x).numpy(), (x > 0.3).astype(np.float32)))

# 4) 인코더·KCExpansion = 수식
X = r.random((5, 30)).astype(np.float32) * 3
a = fd.RateEncoder(30, 12, device="cpu")
P = np.asarray(a.P)
ref = X @ P.T
ok("RateEncoder = 투영 뒤 최댓값 정규화", close(a(X).numpy(), ref / ref.max(1, keepdims=True) * 100))
ok("RateEncoder 시료마다 최댓값 = max_rate", close(a(X).numpy().max(1), np.full(5, 100.0)))
ok("RateEncoder 음수는 0", a(-X).numpy().max() == 0)
from flydnet.ganglion.physiology import kenyon_code
try:
    from flydnet.data import missing
    has_fw = not missing("flywire")
except Exception:                                                        # noqa: BLE001
    has_fw = False
if has_fw:
    mb = fd.flywire()
    ka = fd.KCExpansion(mb, n_in=30, device="cpu")
    pn = X @ np.asarray(ka.proj).T
    d = (pn - pn.mean(1, keepdims=True)) @ np.asarray(ka.W).T
    ref = np.zeros_like(d)
    for i in range(len(d)):
        top = np.argsort(-d[i])[:ka.k]
        ref[i, top] = np.maximum(d[i, top], 0)
    ok("KCExpansion = 투영 → 평균 빼기 → 실제 배선 → 상위 k", close(ka(X).numpy(), ref, 1e-4))
    ok("KCExpansion 켜진 KC 수 = k", np.all((ka(X).numpy() > 0).sum(1) <= ka.k))

# 5) MushroomBodyOutput (구조물판) = AssocReadout (같은 순서로 배우면 같은 원형)
from flydnet.ganglion.tissue import MushroomBodyOutput
A = r.random((40, 20)).astype(np.float32)
y = r.integers(0, 4, 40)
m1 = MushroomBodyOutput(20, 4, per_class=3, device="cpu")
m2 = fd.AssocReadout(20, 4, per_class=3, device="cpu")
for i in range(0, 40, 8):
    m1.learn(A[i:i + 8], y[i:i + 8])
    m2.step(A[i:i + 8], y[i:i + 8])
ok("MushroomBodyOutput = AssocReadout", close(np.asarray(m1.prototypes), np.asarray(m2.W))
   and np.array_equal(np.asarray(m1.predict(A)), np.asarray(m2.predict(A))))

print(f"층: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
