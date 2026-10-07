"""장치·연결 장치: 모든 구조물이 CPU = GPU (값·기울기), .to()로 옮긴 뒤에도 같음, 학습 상태가 장치를 따라감,
torch 연결 장치(bridge)의 기울기 = 자체 엔진에서 직접 받은 기울기, torch 옵티마이저 한 스텝 = 자체 엔진 규칙 한 스텝

  python tests/ref_devices.py          # GPU 필요
"""
from __future__ import annotations

import sys

import numpy as np

import flydnet as fd
from flydnet.ganglion import backend as B

if not B.gpu_available():
    print("GPU 없음 - 건너뜀")
    sys.exit(0)
import torch

bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


def close(a, b, tol=1e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and np.allclose(a, b, rtol=tol, atol=tol * max(1.0, float(np.abs(b).max(initial=0))))


r = np.random.default_rng(0)
g = fd.graphs.layered([10, 60, 6], 0.3, weight=30, seed=0)
X = r.uniform(20, 100, (4, 10)).astype(np.float32)
makers = {
    "Projection": (lambda d: fd.Projection(10, 4, seed=0, device=d), X),
    "Neuropil edge+bias": (lambda d: fd.Neuropil(g, "in", "h1", bias=True, device=d), X),
    "Neuropil pair": (lambda d: fd.Neuropil(g, "in", "h1", train="pair", device=d), X),
    "RateEncoder": (lambda d: fd.RateEncoder(10, 10, device=d), X),
    "Homeostasis": (lambda d: fd.Homeostasis(), X),
    "Connectome LIF": (lambda d: fd.Connectome(g, "in", "out", t_ms=30, trainable=True, input_mode="regular", device=d), X),
    "Connectome LIF poisson": (lambda d: fd.Connectome(g, "in", "out", t_ms=30, trainable=True, device=d), X),
    "Connectome graded": (lambda d: fd.Connectome(g, "in", "out", t_ms=5, neuron="graded", trainable=True, device=d), X / 100),
    "Connectome Izhikevich": (lambda d: fd.Connectome(g, "in", "out", t_ms=20, neuron=fd.neurons.Izhikevich(),
                                                      trainable=True, device=d), X),
    "Pathway": (lambda d: fd.Pathway(fd.RateEncoder(10, 10, device=d),
                                     fd.Connectome(g, "in", "out", t_ms=30, trainable=True, device=d),
                                     fd.Homeostasis(), fd.Projection(6, 3, seed=1, device=d)), X),
}


def fwd_bwd(m, x):
    xs = fd.Signal(x, plastic=True, device=m.device if hasattr(m, "device") else "cpu")
    y = m(xs, seed=2) if "seed" in m.forward.__code__.co_varnames else m(xs)
    gr = np.linspace(-1, 1, int(np.prod(y.shape)), dtype=np.float32).reshape(y.shape)
    y.retrograde(B.to(gr, y.device))
    grads = {n: B.numpy(s.retro) for n, s in m.named_synapses() if s.retro is not None}
    return B.numpy(y.data), B.numpy(xs.retro) if xs.retro is not None else None, grads


for name, (make, x) in makers.items():
    a, b = make("cpu"), make("gpu")
    if hasattr(a, "named_synapses"):                                    # 같은 시작값 (무작위 초기값이 장치마다 같은지와 별개로)
        for (_, s1), (_, s2) in zip(a.named_synapses(), b.named_synapses()):
            s2.data = B.to(B.numpy(s1.data), "gpu")
    ya, xa, ga = fwd_bwd(a, x)
    yb, xb, gb = fwd_bwd(b, x)
    ok(f"{name}: CPU = GPU 값", close(ya, yb))
    if xa is not None:
        ok(f"{name}: CPU = GPU 입력 기울기", close(xa, xb, 1e-3))
    ok(f"{name}: CPU = GPU 학습 값 기울기", ga.keys() == gb.keys() and all(close(ga[k], gb[k], 1e-3) for k in ga))
    # CPU에서 만들어 GPU로 옮김 = 처음부터 GPU
    c_ = make("cpu")
    for (_, s1), (_, s2) in zip(a.named_synapses(), c_.named_synapses()):
        s2.data = s1.data.copy()
    c_.to("gpu")
    yc, _, _ = fwd_bwd(c_, x)
    ok(f"{name}: .to('gpu') 뒤 = GPU", close(yc, yb))
    c_.to("cpu")
    yd, _, _ = fwd_bwd(c_, x)
    ok(f"{name}: 다시 .to('cpu') = CPU", close(yd, ya))

# 학습 상태가 장치를 따라감: CPU에서 3스텝 → GPU로 옮겨 3스텝 = 처음부터 6스텝
for Rule in (fd.Adaptive, lambda s, **k: fd.Plasticity(s, momentum=0.9, **k)):
    p1, p2 = fd.Projection(10, 3, seed=0, device="cpu"), fd.Projection(10, 3, seed=0, device="cpu")
    r1, r2 = Rule(p1.named_synapses(), rate=0.01), Rule(p2.named_synapses(), rate=0.01)
    y = np.array([0, 1, 2, 0])
    for t in range(6):
        if t == 3:
            p2.to("gpu")
        for p, rule in ((p1, r1), (p2, r2)):
            rule.clear(); fd.surprise(p(X), y).retrograde(); rule.step()
    ok(f"규칙 상태가 장치를 따라감 ({type(r1).__name__})", close(p1.weight.numpy(), p2.weight.numpy(), 1e-5))

# torch 연결 장치: bridge 기울기 = 자체 엔진 기울기, torch.optim 한 스텝 = 자체 엔진 Plasticity 한 스텝
for dev in ("cpu", "gpu"):
    L1 = fd.Connectome(g, "in", "out", t_ms=30, trainable=True, input_mode="regular", device=dev)
    L2 = fd.Connectome(g, "in", "out", t_ms=30, trainable=True, input_mode="regular", device=dev)
    br = fd.torch.bridge(L2, seed=2)
    tdev = "cuda" if dev == "gpu" else "cpu"
    xt = torch.tensor(X, device=tdev, requires_grad=True)
    yt = br(xt)
    w = torch.linspace(-1, 1, yt.numel(), device=tdev).reshape(yt.shape)
    (yt * w).sum().backward()
    xs = fd.Signal(X, plastic=True, device=dev)
    ys = L1(xs, seed=2)
    ys.retrograde(B.to(w.cpu().numpy(), dev))
    ok(f"[{dev}] bridge 출력 = 엔진", close(yt.detach().cpu().numpy(), ys.numpy()))
    ok(f"[{dev}] bridge 입력 기울기 = 엔진", close(xt.grad.cpu().numpy(), B.numpy(xs.retro)))
    pt = dict(br.named_parameters())["log_scale"]
    ok(f"[{dev}] bridge 학습 값 기울기 = 엔진", close(pt.grad.cpu().numpy(), B.numpy(L1.log_scale.retro)))
    opt = torch.optim.SGD(br.parameters(), lr=0.1)
    opt.step()
    rule = fd.Plasticity(L1.synapses(), rate=0.1)
    rule.step()
    ok(f"[{dev}] torch.optim 스텝이 엔진 값에 반영", close(B.numpy(L2.log_scale.data), B.numpy(L1.log_scale.data)))
    with torch.no_grad():
        y_after = br(torch.tensor(X, device=tdev)).cpu().numpy()
    with fd.quiescent():
        ok(f"[{dev}] 스텝 뒤 순전파 = 엔진", close(y_after, L1(X, seed=2).numpy()))

print(f"장치·연결 장치: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
