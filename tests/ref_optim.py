"""최적화기 대 torch.optim: 같은 시작값·같은 기울기 순서로 50스텝, 스텝마다 값 비교. clip은 clip_grad_norm_과,
일부 시냅스에만 기울기가 오는 스텝(None)·학습률을 도중에 바꾸는 스케줄도

  python tests/ref_optim.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import flydnet as fd

torch.set_default_dtype(torch.float64)
bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


def run(make_fd, make_t, steps=50, clip=None, skip_some=False, schedule=False, seed=0):
    r = np.random.default_rng(seed)
    shapes = [(3, 4), (5,), (2, 2, 2)]
    init = [r.standard_normal(s) for s in shapes]
    syn = [fd.Synapse(x.copy(), device="cpu") for x in init]
    tp = [torch.nn.Parameter(torch.tensor(x.copy())) for x in init]
    rule, opt = make_fd(syn), make_t(tp)
    if clip is not None:
        rule.clip = clip
    worst = 0.0
    for t in range(steps):
        grads = [r.standard_normal(s) * (10 if t % 7 == 0 else 1) for s in shapes]
        live = [not (skip_some and (t + i) % 3 == 0) for i in range(len(shapes))]
        if schedule:
            lr = 0.05 * 0.5 * (1 + np.cos(np.pi * t / steps))
            rule.rate = lr
            for gr in opt.param_groups:
                gr["lr"] = lr
        rule.clear()
        opt.zero_grad(set_to_none=True)
        for s, p, g, on in zip(syn, tp, grads, live):
            if on:
                s.retro = g.copy()
                p.grad = torch.tensor(g.copy())
        if clip is not None:
            torch.nn.utils.clip_grad_norm_([p for p, on in zip(tp, live) if on], clip)
        rule.step()
        opt.step()
        worst = max(worst, max(float(np.abs(s.numpy() - p.detach().numpy()).max()) for s, p in zip(syn, tp)))
    return worst


cases = {
    "SGD": (lambda s: fd.Plasticity(s, rate=0.05), lambda p: torch.optim.SGD(p, lr=0.05)),
    "SGD 관성": (lambda s: fd.Plasticity(s, rate=0.05, momentum=0.9), lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    "SGD 감쇠+관성": (lambda s: fd.Plasticity(s, rate=0.05, momentum=0.8, decay=0.01),
                   lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.8, weight_decay=0.01)),
    "Adam": (lambda s: fd.Adaptive(s, rate=0.01), lambda p: torch.optim.Adam(p, lr=0.01)),
    "Adam betas·eps": (lambda s: fd.Adaptive(s, rate=0.02, betas=(0.8, 0.99), eps=1e-6),
                       lambda p: torch.optim.Adam(p, lr=0.02, betas=(0.8, 0.99), eps=1e-6)),
    "AdamW": (lambda s: fd.Adaptive(s, rate=0.01, decay=0.1), lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1)),
}
for name, (mf, mt) in cases.items():
    for label, kw in (("", {}), (" + clip", dict(clip=1.0)), (" + 일부만 기울기", dict(skip_some=True)),
                      (" + 학습률 스케줄", dict(schedule=True))):
        w = run(mf, mt, **kw)
        # clip: torch는 최대값 / (크기 + 1e-6), flydnet은 + 1e-12 - 반올림 수준의 규약 차이라 허용 오차를 넓힘
        ok(f"{name}{label}", w < (1e-6 if "clip" in label else 1e-9), f"최대 차 {w:.3g}")

# fd.train의 학습 루프 = torch로 같은 루프 (Projection, 코사인 학습률, clip, AdamW 감쇠)
X = np.random.default_rng(1).standard_normal((40, 6)).astype(np.float32)
y = (X[:, 0] > X[:, 1]).astype(int)
p = fd.Projection(6, 2, seed=3, device="cpu")
w0, b0 = p.weight.numpy().copy(), p.bias.numpy().copy()
h = fd.train(p, X, y, epochs=3, batch=8, rate=0.01, decay=0.05, clip=0.5, seed=7, verbose=False)
lin = torch.nn.Linear(6, 2)
with torch.no_grad():
    lin.weight.copy_(torch.tensor(w0)); lin.bias.copy_(torch.tensor(b0))
opt = torch.optim.AdamW(lin.parameters(), lr=0.01, weight_decay=0.05)
rng = np.random.default_rng(7)
for ep in range(3):
    for gr in opt.param_groups:
        gr["lr"] = 0.01 * 0.5 * (1 + np.cos(np.pi * ep / 3))
    perm = rng.permutation(40)
    for i in range(0, 40, 8):
        j = perm[i:i + 8]
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(torch.tensor(X[j], dtype=torch.float64)), torch.tensor(y[j])).backward()
        torch.nn.utils.clip_grad_norm_(lin.parameters(), 0.5)
        opt.step()
ok("fd.train = torch 루프 (AdamW·코사인·clip)", np.allclose(p.weight.numpy(), lin.weight.detach().numpy(), atol=1e-5),
   float(np.abs(p.weight.numpy() - lin.weight.detach().numpy()).max()))

print(f"최적화기: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
