"""역전파 엔진 구조 대 torch: 같은 값을 여러 곳에서 쓰는 그래프, keep로 두 번, 경로를 푼 뒤, checkpoint, 출력 여러 개,
깊은 그래프, quiescent 중첩. 값·기울기를 torch와 비교하고, 막아야 할 사용은 오류인지

  python tests/ref_engine.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import flydnet as fd
from flydnet.ganglion.signal import checkpoint, multi_output, quiescent, learning_enabled

torch.set_default_dtype(torch.float64)
bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


def close(a, b):
    return np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=1e-10, atol=1e-12)


def S(x):
    return fd.Signal(np.array(x, np.float64), plastic=True)


def T(x):
    return torch.tensor(np.array(x, np.float64), requires_grad=True)


r = np.random.default_rng(0)
xv, wv = r.standard_normal((3, 4)), r.standard_normal((4, 2))

# 1) 다이아몬드: x를 세 갈래로 쓰고 다시 합침
x, w = S(xv), S(wv)
h = x @ w
loss = ((h * h).sum() + (h.exp() * x[:, :2]).sum() + x.sum()) * 0.5
loss.retrograde()
xt, wt = T(xv), T(wv)
ht = xt @ wt
lt = ((ht * ht).sum() + (ht.exp() * xt[:, :2]).sum() + xt.sum()) * 0.5
lt.backward()
ok("다이아몬드 값", close(loss.numpy(), lt.item()))
ok("다이아몬드 x 기울기", close(x.retro, xt.grad))
ok("다이아몬드 w 기울기", close(w.retro, wt.grad))

# 2) keep=True로 두 번 → 2배로 쌓임, 그 뒤 풀린 경로로 다시 → 오류
x = S(xv)
l1 = (x * x).sum()
l1.retrograde(keep=True)
l1.retrograde()
ok("keep 두 번 = 2배", close(x.retro, 4 * xv))
try:
    l1.retrograde()
    ok("푼 경로로 다시 → 오류", False)
except RuntimeError:
    pass

# 3) 중간값을 공유하는 두 손실: 첫 번째가 풀면 두 번째는 오류 (조용히 기울기가 사라지면 안 됨)
x = S(xv)
h = x * 2
a, b = (h * h).sum(), h.sum()
a.retrograde()
try:
    b.retrograde()
    ok("공유 중간값 두 번째 손실 → 오류", False)
except RuntimeError:
    pass
x = S(xv)
h = x * 2
((h * h).sum() + h.sum()).retrograde()
ok("손실을 더해 한 번에", close(x.retro, 8 * xv + 2))

# 4) 잎 기울기 누적 + 사본 (같은 배열이 두 잎에 가도 따로)
x, y = S(xv), S(xv)
(x + y).sum().retrograde()
x.retro[0, 0] = 99.0
ok("두 잎이 기울기 배열을 공유하지 않음", y.retro[0, 0] == 1.0)
(x * 3).sum().retrograde()
ok("잎 기울기 누적", close(x.retro[1:], np.ones((2, 4)) * 4))

# 5) retro를 Signal·리스트로, 모양이 다르면 오류
x = S(xv)
(x * 2).retrograde(fd.Signal(np.ones((3, 4))))
ok("retro Signal", close(x.retro, 2 * np.ones((3, 4))))
x = S(xv)
try:
    (x * 2).retrograde(np.ones((4, 3)))
    ok("retro 모양 다름 → 오류", False)
except ValueError:
    pass
try:
    fd.Signal(xv).sum().retrograde()
    ok("plastic 아닌 신호 → 오류", False)
except RuntimeError:
    pass

# 6) quiescent 중첩·예외 뒤 복구
with quiescent():
    with quiescent():
        pass
    ok("중첩 quiescent 안", not learning_enabled())
ok("quiescent 밖", learning_enabled())
try:
    with quiescent():
        raise KeyError
except KeyError:
    pass
ok("예외 뒤 복구", learning_enabled())
with quiescent():
    y = S(xv) * 2
ok("quiescent 안 연산은 경로 없음", y._back is None)

# 7) checkpoint: 바깥 plastic 신호(w)를 쓰는 함수, 구간 두 개를 이어서 → 일반 계산과 같은 기울기
w = S(wv[:, 0])                                                   # (4,)
x0 = S(xv)


def seg(a):
    return ((a * w).tanh() + a * 0.5,)


(a1,) = checkpoint(seg, x0)
(a2,) = checkpoint(seg, a1)
(a2 * a2).sum().retrograde()
wt, xt = T(wv[:, 0]), T(xv)
b1 = (xt * wt).tanh() + xt * 0.5
b2 = (b1 * wt).tanh() + b1 * 0.5
(b2 * b2).sum().backward()
ok("checkpoint 입력 기울기", close(x0.retro, xt.grad))
ok("checkpoint 바깥 신호 기울기", close(w.retro, wt.grad))

# 8) quiescent 안에서 retrograde해도 checkpoint는 다시 계산할 경로를 만듦
w = S(wv[:, 0])
x0 = S(xv)
(a1,) = checkpoint(seg, x0)
loss = (a1 * a1).sum()
with quiescent():
    loss.retrograde()
ok("quiescent 안 retrograde + checkpoint", x0.retro is not None and w.retro is not None)

# 9) 출력 여러 개 중 일부만 손실에 쓰임
x = S(xv)
o1, o2 = multi_output([x], [xv * 2, xv * 3], back=lambda gs: ((0 if gs[0] is None else gs[0] * 2) +
                                                               (0 if gs[1] is None else gs[1] * 3),))
o2.sum().retrograde()
ok("출력 여러 개 중 하나만", close(x.retro, 3 * np.ones_like(xv)))

# 10) 깊은 그래프 (재귀 없이)
x = S(np.array([1.0]))
h = x
for _ in range(20000):
    h = h * 1.0001
h.sum().retrograde()
ok("깊이 2만", close(x.retro, [1.0001 ** 20000]))

# 11) 같은 신호를 두 번 넣는 연산 (x * x, concat([x, x]))
x = S(xv)
(x * x).sum().retrograde()
ok("x * x", close(x.retro, 2 * xv))
x = S(xv)
fd.ganglion.concat([x, x]).sum().retrograde()
ok("concat([x, x])", close(x.retro, 2 * np.ones_like(xv)))
x = S(xv)
fd.ganglion.where(xv > 0, x, x * 2).sum().retrograde()
ok("where(x, 2x)", close(x.retro, np.where(xv > 0, 1, 2)))

# 12) Synapse가 학습 중 값이 바뀌어도 다음 순전파는 새 값 (경로가 옛 값을 붙잡지 않음)
p = fd.Projection(4, 2, seed=0, device="cpu")
rule = fd.Plasticity(p.synapses(), rate=0.1)
for _ in range(3):
    rule.clear(); fd.surprise(p(xv.astype(np.float32)), [0, 1, 0]).retrograde(); rule.step()
tp = torch.nn.Linear(4, 2)
with torch.no_grad():
    w0 = fd.Projection(4, 2, seed=0, device="cpu")
    tp.weight.copy_(torch.tensor(w0.weight.numpy())); tp.bias.copy_(torch.tensor(w0.bias.numpy()))
opt = torch.optim.SGD(tp.parameters(), lr=0.1)
for _ in range(3):
    opt.zero_grad(); torch.nn.functional.cross_entropy(tp(torch.tensor(xv)), torch.tensor([0, 1, 0])).backward(); opt.step()
ok("Projection + SGD 3스텝 = torch", np.allclose(p.weight.numpy(), tp.weight.detach().numpy(), atol=1e-5))

print(f"역전파 엔진 구조: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
