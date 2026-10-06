"""Signal 연산 대 torch autograd (float64): 값과 기울기(벡터-야코비안 곱)를 비교. 꺾이는 점(relu·abs의 0, max의 동점,
clip의 경계)을 일부러 넣음 - 수치 미분(fuzz_ops.py)은 이 점을 피해야 해서 따로 확인

  python tests/ref_torch_ops.py [반복 수]        # torch 필요 (pip install "flydnet[torch]")
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import flydnet as fd
from flydnet.ganglion import physiology as P
from flydnet.ganglion.signal import concat, where

torch.set_default_dtype(torch.float64)


def _shape(r, nd=None):
    nd = nd if nd is not None else int(r.integers(1, 4))
    return tuple(int(r.integers(1, 5)) for _ in range(nd))


def _kinky(r, shape, at=0.0, ties=False):
    """꺾이는 점·동점이 섞인 값 (정확히 at, 또는 같은 값 여러 개)"""
    x = r.standard_normal(shape)
    x[r.random(shape) < 0.25] = at
    if ties:
        x = np.round(x * 2) / 2                                   # 0.5 단위 → 동점이 흔함
    return x


def _axis(r, nd):
    c = r.random()
    if c < 0.25 or nd == 0:
        return None
    if c < 0.5 and nd > 1:
        return tuple(sorted(r.choice(nd, size=int(r.integers(1, nd + 1)), replace=False).tolist()))
    a = int(r.integers(0, nd))
    return a - nd if r.random() < 0.3 else a


def _t(x, grad=True):
    return torch.tensor(np.asarray(x, np.float64), requires_grad=grad)


def _tmax(t, axis, keepdims):
    """torch: 동점이면 기울기를 고르게 나누는 amax (flydnet과 같은 규약)"""
    if axis is None:
        return t.amax() if not keepdims else t.amax(dim=tuple(range(t.ndim)), keepdim=True)
    return t.amax(dim=axis, keepdim=keepdims)


def cases():
    C = []

    def add(name, make, f_fd, f_t):
        C.append((name, make, f_fd, f_t))

    # 원소별 (꺾이는 점 포함)
    add("relu@0", lambda r: [_kinky(r, _shape(r))], lambda a: a[0].relu(), lambda a: torch.relu(a[0]))
    add("abs@0", lambda r: [_kinky(r, _shape(r))], lambda a: a[0].abs(), lambda a: a[0].abs())
    add("exp", lambda r: [r.standard_normal(_shape(r))], lambda a: a[0].exp(), lambda a: a[0].exp())
    add("log", lambda r: [np.abs(r.standard_normal(_shape(r))) + 0.1], lambda a: a[0].log(), lambda a: a[0].log())
    add("tanh", lambda r: [r.standard_normal(_shape(r)) * 3], lambda a: a[0].tanh(), lambda a: a[0].tanh())
    add("sigmoid±큼", lambda r: [r.standard_normal(_shape(r)) * 30], lambda a: a[0].sigmoid(), lambda a: a[0].sigmoid())
    add("sqrt", lambda r: [np.abs(r.standard_normal(_shape(r))) + 0.1], lambda a: a[0].sqrt(), lambda a: a[0].sqrt())
    add("square", lambda r: [r.standard_normal(_shape(r))], lambda a: a[0].square(), lambda a: a[0].square())
    add("neg", lambda r: [r.standard_normal(_shape(r))], lambda a: -a[0], lambda a: -a[0])
    add("pow3", lambda r: [r.standard_normal(_shape(r))], lambda a: a[0] ** 3, lambda a: a[0] ** 3)
    add("pow-1.5", lambda r: [np.abs(r.standard_normal(_shape(r))) + 0.2], lambda a: a[0] ** -1.5, lambda a: a[0] ** -1.5)
    add("pow0@0", lambda r: [_kinky(r, _shape(r))], lambda a: a[0] ** 0, lambda a: a[0] ** 0)
    add("rpow", lambda r: [r.standard_normal(_shape(r))], lambda a: 2.5 ** a[0], lambda a: 2.5 ** a[0])
    # clip: 경계값에서 기울기 1 (경계 포함, flydnet 규약). torch 2.x clamp는 경계에서 0 - 둘 다 맞는 부분 기울기라 참조를 맞춤
    add("clip@경계", lambda r: [_kinky(r, _shape(r), at=0.5)], lambda a: a[0].clip(-0.5, 0.5),
        lambda a: _clip_incl(a[0], -0.5, 0.5))
    add("clip lo만", lambda r: [_kinky(r, _shape(r), at=-0.2)], lambda a: a[0].clip(lo=-0.2),
        lambda a: _clip_incl(a[0], -0.2, None))

    # 이항 (브로드캐스팅)
    def bin_make(r):
        s = _shape(r)
        b = tuple(1 if r.random() < 0.4 else d for d in s)
        b = b[int(r.integers(0, len(b))):] if r.random() < 0.3 else b
        x, y = r.standard_normal(s), r.standard_normal(b)
        return [x, y] if r.random() < 0.5 else [y, x]
    add("add", bin_make, lambda a: a[0] + a[1], lambda a: a[0] + a[1])
    add("sub", bin_make, lambda a: a[0] - a[1], lambda a: a[0] - a[1])
    add("mul", bin_make, lambda a: a[0] * a[1], lambda a: a[0] * a[1])
    add("div", lambda r: [v if i == 0 else np.sign(v) * (np.abs(v) + 0.3) for i, v in enumerate(bin_make(r))],
        lambda a: a[0] / a[1], lambda a: a[0] / a[1])
    add("rsub 숫자", lambda r: [r.standard_normal(_shape(r))], lambda a: 1.5 - a[0], lambda a: 1.5 - a[0])
    add("rdiv 숫자", lambda r: [np.abs(r.standard_normal(_shape(r))) + 0.3], lambda a: 2.0 / a[0], lambda a: 2.0 / a[0])

    def where_make(r):
        s = _shape(r)
        return [r.standard_normal(s), r.standard_normal(s), (r.random(s) < 0.5).astype(np.float64)]
    add("where", where_make, lambda a: where(a[2].data > 0.5, a[0], a[1]),
        lambda a: torch.where(a[2] > 0.5, a[0], a[1]))
    add("where 숫자", where_make, lambda a: where(a[2].data > 0.5, a[0], 0.0),
        lambda a: torch.where(a[2] > 0.5, a[0], torch.zeros(())))

    # 모으기 (동점 포함)
    for kd in (False, True):
        def red_make(r, ties=False):
            return [_kinky(r, _shape(r), ties=ties)]
        add(f"sum kd={kd}", red_make, lambda a, kd=kd: a[0].sum(axis=a.axis, keepdims=kd),
            lambda a, kd=kd: a[0].sum(dim=a.axis, keepdim=kd) if a.axis is not None else
            (a[0].sum() if not kd else a[0].sum(dim=tuple(range(a[0].ndim)), keepdim=True)))
        add(f"mean kd={kd}", red_make, lambda a, kd=kd: a[0].mean(axis=a.axis, keepdims=kd),
            lambda a, kd=kd: a[0].mean(dim=a.axis, keepdim=kd) if a.axis is not None else
            (a[0].mean() if not kd else a[0].mean(dim=tuple(range(a[0].ndim)), keepdim=True)))
        add(f"max 동점 kd={kd}", lambda r: red_make(r, True), lambda a, kd=kd: a[0].max(axis=a.axis, keepdims=kd),
            lambda a, kd=kd: _tmax(a[0], a.axis, kd))
        add(f"min 동점 kd={kd}", lambda r: red_make(r, True), lambda a, kd=kd: a[0].min(axis=a.axis, keepdims=kd),
            lambda a, kd=kd: -_tmax(-a[0], a.axis, kd))
        add(f"var kd={kd}", red_make, lambda a, kd=kd: a[0].var(axis=a.axis, keepdims=kd),
            lambda a, kd=kd: a[0].var(dim=a.axis, keepdim=kd, unbiased=False) if a.axis is not None else
            (a[0].var(unbiased=False) if not kd else a[0].var(dim=tuple(range(a[0].ndim)), keepdim=True, unbiased=False)))

    # 모양
    add("reshape", lambda r: [r.standard_normal((2, 3, 4))], lambda a: a[0].reshape(4, 6), lambda a: a[0].reshape(4, 6))
    add("transpose 음수축", lambda r: [r.standard_normal((2, 3, 4))], lambda a: a[0].transpose(-1, 0, 1),
        lambda a: a[0].permute(2, 0, 1))
    add("T", lambda r: [r.standard_normal((3, 5))], lambda a: a[0].T, lambda a: a[0].T)
    add("flatten", lambda r: [r.standard_normal((2, 3, 4))], lambda a: a[0].flatten(1), lambda a: a[0].flatten(1))
    add("squeeze", lambda r: [r.standard_normal((1, 3, 1))], lambda a: a[0].squeeze(), lambda a: a[0].squeeze())
    add("getitem 슬라이스", lambda r: [r.standard_normal((5, 4))], lambda a: a[0][1:4, ::2], lambda a: a[0][1:4, ::2])
    add("getitem 중복 번호", lambda r: [r.standard_normal((5, 4))], lambda a: a[0][np.array([0, 2, 2, 4, 0])],
        lambda a: a[0][torch.tensor([0, 2, 2, 4, 0])])
    add("getitem 불리언", lambda r: [r.standard_normal((5, 4))], lambda a: a[0][a[0].data > 0],
        lambda a: a[0][a[0] > 0])
    add("getitem 두 축 번호", lambda r: [r.standard_normal((5, 4))],
        lambda a: a[0][np.array([0, 1, 1]), np.array([3, 2, 2])], lambda a: a[0][torch.tensor([0, 1, 1]), torch.tensor([3, 2, 2])])
    add("concat", lambda r: [r.standard_normal((2, 3)), r.standard_normal((4, 3))], lambda a: concat([a[0], a[1]]),
        lambda a: torch.cat([a[0], a[1]]))
    add("concat 축1", lambda r: [r.standard_normal((2, 3)), r.standard_normal((2, 1))],
        lambda a: concat([a[0], a[1]], axis=1), lambda a: torch.cat([a[0], a[1]], 1))

    # 행렬 곱 (1차원 포함)
    for sa, sb in (((3, 4), (4, 2)), ((4,), (4, 2)), ((3, 4), (4,)), ((4,), (4,))):
        add(f"matmul {sa}@{sb}", lambda r, sa=sa, sb=sb: [r.standard_normal(sa), r.standard_normal(sb)],
            lambda a: a[0] @ a[1], lambda a: a[0] @ a[1])

    # 생리 함수
    add("log_softmax", lambda r: [r.standard_normal((4, 5)) * 5], lambda a: P.log_softmax(a[0]),
        lambda a: torch.log_softmax(a[0], -1))
    add("softmax", lambda r: [r.standard_normal((4, 5)) * 5], lambda a: a[0].softmax(), lambda a: torch.softmax(a[0], -1))

    def ce_make(r):
        return [r.standard_normal((6, 4)) * 3]
    yv = np.array([0, 3, 1, 1, 2, 0])
    add("surprise", ce_make, lambda a: P.surprise(a[0], yv),
        lambda a: torch.nn.functional.cross_entropy(a[0], torch.tensor(yv)))
    add("inhibit k 동점", lambda r: [_kinky(r, (3, 8), ties=True)], lambda a: P.inhibit(a[0], k=3),
        lambda a: a[0] * (a[0] >= torch.topk(a[0], 3, -1).values[..., -1:]).double())
    add("fire 대리 기울기", lambda r: [r.standard_normal((3, 5))], lambda a: P.fire(a[0], 0.2, 4.0),
        lambda a: _SpikeRef.apply(a[0] - 0.2, 4.0))
    add("std (0 아님)", lambda r: [r.standard_normal((3, 5))], lambda a: a[0].std(axis=1),
        lambda a: a[0].std(dim=1, unbiased=False))
    add("homeostasis", lambda r: [r.standard_normal((3, 6)) * 4 + 2], lambda a: fd.Homeostasis(eps=1e-5)(a[0]),
        lambda a: torch.nn.functional.layer_norm(a[0], (6,), eps=1e-5))
    return C


def _clip_incl(x, lo, hi):
    """경계를 포함해 기울기를 통과시키는 clamp"""
    inside = torch.ones_like(x, dtype=torch.bool)
    if lo is not None:
        inside &= x >= lo
    if hi is not None:
        inside &= x <= hi
    return torch.where(inside, x, x.clamp(lo, hi).detach())


class _SpikeRef(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x); ctx.slope = slope
        return (x > 0).double()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + ctx.slope * x.abs()) ** 2, None


class _Args(list):
    axis = None


def run(n=30, seed=0, device="cpu"):
    r = np.random.default_rng(seed)
    bad = []
    for name, make, f_fd, f_t in cases():
        for it in range(n):
            xs = make(r)
            a_fd = _Args(fd.Signal(x, plastic=True, device=device) for x in xs)
            a_t = _Args(_t(x) for x in xs)
            a_fd.axis = a_t.axis = _axis(r, xs[0].ndim)
            try:
                y_t = f_t(a_t)
            except (RuntimeError, IndexError, TypeError):
                continue                                             # torch가 못 하는 축 조합은 건너뜀
            try:
                y = f_fd(a_fd)
            except Exception as e:                                   # noqa: BLE001
                bad.append(f"{name}: flydnet 오류 {type(e).__name__}: {e}")
                break
            vy = np.asarray(y.numpy(), np.float64)
            if vy.shape != tuple(y_t.shape) or not np.allclose(vy, y_t.detach().numpy(), rtol=1e-10, atol=1e-12):
                bad.append(f"{name}: 값 다름 {vy.shape} vs {tuple(y_t.shape)} (축 {a_fd.axis})")
                break
            g = r.standard_normal(vy.shape)
            if y.plastic:
                y.retrograde(g if vy.shape else np.asarray(g))
            (y_t * torch.tensor(g)).sum().backward()
            for i, (s, t) in enumerate(zip(a_fd, a_t)):
                gt = t.grad.numpy() if t.grad is not None else np.zeros(t.shape)
                gf = np.zeros(t.shape) if s.retro is None else np.asarray(fd.ganglion.backend.numpy(s.retro), np.float64)
                if gf.shape != gt.shape or not np.allclose(gf, gt, rtol=1e-9, atol=1e-11):
                    err = float(np.abs(gf - gt).max()) if gf.shape == gt.shape else "모양"
                    bad.append(f"{name}: 입력 {i} 기울기 다름 (최대 차 {err}, 축 {a_fd.axis})")
                    break
            else:
                continue
            break
    return bad


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    total = 0
    for dev in ["cpu"] + (["gpu"] if fd.ganglion.backend.gpu_available() else []):
        bad = run(n, device=dev)
        total += len(bad)
        for b in bad:
            print(f"  ✗ [{dev}] {b}")
        print(f"[{dev}] 연산 {len(cases())}종 x {n}번: 문제 {len(bad)}개")
    sys.exit(1 if total else 0)
