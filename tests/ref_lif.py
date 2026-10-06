"""내장 LIF(합친 CUDA 커널·원소별 경로) 대 플러그인 fd.neurons.LIF(일반 Signal 연산 + 자동 미분): 같은 수식이라
출력 스파이크와 역전파 기울기(연결 배율·입력)가 같아야 함. 내장판의 손으로 쓴 순전파·역전파를 독립 구현으로 확인

  python tests/ref_lif.py [조합 수]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flydnet as fd
from flydnet.ganglion import backend as B
from test_threefactor import _rec

G = fd.genetics
c = _rec(feedback_edges=True, strong=True)
H = c.groups["H"]
bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


def run(neuron, dev, cfg, X, f64=False):
    kw = dict(t_ms=30, device=dev, trainable=True, input_mode=cfg["mode"], dt=cfg["dt"])
    kw.update(cfg["kw"])
    L = fd.Connectome(c, "IN", "O", neuron=neuron, **kw)
    if f64:                                                   # float64로 (원소별 경로): 반올림 차이 없이 수식만 비교
        L.w_syn = L.w_syn.astype(np.float64)
        L._build()
        L.log_scale.data = L.log_scale.data.astype(np.float64)
        X = X.astype(np.float64)
    eff = []
    if "silence" in cfg["eff"]:
        eff.append(G.silence(L, G.Line(c, H[3:9], "s")))
    if "block" in cfg["eff"]:
        eff.append(G.block(L, G.Line(c, H[12:], "b")))
    if "activate" in cfg["eff"]:
        eff.append(G.activate(L, G.Line(c, H[:5], "a"), hz=120))
    try:
        x = fd.Signal(X, device=dev, plastic=True)
        y = L(x, seed=5, return_all=True)
        w = np.linspace(-1, 1, y.shape[1], dtype=np.float32)[None] * np.ones((y.shape[0], 1), np.float32)
        (y * fd.Signal(w, device=dev)).sum().retrograde()
        return B.numpy(y.data), B.numpy(L.log_scale.retro), B.numpy(x.retro)
    finally:
        for e in eff:
            e.remove()


rng = np.random.default_rng(0)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])
for it in range(n):
    kw = {}
    if rng.random() < 0.3:
        kw["count_from_ms"] = 6.0
    if rng.random() < 0.3:
        kw["ckpt"] = 40
    if rng.random() < 0.3:
        kw["damp"] = float(rng.choice([0.3, 1.0]))
    if rng.random() < 0.25:
        kw["truncate"] = 60
    cfg = dict(kw=kw, mode=str(rng.choice(["poisson", "regular"])), dt=float(rng.choice([0.1, 0.25])),
               eff=[e for e in ("silence", "block", "activate") if rng.random() < 0.3])
    X = rng.uniform(40, 220, (3, 6)).astype(np.float32)
    if rng.random() < 0.3:
        X = rng.uniform(40, 220, (3, 4, 6)).astype(np.float32)                 # 시간에 따라 바뀌는 입력
    for dev in devs:
        a = run("lif", dev, cfg, X)
        b = run(fd.neurons.LIF(), dev, cfg, X)
        names = ("출력", "연결 기울기", "입력 기울기")
        for k, (u, v) in enumerate(zip(a, b)):
            scale = max(1.0, float(np.abs(u).max()))
            # float32: 300스텝 역전파의 덧셈 순서 차이로 상대 1e-4까지 (float64로는 1e-9 - 아래에서 따로 확인)
            same = np.array_equal(u, v) if k == 0 else np.allclose(u, v, rtol=1e-3, atol=1e-4 * scale)
            ok(f"[{dev}] {names[k]} {cfg}", same, f"최대 차 {float(np.abs(u - v).max()):.3g} / 크기 {scale:.3g}")
# float64 (CPU 원소별 경로): 수식이 같으면 반올림 없이 거의 같아야 함
for it in range(8):
    cfg = dict(kw={}, mode="regular" if it % 2 else "poisson", dt=0.1, eff=[["silence"], ["block"], ["activate"], []][it % 4])
    X = rng.uniform(40, 220, (3, 6))
    a, b = run("lif", "cpu", cfg, X, f64=True), run(fd.neurons.LIF(), "cpu", cfg, X, f64=True)
    for k, (u, v) in enumerate(zip(a, b)):
        scale = max(1.0, float(np.abs(u).max()))
        ok(f"[float64] {('출력', '연결 기울기', '입력 기울기')[k]} {cfg}", np.allclose(u, v, rtol=1e-7, atol=1e-8 * scale),
           f"최대 차 {float(np.abs(u - v).max()):.3g}")
# legacy LIF·graded 뉴런 대 torch 기본 연산으로 다시 짠 참조 구현 (tests/_ref_models.py, float64 autograd)
# - 0.1.17까지는 torch판 ConnectomeLayer가 기준이었음
try:
    import torch
    from _ref_models import torch_graded, torch_legacy_lif
except ImportError:
    torch = None
if torch is not None:
    Xl = rng.uniform(40, 220, (3, 6)).astype(np.float32)
    for kind, kw in (("legacy", dict(timing="legacy", input_mode="regular", t_ms=30, dt=0.1)),
                     ("legacy+뉴런 매개변수", dict(timing="legacy", input_mode="regular", t_ms=30, dt=0.25,
                                               bias={"H": 3.0}, train_neurons=True, v_init="random",
                                               count_from_ms=5, share="pair")),
                     ("graded", dict(neuron="graded", t_ms=8)),
                     ("graded+bias", dict(neuron="graded", t_ms=8, bias={"H": 0.3}, train_neurons=True))):
        a = fd.Connectome(c, "IN", "O", device="cpu", trainable=True, **kw)
        xin = Xl if kind.startswith("legacy") else Xl / 220
        x = fd.Signal(xin, plastic=True)
        ya = a(x, seed=1, return_all=False)
        w = np.linspace(-1, 1, ya.size, dtype=np.float32).reshape(ya.shape)
        (ya * fd.Signal(w)).sum().retrograde()
        xt = torch.tensor(xin, dtype=torch.float64, requires_grad=True)
        yb, P = (torch_legacy_lif if kind.startswith("legacy") else torch_graded)(a, xt)
        (yb * torch.tensor(w, dtype=torch.float64)).sum().backward()
        pairs = [("출력", ya.numpy(), yb.detach().numpy()), ("입력 기울기", np.asarray(x.retro), xt.grad.numpy())]
        pairs += [(k, np.asarray(getattr(a, k).retro), t.grad.numpy()) for k, t in P.items()
                  if isinstance(getattr(a, k), fd.Synapse)]
        for nm, u, v in pairs:
            ok(f"[참조 구현] {kind} {nm}", np.allclose(u, v, rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(v).max()))),
               f"최대 차 {float(np.abs(u - v).max()):.3g}")
print(f"내장 LIF 대 플러그인 LIF: 조합 {n}개 x 장치 {len(devs)}, 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
