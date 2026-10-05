"""옵션 조합 무작위 시험: 합친 커널 = 원소별 (GPU, 비트 단위), GPU ≈ CPU. 출력·연결 기울기·입력 기울기
  python tests/combo_check.py [seed] [개수]"""
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flydnet as fd
from flydnet.ganglion import backend as B, kernels as K, physiology as P
from test_threefactor import _rec

warnings.simplefilter("ignore")
G = fd.genetics
c = _rec(feedback_edges=True, strong=True)
H = c.groups["H"]
X = np.random.default_rng(0).uniform(50, 200, (4, 6)).astype(np.float32)
ORIG = (K._fused_ok, K.poisson_spikes)


def elementwise(on):
    if on:
        K._fused_ok = lambda *a, **k: False
        K.poisson_spikes = lambda xp, seed, step, p: (P.hash_uniform(xp, seed, step, p.shape[::-1]).T < p).astype(p.dtype)
    else:
        K._fused_ok, K.poisson_spikes = ORIG


def run(dev, cfg):
    kw = dict(t_ms=40, device=dev, trainable=True)
    kw.update(cfg["kw"])
    L = fd.Connectome(c, "IN", "O", **kw)
    effs = []
    for e in cfg["eff"]:
        if e == "activate":
            effs.append(G.activate(L, G.Line(c, H[:6], "a"), hz=150))
        elif e == "activate2":
            effs.append(G.activate(L, G.Line(c, H[6:12], "b"), hz=90))
        elif e == "block":
            effs.append(G.block(L, G.Line(c, H[12:], "k")))
        elif e == "silence":
            effs.append(G.silence(L, G.Line(c, H[3:9], "s")))
        elif e == "mosaic":
            effs.append(G.mosaic(L, p=0.3))
    try:
        if cfg["tf"]:
            t = fd.ThreeFactor(L, feedback="random", seed=0)
            o = t(X, seed=3)
            (o * o).sum().retrograde()
            return [B.numpy(o.data), B.numpy(t.assign(o))]
        x = fd.Signal(X, device=dev, plastic=True)
        y = L(x, seed=3)
        (y * y).sum().retrograde()
        return [B.numpy(y.data), B.numpy(L.log_scale.retro), B.numpy(x.retro)]
    finally:
        for e in effs:
            e.remove()


rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
bad = 0
for it in range(n):
    kw = {}
    if rng.random() < 0.4:
        kw["ckpt"] = int(rng.choice([25, 60]))
    if rng.random() < 0.3:
        kw["truncate"] = int(rng.choice([40, 90]))
    if rng.random() < 0.3:
        kw["noise"] = 0.5
    if rng.random() < 0.3:
        kw["input_mode"] = "regular"
    if rng.random() < 0.3:
        kw["damp"] = float(rng.choice([0.3, 1.0]))
    if rng.random() < 0.3:
        kw["count_from_ms"] = 8.0
    if rng.random() < 0.2:
        kw["share"] = "pair"
    eff = [e for e in ("activate", "activate2", "block", "silence", "mosaic") if rng.random() < 0.3]
    tf = rng.random() < 0.2
    cfg = dict(kw=kw, eff=eff, tf=tf)
    try:
        elementwise(False); fast = run("gpu", cfg)
        elementwise(True); slow = run("gpu", cfg)
        elementwise(False); cpu = run("cpu", cfg)
    except Exception as e:                                                     # noqa: BLE001
        bad += 1
        print(f"✗ {cfg}: {type(e).__name__}: {e}")
        continue
    pair = kw.get("share") == "pair"                               # 원자적 덧셈 순서로 GPU 안에서도 반올림 차이
    same = all(np.array_equal(a, b) if not (pair and i == 1) else np.allclose(a, b, rtol=1e-5, atol=1e-6 * max(1, np.abs(a).max()))
               for i, (a, b) in enumerate(zip(fast, slow)))
    close = all(np.allclose(a, b, rtol=2e-3, atol=2e-3 * max(1.0, float(np.abs(a).max()))) for a, b in zip(fast, cpu))
    if not (same and close):
        bad += 1
        print(f"✗ {cfg}: 합친=원소별 {same}, GPU≈CPU {close}",
              [float(np.abs(a - b).max()) for a, b in zip(fast, slow)], [float(np.abs(a - b).max()) for a, b in zip(fast, cpu)])
print(f"조합 {n}개 중 문제 {bad}개")
