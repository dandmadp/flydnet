"""학습 규칙 대 기준: ThreeFactor(feedback="none")의 출력 쪽 연결 기울기 = 역전파 (설정·효과기를 바꿔 가며),
STDP 변화량 = 스파이크 기록으로 직접 센 쌍 기반 STDP, 도파민 리드아웃 = 문서의 수식

  python tests/ref_rules.py [조합 수]
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
bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


c = _rec(strong=True)                                                    # 출력에서 되돌아오는 경로 없음
H = c.groups["H"]
rng = np.random.default_rng(0)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])

# 1) ThreeFactor 출력 쪽 = 역전파
compared = skipped = 0
for it in range(n):
    kw = dict(t_ms=60, trainable=True, input_mode=str(rng.choice(["poisson", "regular"])), dt=float(rng.choice([0.1, 0.2])))
    if rng.random() < 0.3:
        kw["count_from_ms"] = 10.0
    if rng.random() < 0.3:
        kw["damp"] = float(rng.choice([0.3, 1.0]))
    eff = [e for e in ("silence", "block", "activate", "mosaic") if rng.random() < 0.3]
    for dev in devs:
        L = fd.Connectome(c, "IN", "O", device=dev, **kw)
        ex = []
        if "silence" in eff:
            ex.append(G.silence(L, G.Line(c, H[2:6], "s")))
        if "block" in eff:
            ex.append(G.block(L, G.Line(c, H[10:14], "b")))
        if "activate" in eff:
            ex.append(G.activate(L, G.Line(c, H[15:18], "a"), hz=100))
        if "mosaic" in eff:
            ex.append(G.mosaic(L, p=0.3, by="group"))
        x = rng.uniform(50, 200, (6, 6)).astype(np.float32)
        tgt = rng.uniform(0, 1, (6, 5)).astype(np.float32)
        out = L(x, seed=3)
        if float(out.data.mean()) < 1:
            skipped += 1
            for e in ex:
                e.remove()
            continue
        compared += 1
        ((out * 0.02 - tgt) ** 2).sum().retrograde()
        bptt = B.numpy(L.log_scale.retro).copy()
        L.log_scale.retro = None
        tf = fd.ThreeFactor(L, feedback="none")
        o = tf(x, seed=3)
        ok(f"[{dev}] ThreeFactor 순전파 = 층 순전파 {kw} {eff}", np.array_equal(B.numpy(o.data), B.numpy(out.data)))
        ((o * 0.02 - tgt) ** 2).sum().retrograde()
        tf.assign(o)
        tf3 = B.numpy(L.log_scale.retro)
        onto = np.isin(B.numpy(L.wiring.post), B.numpy(L.out_idx))[B.numpy(L.train_pos)]
        s = np.abs(bptt[onto]).max() + 1e-12
        ok(f"[{dev}] ThreeFactor = 역전파 (출력 쪽) {kw} {eff}",
           np.allclose(tf3[onto], bptt[onto], rtol=2e-3, atol=2e-5 * s), f"최대 차 {np.abs(tf3[onto] - bptt[onto]).max() / s:.3g} (상대)")
        ok(f"[{dev}] feedback=none이면 숨은 연결 0", np.abs(tf3[~onto]).max() == 0)
        for e in ex:
            e.remove()

ok("ThreeFactor: 대부분의 조합을 실제로 비교", compared >= skipped * 3, f"비교 {compared}, 건너뜀 {skipped}")
print(f"ThreeFactor 비교 {compared}개, 출력이 약해 건너뜀 {skipped}개")

# 2) STDP = 기록한 스파이크로 직접 센 쌍 기반 STDP (지연 무시, 보낸 시각 기준)
for dev in devs:
    L = fd.Connectome(c, "IN", "O", t_ms=40, dt=0.1, trainable=True, input_mode="regular", device=dev)
    blk = G.block(L, G.Line(c, H[:4], "b"))                              # 막힌 뉴런: 시냅스 전은 0, 시냅스 후는 실제 발화
    x = rng.uniform(80, 200, (2, 6)).astype(np.float32)
    st = fd.STDP(L, a_plus=0.01, a_minus=0.012, tau_plus=15.0, tau_minus=25.0)
    st(x, seed=2, batch=2)
    d = B.numpy(st.delta)
    with fd.quiescent():
        _, rec = L(x, seed=2, record=np.arange(c.N))                      # (B, steps, N) 실제 발화
    blk.remove()
    fired = rec
    sent = fired.copy()
    sent[:, :, H[:4]] = 0
    pre, post = B.numpy(L.wiring.pre)[B.numpy(L.train_pos)], B.numpy(L.wiring.post)[B.numpy(L.train_pos)]
    dp, dm = np.exp(-0.1 / 15.0), np.exp(-0.1 / 25.0)
    xtr = np.zeros((2, c.N)); ytr = np.zeros((2, c.N)); ref = np.zeros(len(pre))
    for s in range(fired.shape[1]):
        xtr *= dp; ytr *= dm
        S, F = sent[:, s], fired[:, s]
        ref += (0.01 * (xtr[:, pre] * F[:, post]).sum(0) - 0.012 * (ytr[:, post] * S[:, pre]).sum(0)) / 2
        xtr += S; ytr += F
    ok(f"[{dev}] STDP = 직접 센 쌍 기반", np.allclose(d, ref, rtol=1e-4, atol=1e-7), f"최대 차 {np.abs(d - ref).max():.3g}")
    ok(f"[{dev}] STDP 변화가 0이 아님", np.abs(ref).max() > 0)

# 3) 도파민 리드아웃 = 문서의 수식 (한 묶음)
X = rng.random((8, 10)).astype(np.float32)
y = rng.integers(0, 3, 8)
a = X / X.max(1, keepdims=True)
oh = np.eye(3)[y]
for mode in ("bidir", "ltd", "ltd_err", "ltp", "assoc"):
    D = fd.DopamineReadout(10, 3, mode=mode, lr=0.1, device="cpu")
    W0 = np.asarray(D.W).copy()
    s0 = a @ W0.T if mode != "assoc" else None
    D.step(X, y)
    if mode == "bidir":
        win = s0.argmax(1); wrong = (win != y)[:, None]
        da = wrong * (oh - np.eye(3)[win]); ref = np.maximum(W0 + 0.1 * da.T @ a / 8, 0)
    elif mode == "ltp":
        ref = W0 + 0.1 * (oh.T @ a / 8) * (1 - W0)
    elif mode == "ltd":
        ref = W0 * np.maximum(1 - 0.1 * ((1 - oh).T @ a / 8), 0)
    elif mode == "ltd_err":
        da = ((s0 >= s0[np.arange(8), y][:, None]) & (oh == 0)).astype(float)
        ref = W0 * np.maximum(1 - 0.1 * (da.T @ a / 8), 0)
    else:                                                                 # assoc: 클래스마다 받은 시료 평균
        ref = np.stack([a[y == k].mean(0) if (y == k).any() else np.zeros(10) for k in range(3)])
    ok(f"DopamineReadout {mode} = 수식", np.allclose(np.asarray(D.W), ref, atol=1e-6), f"{np.abs(np.asarray(D.W) - ref).max():.3g}")

print(f"학습 규칙: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
