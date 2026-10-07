"""
실험 ⑦: 연합 학습(AssocReadout)에 KC 층이 필요한가? — 실제 냄새 조건부 구별 (XOR형)

  python lab/door_assoc.py
  python lab/door_assoc.py --sets 30 --seeds 1   # 빠르게

과제는 실험 ⑤와 같음: 냄새 4개 묶음마다 AB+, CD+, AC−, BD− (선형 분류기로는 못 품)
특징 × 리드아웃을 묶음마다 따로 학습·평가 (찍기 50%)
  특징:     사구체 / KC 실제 배선 / KC 무작위 배선
  리드아웃: 로지스틱(역전파, 선형), MLP(역전파, 은닉 64), 연합 k (역전파 없음, 클래스당 원형 k개)
원형이 여러 개면(k ≥ 2) 연합 학습 자체가 비선형이라 KC 없이도 풀릴 수 있음 → 그걸 확인
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="DoOR 폴더 (기본: flydnet.data_dir(\"door\"))")
ap.add_argument("--sets", type=int, default=100)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--ks", default="1,2,4,10")
ap.add_argument("--pn-kc-gain", type=float, default=3.0)
ap.add_argument("--n-train", type=int, default=10, help="묶음마다 혼합물 4종 각 n개")
ap.add_argument("--n-test", type=int, default=20)
args = ap.parse_args()
KS = [int(k) for k in args.ks.split(",")]

mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)
X0 = fd.door_odors(enc.glomeruli, args.data)["X"]
ok = np.nonzero((X0 > 0.05).sum(1) >= 3)[0]
mk = lambda c: fd.ConnectomeLayer(c, "PN", "KC", gains={"PN>KC": args.pn_kc_gain}, input_mode="regular")
layers = {"KC 실제": mk(mb), "KC 무작위": mk(mb.shuffled(seed=0))}


def mlp(Ftr, ytr, Fte, yte, hidden=64, steps=300, seed=0):
    """비교용 역전파 MLP (은닉 1층, 자체 엔진): 전체 배치 Adam 300스텝"""
    mu = Ftr.mean(0)
    sd = max(float((Ftr - mu).std()), 1e-6)
    f = lambda X: ((X - mu) / sd).astype(np.float32)
    net = fd.Pathway(fd.Projection(Ftr.shape[1], hidden, seed=seed), fd.Activation("relu"),
                     fd.Projection(hidden, 2, seed=seed + 1))
    rule = fd.Adaptive(net.named_synapses(), rate=1e-2, decay=1e-4)           # 약한 감쇠 (AdamW식, 예전 torch판은 L2)
    x = f(Ftr)
    for _ in range(steps):
        rule.clear(); fd.surprise(net(x), ytr).retrograde(); rule.step()
    with fd.quiescent():
        return float((net(f(Fte)).numpy().argmax(1) == yte).mean())


readouts = {"로지스틱": lambda a, b, c, d: fd.train_linear(a, b, c, d, n_classes=2)["test_acc"], "MLP": mlp}
for k in KS:
    readouts[f"연합 k={k}"] = lambda a, b, c, d, k=k: fd.AssocReadout(a.shape[1], 2, per_class=k).fit(a, b).accuracy(c, d)

feat_names = ["사구체"] + list(layers)
res = {(f, r): [] for f in feat_names for r in readouts}
for s in range(args.seeds):
    t = time.time()
    rng = np.random.default_rng(s)
    sets = [tuple(rng.choice(ok, 4, replace=False)) for _ in range(args.sets)]
    g = np.random.default_rng(10_000 + s)
    (xtr, ytr, ptr) = fd.biconditional_mixtures(X0, sets, args.n_train, rng=g)
    (xte, yte, pte) = fd.biconditional_mixtures(X0, sets, args.n_test, rng=g)
    F = {"사구체": (xtr, xte)} | {k: (fd.extract(L, xtr, enc), fd.extract(L, xte, enc)) for k, L in layers.items()}
    for fname, (Ftr, Fte) in F.items():
        for p in range(args.sets):
            a, b = ptr == p, pte == p
            for rname, fn in readouts.items():
                res[(fname, rname)].append(fn(Ftr[a], ytr[a], Fte[b], yte[b]) * 100)
    print(f"seed {s}: {time.time() - t:.0f}s", flush=True)

n = len(res[(feat_names[0], "로지스틱")])
print(f"\n조건부 구별 정확도 (찍기 50%, 묶음 {n}개 평균 ± 표준오차)")
print(f"{'리드아웃':<12}" + "".join(f"{f:>18}" for f in feat_names))
for r in readouts:
    row = f"{r:<12}"
    for f in feat_names:
        v = np.array(res[(f, r)])
        row += f"{v.mean():>11.1f} ± {v.std(ddof=1) / len(v) ** 0.5:.1f}"
    print(row)
