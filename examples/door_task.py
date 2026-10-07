"""
실험 ④: 실제 냄새 데이터 (DoOR 2.0) — 실제 버섯체 배선이 무작위 배선보다 나은가?

  python examples/door_task.py
  python examples/door_task.py --seeds 2 --shuffles 3   # 빠르게

데이터: data/door/ 에 DoOR.data의 door_response_matrix.csv, door_mappings.csv, odor.csv
        (https://github.com/ropensci/DoOR.data, Münch & Galizia 2016, CC BY-SA 4.0)
        사구체 56개 중 47개가 측정됨, 측정 안 된 사구체는 0

A. 냄새 구별: 냄새 하나 = 클래스. 측정마다 세기가 흔들리는 샘플 (synthetic_odors와 같은 잡음 모델)
B. 화학 계열 일반화: 8개 계열(에스테르, 알코올 등). 학습에 안 쓴 '새 냄새'의 계열 맞히기 (냄새 5겹 교차검증)
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
ap.add_argument("--min-measured", type=int, default=20)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--shuffles", type=int, default=5)
ap.add_argument("--noise", type=float, default=0.5)
ap.add_argument("--add-noise", type=float, default=0.1)
ap.add_argument("--pn-kc-gain", type=float, default=3.0)
ap.add_argument("--min-class-size", type=int, default=7, help="B에서 쓸 계열의 최소 냄새 수")
args = ap.parse_args()

mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)
door = fd.door_odors(enc.glomeruli, args.data, min_measured=args.min_measured)
X0, classes = door["X"], np.asarray(door["classes"])
print(f"{mb}\nDoOR 냄새 {len(X0)}개 | 사구체 {enc.n_glomeruli}개 중 측정 {int(door['measured'].any(0).sum())}개 | "
      f"냄새당 켜진 사구체 {(X0 > 0.05).sum(1).mean():.1f}개\n")

mk = lambda c: fd.ConnectomeLayer(c, "PN", "KC", gains={"PN>KC": args.pn_kc_gain}, input_mode="regular")
layers = {"real": mk(mb)} | {f"shuffled{k}": mk(mb.shuffled(seed=k)) for k in range(args.shuffles)}


def jitter(X, n, seed):
    """냄새마다 n개 샘플: 세기 흔들림 × exp(N(0, noise)) + |N(0, add_noise)|"""
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(len(X)), n)
    x = X[y] * np.exp(args.noise * g.standard_normal((len(y), X.shape[1])))
    return (x + np.abs(args.add_noise * g.standard_normal((len(y), X.shape[1])))).astype(np.float32), y


def features(x):
    return {"glomeruli": x} | {name: fd.extract(L, x, enc) for name, L in layers.items()}


acc = lambda Ftr, ytr, Fte, yte: fd.train_linear(Ftr, ytr, Fte, yte)["test_acc"] * 100


def report(title, res):
    glo, real = np.array(res["glomeruli"]), np.array(res["real"])
    shuf = np.array([res[f"shuffled{k}"] for k in range(args.shuffles)]).T   # (반복, 무작위)
    diff = real - shuf.mean(1)
    print(f"{title}\n  glomeruli {glo.mean():5.1f} | KC real {real.mean():5.1f} | KC shuffled {shuf.mean():5.1f} "
          f"(무작위끼리 범위 {shuf.mean(0).min():.1f}~{shuf.mean(0).max():.1f})\n"
          f"  실제 - 무작위: {diff.mean():+.2f} ± {diff.std(ddof=1) / len(diff) ** 0.5:.2f} (표준오차, {len(diff)}회) | "
          f"양수 {int((diff > 0).sum())}/{len(diff)} | 실제가 이긴 비율 {(real[:, None] > shuf).mean() * 100:.0f}%",
          flush=True)


# ───────────── A. 냄새 구별 ─────────────
t = time.time()
res = {k: [] for k in ["glomeruli"] + list(layers)}
for s in range(args.seeds):
    (xtr, ytr), (xte, yte) = jitter(X0, 10, seed=2 * s), jitter(X0, 20, seed=2 * s + 1)
    Ftr, Fte = features(xtr), features(xte)
    for k in res:
        res[k].append(acc(Ftr[k], ytr, Fte[k], yte))
report(f"A. 냄새 구별: {len(X0)}개 냄새, 학습 10 / 평가 20 샘플씩, seed {args.seeds}개 ({time.time() - t:.0f}s)", res)

# ───────────── B. 화학 계열 일반화 ─────────────
t = time.time()
names, counts = np.unique(classes, return_counts=True)
use = [c for c, n in zip(names, counts) if n >= args.min_class_size and c != "other"]
idx = np.nonzero(np.isin(classes, use))[0]
lab = np.array([use.index(c) for c in classes[idx]])
print(f"\nB. 화학 계열: " + ", ".join(f"{c} {int((classes == c).sum())}" for c in use)
      + f" (냄새 {len(idx)}개, 가장 큰 계열만 찍으면 {np.bincount(lab).max() / len(lab) * 100:.0f}%)")
res = {k: [] for k in ["glomeruli"] + list(layers)}
for s in range(args.seeds):
    x, which = jitter(X0[idx], 20, seed=100 + s)                 # which = 몇 번째 냄새의 샘플인지
    F = features(x)
    y = lab[which]
    rng = np.random.default_rng(s)
    fold = np.empty(len(idx), int)                               # 계열별로 고르게 5겹 나누기
    for c in range(len(use)):
        members = rng.permutation(np.nonzero(lab == c)[0])
        fold[members] = np.arange(len(members)) % 5
    for f in range(5):
        te = fold[which] == f
        for k in res:
            res[k].append(acc(F[k][~te], y[~te], F[k][te], y[te]))
report(f"  학습에 안 쓴 냄새의 계열 정확도 (5겹 × seed {args.seeds}개, {time.time() - t:.0f}s)", res)
