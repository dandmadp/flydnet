"""
실험 ⑤: 실제 냄새 혼합물 — 조건부 구별 (biconditional discrimination, XOR형)

  python examples/door_mixtures.py
  python examples/door_mixtures.py --sets 30 --shuffles 3   # 빠르게

냄새 4개 (A, B, C, D)마다: AB → 보상(1), CD → 보상(1), AC → 보상 없음(0), BD → 보상 없음(0)
모든 냄새가 보상·무보상에 똑같이 한 번씩 들어가고 모든 샘플이 2개 혼합이라, 반응이 더해지면
어떤 선형 분류기로도 못 품 (위 두 식의 합 > 2t, 아래 두 식의 합 < 2t 인데 둘은 같은 값).
초파리의 형태(configural) 학습 과제 (Young et al. 2011).
냄새 묶음마다 따로 학습·평가하고 평균 (찍기 50%)
(A+, B+, AB−의 부정 패턴은 '켜진 사구체가 많으면 무보상'이라는 선형 규칙으로 풀려서 쓰지 않음)

혼합물 사구체 반응 = 포화(A + B), 포화(x) = x / (x + c)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="DoOR 폴더 (기본: flydnet.data_dir(\"door\"))")
ap.add_argument("--sets", type=int, default=100, help="냄새 4개 묶음 수")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--shuffles", type=int, default=5)
ap.add_argument("--noise", type=float, default=0.5)
ap.add_argument("--add-noise", type=float, default=0.05)
ap.add_argument("--sat", type=float, default=0.3, help="포화 상수 c")
ap.add_argument("--pn-kc-gain", type=float, default=3.0, help="실제 냄새에서 KC 약 6% 활성")
ap.add_argument("--n-train", type=int, default=10, help="묶음마다 혼합물 4종 각 n개")
ap.add_argument("--n-test", type=int, default=20)
args = ap.parse_args()

mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)
X0 = fd.door_odors(enc.glomeruli, args.data)["X"]
ok = np.nonzero(((X0 > 0.05).sum(1) >= 3).numpy())[0]             # 사구체 3개 이상 켜는 냄새만
mk = lambda c: fd.torch.ConnectomeLayer(c, "PN", "KC", gains={"PN>KC": args.pn_kc_gain}, input_mode="regular")
layers = {"real": mk(mb)} | {f"shuffled{k}": mk(mb.shuffled(seed=k)) for k in range(args.shuffles)}
print(f"{mb}")
print(f"후보 냄새 {len(ok)}개, 냄새 4개 묶음 {args.sets}개 × seed {args.seeds}개, 무작위 배선 {args.shuffles}개\n")


def make(sets, n, g):
    return fd.biconditional_mixtures(X0, sets, n, args.noise, args.add_noise, args.sat, g)


res = {k: [] for k in ["glomeruli"] + list(layers)}
for s in range(args.seeds):
    t = time.time()
    rng = np.random.default_rng(s)
    sets = [tuple(rng.choice(ok, 4, replace=False)) for _ in range(args.sets)]
    g = torch.Generator().manual_seed(s)
    (xtr, ytr, ptr), (xte, yte, pte) = make(sets, args.n_train, g), make(sets, args.n_test, g)
    F = {"glomeruli": (xtr, xte)} | {k: (fd.extract(L, enc, xtr), fd.extract(L, enc, xte)) for k, L in layers.items()}
    for k, (Ftr, Fte) in F.items():
        for p in range(args.sets):                               # 묶음마다 따로 학습 (한 마리가 한 과제를 배우듯)
            a, b = ptr == p, pte == p
            res[k].append(fd.train_linear(Ftr[a], ytr[a], Fte[b], yte[b], n_classes=2)["test_acc"] * 100)
    print(f"seed {s}: {time.time() - t:.0f}s", flush=True)

glo, real = torch.tensor(res["glomeruli"]), torch.tensor(res["real"])
shuf = torch.tensor([res[f"shuffled{k}"] for k in range(args.shuffles)]).T     # (묶음×seed, 무작위)
diff = real - shuf.mean(1)
print(f"\n조건부 구별 정확도 (찍기 50%, 묶음 {len(real)}개 평균)")
print(f"  glomeruli {glo.mean():5.1f} | KC real {real.mean():5.1f} | KC shuffled {shuf.mean():5.1f} "
      f"(무작위끼리 범위 {shuf.mean(0).min():.1f}~{shuf.mean(0).max():.1f})")
print(f"  실제 − 무작위: {diff.mean():+.2f} ± {diff.std() / len(diff) ** 0.5:.2f} (표준오차) | "
      f"실제가 나은 묶음 {int((diff > 0).sum())}, 같은 묶음 {int((diff == 0).sum())}, 못한 묶음 {int((diff < 0).sum())}")
