"""
실험 ②: 도파민 학습 리드아웃 (역전파 없음) — MNIST
  먼저 실험 ① (examples/mnist_reservoir.py)을 기본 설정으로 돌려 KC 특징 캐시를 만들어야 함

  python examples/mnist_dopamine.py

A. 전체 학습: KC 특징 위에 도파민 규칙 vs 역전파(로지스틱 회귀), 실제 배선 vs 무작위 배선
B. 연속 학습 (class-incremental split MNIST): 0/1 → 2/3 → 4/5 → 6/7 → 8/9 순서로 한 번씩만 학습,
   매 단계 후 지금까지 본 모든 숫자로 평가. 역전파는 앞서 배운 것을 잊는 '파국적 망각'이 알려져 있음
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
ap.add_argument("--tag", default="60000_10000_100ms_g2_r100_regular", help="실험 ① 캐시 이름")
ap.add_argument("--lr", type=float, default=0.1, help="도파민 규칙 학습률")
ap.add_argument("--epochs", type=int, default=2, help="A의 에폭 수")
ap.add_argument("--seeds", type=int, default=3, help="반복 횟수 (평균 ± 표준편차)")
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tr, te = datasets.MNIST(args.data, train=True), datasets.MNIST(args.data, train=False)
ytr, yte = tr.targets, te.targets
N_KC = 2597


def load(name):
    Ftr, Fte = torch.load(Path(args.data) / f"feat_{args.tag}_{name}.pt")
    return Ftr[:, :N_KC], Fte[:, :N_KC]


feats = {"pixels": (tr.data.flatten(1).float() / 255, te.data.flatten(1).float() / 255),
         "KC real": load("real"), "KC shuffled": load("shuffled0")}


def mean_sd(xs):
    t = torch.tensor(xs) * 100
    return f"{t.mean():6.2f} ± {t.std():.2f}%" if len(xs) > 1 else f"{t.mean():6.2f}%"


# ───────────── A. 전체 학습 ─────────────
print(f"A. 전체 학습 ({args.seeds}회 반복)\n{'특징':<13}{'역전파(로지스틱)':>20}{'도파민 bidir':>20}{'도파민 assoc':>16}")
for name, (Ftr, Fte) in feats.items():
    bp = [fd.torch.train_linear(Ftr, ytr, Fte, yte)["test_acc"]]           # 결정적이라 1회
    da = [fd.torch.DopamineReadout(Ftr.shape[1], 10, "bidir", args.lr).fit(Ftr, ytr, args.epochs, seed=s).accuracy(Fte, yte)
          for s in range(args.seeds)]
    asc = [fd.torch.DopamineReadout(Ftr.shape[1], 10, "assoc").fit(Ftr, ytr).accuracy(Fte, yte)]  # 순서 무관 → 1회
    print(f"{name:<13}{mean_sd(bp):>20}{mean_sd(da):>20}{mean_sd(asc):>16}", flush=True)


# ───────────── B. 연속 학습 ─────────────
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class SGDLogistic:
    """비교용: 역전파로 한 샘플 묶음씩 학습하는 로지스틱 회귀 (출력 10개, 소프트맥스)"""

    def __init__(self, n_in, mu, sd, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.mu, self.sd = mu.to(dev), sd.to(dev)
        self.m = nn.Linear(n_in, 10).to(dev)
        self.opt = torch.optim.Adam(self.m.parameters(), lr=lr)

    def f(self, X):
        return (X.to(dev).float() - self.mu) / self.sd

    def fit(self, X, y, seed=0, batch=32, classes=None):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            j = perm[i:i + batch]
            loss = nn.functional.cross_entropy(self.m(self.f(X[j])), y[j].to(dev))
            self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def accuracy(self, X, y, classes):
        s = self.m(self.f(X))
        mask = torch.full((10,), float("-inf"), device=dev); mask[list(classes)] = 0
        return ((s + mask).argmax(1).cpu() == y).float().mean().item()


def continual(make, Ftr, Fte, seed):
    model, seen, curve = make(seed), [], []
    for t, task in enumerate(TASKS):
        seen += list(task)
        itr = torch.isin(ytr, torch.tensor(task))
        model.fit(Ftr[itr], ytr[itr], seed=seed * 10 + t, classes=seen)
        ite = torch.isin(yte, torch.tensor(seen))
        curve.append(model.accuracy(Fte[ite], yte[ite], classes=seen))
    # 마지막에 과제별 정확도 (첫 과제를 얼마나 기억하나)
    per_task = [model.accuracy(Fte[torch.isin(yte, torch.tensor(k))], yte[torch.isin(yte, torch.tensor(k))], classes=seen)
                for k in TASKS]
    return curve, per_task


print(f"\nB. 연속 학습 (0/1 → 2/3 → 4/5 → 6/7 → 8/9, 과제당 1에폭, {args.seeds}회 평균)")
print("   표: 각 단계 후 '지금까지 본 숫자 전체' 정확도 / 마지막 단계 후 과제별 정확도")
for name, (Ftr, Fte) in feats.items():
    mu, sd = Ftr.mean(0), (Ftr - Ftr.mean(0)).std().clamp_min(1e-6)
    makers = {"역전파": lambda s: SGDLogistic(Ftr.shape[1], mu, sd, seed=s),
              "bidir": lambda s: fd.torch.DopamineReadout(Ftr.shape[1], 10, "bidir", args.lr),
              "assoc": lambda s: fd.torch.DopamineReadout(Ftr.shape[1], 10, "assoc")}
    for mname, make in makers.items():
        runs = [continual(make, Ftr, Fte, s) for s in range(args.seeds)]
        curve = torch.tensor([r[0] for r in runs]).mean(0) * 100
        per = torch.tensor([r[1] for r in runs]).mean(0) * 100
        print(f"  {name:<12} {mname}: 단계별 " + " → ".join(f"{c:5.1f}" for c in curve)
              + " | 과제별 " + " ".join(f"{p:5.1f}" for p in per), flush=True)
