"""
실험 ⑥: 연속 학습 비교 — 도파민 연합 학습(AssocReadout) 대 역전파(재생 버퍼 포함)
  실험 ①의 KC 특징 캐시가 있으면 KC로도 비교 (없으면 픽셀만)

  python examples/continual_mnist.py

class-incremental split MNIST: 0/1 → 2/3 → 4/5 → 6/7 → 8/9, 과제마다 한 번(1에폭)씩만, 매 단계 후
지금까지 본 숫자 전체로 평가. 방법:
  역전파            : 로지스틱 회귀를 새 과제 데이터로만 학습 (대책 없음)
  역전파 + 재생 m    : 클래스당 m개 옛 샘플을 저장해 두고 매 묶음에 섞어 학습 (표준 대책)
  연합 k            : AssocReadout, 클래스당 원형 k개 (역전파 없음, 다른 클래스는 안 바뀜)
저장량을 맞추기 위해 재생 m = 연합 k로 비교 (클래스당 벡터 m개 = 원형 k개)
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
ap.add_argument("--kc-cache", default="feat_60000_10000_100ms_g2_r100_regular_real.pt")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--ks", default="1,20,50", help="연합 원형 수 / 재생 버퍼 크기 (클래스당)")
args = ap.parse_args()
KS = [int(k) for k in args.ks.split(",")]

dev = "cuda" if torch.cuda.is_available() else "cpu"
tr, te = datasets.MNIST(args.data, train=True), datasets.MNIST(args.data, train=False)
ytr, yte = tr.targets, te.targets
feats = {"pixels": (tr.data.flatten(1).float() / 255, te.data.flatten(1).float() / 255)}
cache = Path(args.data) / args.kc_cache
if cache.exists():
    a, b = torch.load(cache)
    feats["KC"] = (a[:, :2597], b[:, :2597])
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class Backprop:
    """로지스틱 회귀 + (선택) 재생 버퍼. 버퍼는 클래스당 처음 본 m개 샘플"""

    def __init__(self, n_in, mu, sd, replay=0, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.mu, self.sd, self.m = mu.to(dev), sd.to(dev), replay
        self.net = nn.Linear(n_in, 10).to(dev)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.bx, self.by = [], []

    def f(self, X):
        return (X.to(dev).float() - self.mu) / self.sd

    def fit(self, X, y, seed=0, batch=32, classes=None):
        g = torch.Generator().manual_seed(seed)
        if self.m:
            for c in y.unique():
                self.bx.append(X[y == c][:self.m]); self.by.append(y[y == c][:self.m])
            BX, BY = torch.cat(self.bx), torch.cat(self.by)
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            j = perm[i:i + batch]
            x, t = X[j], y[j]
            if self.m:                                          # 옛 샘플 같은 수만큼 섞기
                r = torch.randint(0, len(BX), (len(j),), generator=g)
                x, t = torch.cat([x, BX[r]]), torch.cat([t, BY[r]])
            loss = nn.functional.cross_entropy(self.net(self.f(x)), t.to(dev))
            self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def accuracy(self, X, y, classes):
        s = self.net(self.f(X))
        mask = torch.full((10,), float("-inf"), device=dev); mask[list(classes)] = 0
        return ((s + mask).argmax(1).cpu() == y).float().mean().item()


def run(model, Ftr, Fte, seed):
    seen, curve = [], []
    for t, task in enumerate(TASKS):
        seen += list(task)
        i = torch.isin(ytr, torch.tensor(task))
        model.fit(Ftr[i], ytr[i], seed=seed * 10 + t, classes=seen)
        j = torch.isin(yte, torch.tensor(seen))
        curve.append(model.accuracy(Fte[j], yte[j], classes=seen))
    first = torch.isin(yte, torch.tensor(TASKS[0]))
    return curve, model.accuracy(Fte[first], yte[first], classes=seen)


print("class-incremental split MNIST, 과제당 1에폭, "
      f"{args.seeds}회 평균 | 표: 단계별 '본 숫자 전체' 정확도 → 최종 | 마지막 후 0/1 정확도\n")
for name, (Ftr, Fte) in feats.items():
    mu, sd = Ftr.mean(0), (Ftr - Ftr.mean(0)).std().clamp_min(1e-6)
    methods = {"역전파": lambda s: Backprop(Ftr.shape[1], mu, sd, seed=s)}
    for k in KS:
        methods[f"역전파 + 재생 {k}"] = lambda s, k=k: Backprop(Ftr.shape[1], mu, sd, replay=k, seed=s)
        methods[f"연합 {k}"] = lambda s, k=k: fd.torch.AssocReadout(Ftr.shape[1], 10, per_class=k)
    print(f"[{name}, {Ftr.shape[1]}차원]")
    for mname, make in methods.items():
        runs = [run(make(s), Ftr, Fte, s) for s in range(args.seeds)]
        curve = torch.tensor([r[0] for r in runs]) * 100
        first = torch.tensor([r[1] for r in runs]) * 100
        print(f"  {mname:<14} " + " → ".join(f"{c:5.1f}" for c in curve.mean(0))
              + f"  (최종 ± {curve[:, -1].std():.1f}) | 0/1 {first.mean():5.1f}", flush=True)
    print()
