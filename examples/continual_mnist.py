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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd
from _mnist import mnist

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
ap.add_argument("--kc-cache", default="feat_60000_10000_100ms_g2_r100_regular_brian_real.npz")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--ks", default="1,20,50", help="연합 원형 수 / 재생 버퍼 크기 (클래스당)")
args = ap.parse_args()
KS = [int(k) for k in args.ks.split(",")]

Xtr, ytr, Xte, yte = mnist(args.data)
feats = {"pixels": (Xtr, Xte)}
cache = Path(args.data) / args.kc_cache
if cache.exists():
    with np.load(cache) as z:
        feats["KC"] = (z["Ftr"][:, :2597], z["Fte"][:, :2597])
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class Backprop:
    """로지스틱 회귀 + (선택) 재생 버퍼. 버퍼는 클래스당 처음 본 m개 샘플 (자체 엔진, Adam 1e-3, 묶음 32)"""

    def __init__(self, n_in, mu, sd, replay=0, lr=1e-3, seed=0):
        self.mu, self.sd, self.m = mu, sd, replay
        self.net = fd.Projection(n_in, 10, seed=seed)
        self.rule = fd.Adaptive(self.net.named_synapses(), rate=lr)
        self.bx, self.by = [], []

    def f(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def fit(self, X, y, seed=0, batch=32, classes=None):
        g = np.random.default_rng(seed)
        if self.m:
            for c in np.unique(y):
                self.bx.append(X[y == c][:self.m]); self.by.append(y[y == c][:self.m])
            BX, BY = np.concatenate(self.bx), np.concatenate(self.by)
        perm = g.permutation(len(X))
        for i in range(0, len(X), batch):
            j = perm[i:i + batch]
            x, t = X[j], y[j]
            if self.m:                                          # 옛 샘플 같은 수만큼 섞기
                r = g.integers(0, len(BX), len(j))
                x, t = np.concatenate([x, BX[r]]), np.concatenate([t, BY[r]])
            self.rule.clear(); fd.surprise(self.net(self.f(x)), t).retrograde(); self.rule.step()

    def accuracy(self, X, y, classes):
        with fd.quiescent():
            s = self.net(self.f(X)).numpy()
        mask = np.full(10, -np.inf); mask[list(classes)] = 0
        return float(((s + mask).argmax(1) == y).mean())


def run(model, Ftr, Fte, seed):
    seen, curve = [], []
    for t, task in enumerate(TASKS):
        seen += list(task)
        i = np.isin(ytr, task)
        model.fit(Ftr[i], ytr[i], seed=seed * 10 + t, classes=seen)
        j = np.isin(yte, seen)
        curve.append(model.accuracy(Fte[j], yte[j], classes=seen))
    first = np.isin(yte, TASKS[0])
    return curve, model.accuracy(Fte[first], yte[first], classes=seen)


print("class-incremental split MNIST, 과제당 1에폭, "
      f"{args.seeds}회 평균 | 표: 단계별 '본 숫자 전체' 정확도 → 최종 | 마지막 후 0/1 정확도\n")
for name, (Ftr, Fte) in feats.items():
    mu = Ftr.mean(0)
    sd = max(float((Ftr - mu).std()), 1e-6)
    methods = {"역전파": lambda s: Backprop(Ftr.shape[1], mu, sd, seed=s)}
    for k in KS:
        methods[f"역전파 + 재생 {k}"] = lambda s, k=k: Backprop(Ftr.shape[1], mu, sd, replay=k, seed=s)
        methods[f"연합 {k}"] = lambda s, k=k: fd.AssocReadout(Ftr.shape[1], 10, per_class=k)
    print(f"[{name}, {Ftr.shape[1]}차원]")
    for mname, make in methods.items():
        runs = [run(make(s), Ftr, Fte, s) for s in range(args.seeds)]
        curve = np.array([r[0] for r in runs]) * 100
        first = np.array([r[1] for r in runs]) * 100
        spread = curve[:, -1].std(ddof=1) if len(curve) > 1 else 0.0
        print(f"  {mname:<14} " + " → ".join(f"{c:5.1f}" for c in curve.mean(0))
              + f"  (최종 ± {spread:.1f}) | 0/1 {first.mean():5.1f}", flush=True)
    print()
