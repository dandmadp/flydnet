"""
실험 ⑪: CIFAR-100 클래스 증가 연속 학습 — 사전학습 특징 + 버섯체(실제 FlyWire PN→KC) + 도파민 연합 학습

  python examples/continual_cifar.py                      # ResNet-18 특징 (처음에 CIFAR-100·가중치 받고 특징 캐시)
  python examples/continual_cifar.py --backbone resnet50

클래스 100개를 10개씩 10과제로 나눠 차례로 배움. 과제마다 그 과제 데이터만 한 번(1에폭) 봄.
각 과제가 끝날 때마다 지금까지 본 클래스 전체로 평가 → 마지막 정확도와 과제별 평균.
특징 추출기(ImageNet 사전학습)는 고정. 비교:
  joint      : 모든 데이터를 한꺼번에 여러 에폭 (연속 학습이 아님, 상한선)
  finetune   : 선형 분류기를 과제마다 이어서 학습 (앞의 클래스를 잊음)
  replay-20  : + 클래스당 샘플 20개 저장해 같이 학습
  NCM        : 클래스 평균 (코사인)
  assoc k    : 도파민 연합 학습 (클래스마다 원형 k개) — 특징 그대로 / KC 확장(실제·무작위 배선·가우스 무작위)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--seed", type=int, default=0, help="클래스 순서")
ap.add_argument("--ks", default="1,10", help="연합 학습 원형 수")
ap.add_argument("--k-frac", type=float, default=0.2)
ap.add_argument("--projection", default="gaussian", choices=["sparse", "gaussian"])
ap.add_argument("--side", default="both", choices=["right", "both"], help="both = 양쪽 버섯체 (KC 약 2배)")
args = ap.parse_args()


# ─────────────── 1. 사전학습 특징 (캐시) ───────────────
def features():
    """ImageNet 사전학습 특징 (torch·torchvision이 필요한 곳은 처음 한 번 뽑을 때뿐). 캐시는 numpy (.npz)"""
    cache = Path(args.data) / f"cifar100_{args.backbone}_feats.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    old = cache.with_suffix(".pt")                                    # 예전 torch 캐시가 있으면 옮김
    if old.exists():
        import torch
        out = [t.numpy() for t in torch.load(old)]
    else:
        out = _extract_with_torch()
    np.savez(cache, Xtr=out[0], ytr=out[1], Xte=out[2], yte=out[3])
    return tuple(out)


def _extract_with_torch():
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = {"resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
         "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2}[args.backbone]
    net = getattr(torchvision.models, args.backbone)(weights=w)
    net.fc = nn.Identity()
    net = net.to(dev).eval()
    tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR100(args.data, train=train, download=True, transform=tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=0)
        F, Y = [], []
        t = time.time()
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            for x, y in dl:
                F.append(net(x.to(dev)).float().cpu()); Y.append(y)
        out += [torch.cat(F).numpy(), torch.cat(Y).numpy()]
        print(f"  특징 {'학습' if train else '평가'} {len(out[-1])}개, {time.time() - t:.0f}s", flush=True)
    return out


Xtr, ytr, Xte, yte = features()
Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
ytr, yte = ytr.astype(np.int64), yte.astype(np.int64)
print(f"특징 {args.backbone}: {tuple(Xtr.shape)}")
order = np.random.default_rng(args.seed).permutation(100)
tasks = order.reshape(args.tasks, -1)                                 # 과제별 클래스


def run_task_loop(learn, predict):
    """과제를 차례로 배우며, 매 과제 뒤 지금까지 본 클래스로 평가 → [정확도]"""
    accs, seen = [], []
    for t, cls in enumerate(tasks):
        seen += cls.tolist()
        m = np.isin(ytr, cls)
        learn(Xtr[m], ytr[m], seen)
        mt = np.isin(yte, seen)
        accs.append(float((np.asarray(predict(Xte[mt], seen)) == yte[mt]).mean()))
    return accs


# ─────────────── 2. 역전파 기준선 ───────────────
mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
norm = lambda X: ((X - mu) / sd).astype(np.float32)


def _mask(seen):
    m = np.full(100, -1e9, np.float32)                                 # 아직 안 본 클래스는 고르지 않음
    m[seen] = 0
    return m


def linear_cl(buffer_per_class=0, epochs=1, lr=1e-3):
    lin = fd.Projection(Xtr.shape[1], 100, seed=0)
    rule = fd.Adaptive(lin.named_synapses(), rate=lr)
    buf_x, buf_y = [], []

    def learn(X, y, seen):
        Xb, yb = X, y
        if buf_x:
            Xb, yb = np.concatenate([X] + buf_x), np.concatenate([y] + buf_y)
        mask = _mask(seen)
        g = np.random.default_rng(len(seen))
        for _ in range(epochs):
            perm = g.permutation(len(Xb))
            for i in range(0, len(Xb), 64):
                j = perm[i:i + 64]
                rule.clear(); fd.surprise(lin(norm(Xb[j])) + mask, yb[j]).retrograde(); rule.step()
        if buffer_per_class:
            for c in np.unique(y):
                idx = np.nonzero(y == c)[0][:buffer_per_class]
                buf_x.append(X[idx]); buf_y.append(y[idx])

    def predict(X, seen):
        with fd.quiescent():
            return (lin(norm(X)).numpy() + _mask(seen)).argmax(1)
    return run_task_loop(learn, predict)


def assoc_cl(encode, k):
    ro = fd.AssocReadout(int(np.shape(encode(Xtr[:2]))[1]), 100, per_class=k)
    learn = lambda X, y, seen: ro.fit(encode(X), y, batch=256)
    predict = lambda X, seen: ro.predict(encode(X), classes=seen)
    return run_task_loop(learn, predict)


results = {}
t0 = time.time()
r = fd.train_linear(Xtr, ytr, Xte, yte, n_classes=100, epochs=30)
results["joint (상한선, 연속 학습 아님)"] = [r["test_acc"]]
results["finetune 선형"] = linear_cl()
results["replay-20 선형"] = linear_cl(buffer_per_class=20)
print(f"역전파 기준선 {time.time() - t0:.0f}s", flush=True)

# ─────────────── 3. 버섯체 확장 + 도파민 연합 학습 ───────────────
mb = fd.Circuit.from_flywire(side=None if args.side == "both" else "right")
n_in = Xtr.shape[1]
expanders = {
    "KC 실제 배선": fd.KCExpansion(mb, n_in=n_in, k_frac=args.k_frac, projection=args.projection),
    "KC 무작위 배선": fd.KCExpansion(mb.shuffled(seed=0), n_in=n_in, k_frac=args.k_frac, projection=args.projection),
}
# 같은 크기의 희소 무작위 확장 (커넥톰 없이 흔히 쓰는 방식) — 같은 PN 투영 뒤, 같은 연결 밀도의 무작위 행렬
g_exp = fd.KCExpansion(mb, n_in=n_in, k_frac=args.k_frac, projection=args.projection)
W0 = fd.ganglion.backend.numpy(g_exp.W)
rng = np.random.default_rng(1)
W = (np.abs(rng.standard_normal(W0.shape)) * (rng.random(W0.shape) < (W0 > 0).mean())).astype(np.float32)
g_exp.buffer("W", fd.ganglion.backend.to(W, fd.ganglion.backend.device_of(g_exp.W)))
g_exp.name = "희소 무작위 (같은 연결 밀도)"
expanders["KC 희소 무작위 행렬"] = g_exp
print(expanders["KC 실제 배선"])

KS = [int(k) for k in args.ks.split(",")]
results["NCM (클래스 평균)"] = assoc_cl(lambda X: X, 1)
for k in KS:
    if k > 1:
        results[f"assoc k={k}, 특징 그대로"] = assoc_cl(lambda X: X, k)
for name, ex in expanders.items():
    for k in KS:
        results[f"assoc k={k}, {name}"] = assoc_cl(ex, k)
print(f"전체 {time.time() - t0:.0f}s\n")

print(f"CIFAR-100 클래스 증가 {args.tasks}과제 ({args.backbone} 특징 고정, 과제당 1에폭)")
print(f"{'방법':<34}{'마지막':>8}{'과제 평균':>10}")
for name, a in results.items():
    print(f"{name:<34}{a[-1] * 100:>7.1f}%{np.mean(a) * 100:>9.1f}%")
