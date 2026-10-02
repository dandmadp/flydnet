"""
실험 ⑧: 커넥톰 층을 PyTorch 층처럼 역전파로 학습 — MNIST

  python examples/train_backprop.py
  python examples/train_backprop.py --n-train 5000 --epochs 1   # 빠르게

모델 = RateEncoder → ConnectomeLayer(버섯체, 대리 기울기) → nn.Linear, 일반 PyTorch 학습 루프
배선은 고정, 연결별 세기만 학습 (부호 유지). dt 0.5 ms, 50 ms 창, PN→KC 배율 2.0 (KC 약 5% 활성)

설정
  고정 + 선형        : 커넥톰 고정, 마지막 선형 층만 학습 (실험 ①과 같은 방식)
  전체 학습 (KC)     : 연결 19만 개 전부 학습, KC 발화율 → 선형
  KC→MBON만 (MBON)  : 초파리에서 도파민으로 바뀌는 KC→MBON 연결만 학습, MBON 48개 → 선형
  무작위 전체 학습    : 무작위 배선에서 연결 전부 학습 (학습해도 실제 배선이 나은가)
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=20000)
ap.add_argument("--n-test", type=int, default=10000)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-3, help="선형 층 학습률")
ap.add_argument("--conn-lr", type=float, default=3e-2, help="커넥톰 연결 세기(log 배율) 학습률")
ap.add_argument("--dt", type=float, default=0.5)
ap.add_argument("--t-ms", type=float, default=50)
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

tr, te = datasets.MNIST(args.data, train=True, download=True), datasets.MNIST(args.data, train=False, download=True)
Xtr, ytr = tr.data[:args.n_train].float() / 255, tr.targets[:args.n_train]
Xte, yte = te.data[:args.n_test].float() / 255, te.targets[:args.n_test]
mb = fd.Circuit.from_flywire()


class PerHundredHz(nn.Module):
    """발화율(Hz)을 선형 층에 넣기 좋은 크기로"""
    def forward(self, x):
        return x / 100.0


def build(circuit, outputs, trainable):
    layer = fd.ConnectomeLayer(circuit, "PN", outputs, t_ms=args.t_ms, dt=args.dt, gains={"PN>KC": 2.0},
                               input_mode="regular", trainable=trainable)
    return nn.Sequential(fd.RateEncoder(784, len(circuit.groups["PN"])), layer, PerHundredHz(),
                         nn.Linear(layer.n_out, 10)).to(dev)


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    correct = sum((model(X[i:i + 256].to(dev)).argmax(1).cpu() == y[i:i + 256]).sum().item()
                  for i in range(0, len(X), 256))
    return correct / len(X) * 100


CONFIGS = {
    "고정 + 선형": (mb, "KC", False),
    "전체 학습 (KC)": (mb, "KC", True),
    "KC→MBON만 (MBON)": (mb, "MBON", ["KC>MBON"]),
    "무작위 전체 학습 (KC)": (mb.shuffled(seed=0), "KC", True),
}
print(f"{mb}\nMNIST 학습 {len(Xtr)} / 평가 {len(Xte)}, {args.epochs}에폭, dt {args.dt} ms, {args.t_ms} ms 창\n")
summary = {}
for name, (circ, out, trainable) in CONFIGS.items():
    torch.manual_seed(0)
    model = build(circ, out, trainable)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conn = [p for n, p in model.named_parameters() if n.endswith("log_scale")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("log_scale")]
    opt = torch.optim.Adam([{"params": rest, "lr": args.lr}, {"params": conn, "lr": args.conn_lr}])
    g = torch.Generator().manual_seed(0)
    t = time.time()
    curve = []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), args.batch):
            j = perm[i:i + args.batch]
            loss = nn.functional.cross_entropy(model(Xtr[j].to(dev)), ytr[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        curve.append(evaluate(model, Xte, yte))
    layer = model[1]
    w = layer.log_scale.detach().exp() if layer.trainable else None
    wtxt = f" | 연결 세기 배율 중앙값 {w.median():.2f} (5~95%: {w.quantile(.05):.2f}~{w.quantile(.95):.2f})" if w is not None else ""
    print(f"{name:<18} 학습 파라미터 {n_par:>7,} | 에폭별 test " + " → ".join(f"{c:.2f}" for c in curve)
          + f" | {time.time() - t:.0f}s{wtxt}", flush=True)
    summary[name] = curve[-1]
