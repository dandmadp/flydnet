"""
fd.torch.bridge 예: torch 모델 안에 flydnet 자체 엔진의 버섯체 층을 넣고 torch 옵티마이저로 학습

  python examples/torch_bridge.py              # 약 1분 (GPU)

모델 (torch.nn.Sequential):
  사구체 → [torch] 고정된 사구체 → PN 사상 → [flydnet] 버섯체 ConnectomeLayer (PN → KC → MBON, 연결 학습)
         → [torch] LayerNorm → Linear → 냄새 클래스
과제: DoOR 2.0 실제 냄새 12개 (잡음 섞인 시료)
torch 쪽은 바깥 층·옵티마이저·학습률 스케줄만, 커넥톰 계산은 전부 자체 엔진 (원본 Brian2와 같은 계산, 전용 GPU 커널)
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
ap.add_argument("--odors", type=int, default=12)
ap.add_argument("--samples", type=int, default=24)
ap.add_argument("--noise", type=float, default=0.8)
ap.add_argument("--epochs", type=int, default=12)
args = ap.parse_args()
fd.ganglion.limit_gpu_memory(0.6)
dev = "cuda" if torch.cuda.is_available() and fd.ganglion.gpu_available() else "cpu"
torch.manual_seed(0)

mb = fd.Circuit.from_flywire(dict(fd.MUSHROOM_BODY, DAN=("cell_class", "DAN")))
enc = fd.GlomerularEncoder(mb, device="cpu")
door = fd.door_odors(enc.glomeruli)
pick = np.argsort(door["X"].sum(1))[::-1][:args.odors]
proto = door["X"][pick] / door["X"][pick].max()
rng = np.random.default_rng(0)


def make(n):
    y = np.repeat(np.arange(args.odors), n)
    X = proto[y] * rng.lognormal(0, args.noise, (len(y), proto.shape[1])) + rng.uniform(0, 0.15, (len(y), proto.shape[1]))
    return torch.tensor(np.clip(X, 0, 1), dtype=torch.float32, device=dev), torch.tensor(y, device=dev)


Xtr, ytr = make(args.samples)
Xte, yte = make(args.samples)
P = torch.tensor(fd.ganglion.backend.numpy(enc.P), device=dev)              # 사구체 → PN (고정)
kw = dict(t_ms=50, dt=0.5, gains={"PN>KC": 3.0, "KC>MBON": 3.0}, input_mode="regular", trainable=["PN>KC", "KC>MBON"])


class Glomeruli(torch.nn.Module):
    def forward(self, x):
        return 100.0 * x @ P.T


def train(core, name, show_acc=True):
    model = torch.nn.Sequential(Glomeruli(), core, torch.nn.LayerNorm(48), torch.nn.Linear(48, args.odors)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    times = []
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            t = time.perf_counter()
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(Xtr[j]), ytr[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        sched.step()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f"  {name:<40} " + (f"정확도 {acc:.3f}   " if show_acc else "") + f"학습 1스텝 {np.median(times) * 1000:.0f} ms")
    return model


print(f"장치 {dev}, 냄새 {args.odors}개, 시료 {len(Xtr)}개")
layer = fd.ConnectomeLayer(mb, "PN", "MBON", device="gpu" if dev == "cuda" else "cpu", **kw)
model = train(fd.torch.bridge(layer, seed=0), "flydnet 자체 엔진 (bridge, timing=brian)")
print(f"  학습한 커넥톰 값이 자체 엔진에도 반영됨: 배율 평균 {np.exp(layer.log_scale.numpy()).mean():.3f} (처음 1.000)")
