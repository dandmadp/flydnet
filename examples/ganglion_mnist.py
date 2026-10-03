"""
flydnet 자체 엔진(ganglion)으로 MNIST 학습 — torch 없이

  python examples/ganglion_mnist.py               # GPU가 있으면 GPU (CuPy), 없으면 CPU (NumPy)
  python examples/ganglion_mnist.py --device cpu

픽셀 784 ─Projection─▶ PN 344 ─Neuropil(실제 FlyWire PN→KC 배선)─▶ KC 2,597 ─LateralInhibition(5%)─▶ ─Projection─▶ 10
MNIST 파일은 data/MNIST/raw/ 의 원본(idx) 파일을 직접 읽음 (torchvision 불필요).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd
import flydnet.ganglion as G

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "MNIST" / "raw"))
ap.add_argument("--device", default=None, choices=["cpu", "gpu"])
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--rate", type=float, default=1e-3)
ap.add_argument("--shuffled", action="store_true", help="무작위 배선 대조군")
args = ap.parse_args()
dev = args.device or G.default_device()


def idx(path):
    """MNIST idx 파일 읽기"""
    raw = Path(path).read_bytes()
    nd = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * nd).reshape(dims)


d = Path(args.data)
Xtr = idx(d / "train-images-idx3-ubyte").reshape(60000, -1).astype(np.float32) / 255
ytr = idx(d / "train-labels-idx1-ubyte").astype(np.int64)
Xte = idx(d / "t10k-images-idx3-ubyte").reshape(10000, -1).astype(np.float32) / 255
yte = idx(d / "t10k-labels-idx1-ubyte").astype(np.int64)

mb = fd.Circuit.from_flywire()
if args.shuffled:
    mb = mb.shuffled(seed=0)
model = G.Pathway(
    G.Projection(784, 344, seed=0, device=dev),
    G.Neuropil(mb, "PN", "KC", device=dev),
    G.LateralInhibition(frac=0.05),
    G.Projection(2597, 10, seed=1, device=dev),
)
print(model)
print(f"장치 {dev}, 학습 시냅스 {model.n_synapses():,}개", flush=True)

xp = G.backend.xp(dev)
Xtr_d, ytr_d = G.backend.to(Xtr, dev), G.backend.to(ytr, dev)
rule = G.AdaptivePlasticity(model.synapses(), rate=args.rate)
rng = np.random.default_rng(0)
t0 = time.time()
for ep in range(args.epochs):
    perm = G.backend.to(rng.permutation(60000), dev)
    for i in range(0, 60000, args.batch):
        j = perm[i:i + args.batch]
        loss = G.surprise(model(G.Signal(Xtr_d[j])), ytr_d[j])
        rule.clear(); loss.retrograde(); rule.step()
    with G.quiescent():
        acc = np.mean([(G.backend.numpy(model(Xte[i:i + 2000]).data).argmax(1) == yte[i:i + 2000]).mean()
                       for i in range(0, 10000, 2000)])
    print(f"에폭 {ep + 1}: 손실 {loss.item():.3f}, 평가 정확도 {acc * 100:.2f}%, 누적 {time.time() - t0:.1f}s", flush=True)
