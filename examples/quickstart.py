"""
가장 짧은 예: 실제 초파리 버섯체로 실제 냄새 12개 구분 (가중치 자동 보정 + 학습 루프 한 줄)

  python examples/quickstart.py                 # 약 30초 (GPU), 평가 정확도 약 0.96 (찍기 0.083)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--read", default="KC", choices=["KC", "MBON"], help="어디서 읽을지 (MBON 48개는 병목: 약 0.57)")
args = ap.parse_args()

mb = fd.flywire()                                                   # 오른쪽 버섯체 (FlyWire v783)
Xtr, ytr, Xte, yte = fd.door_task(n_odors=12)                       # DoOR 실제 냄새 (사구체 활성)
enc = fd.Glomeruli(mb)                                              # 사구체 → PN 발화율
layer = fd.Connectome(mb, "PN", args.read, t_ms=50, dt=0.5, input_mode="regular", trainable=["PN>KC", "KC>MBON"])
print(layer.calibrate(enc(Xtr[::3]), {"KC": 5, "MBON": 20}).round(2).to_string(index=False))   # 가중치 자동 보정
model = fd.Pathway(enc, layer, fd.Homeostasis(), fd.Projection(len(mb.groups[args.read]), 12))
hist = fd.train(model, Xtr, ytr, val=(Xte, yte), epochs=args.epochs)
print(f"평가 정확도 {hist['val_acc'][-1]:.3f} (찍기 {1 / 12:.3f}), 냄새: {', '.join(fd.door_task.names[:4])} ...")
