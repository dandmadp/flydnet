"""
가장 짧은 예: 실제 초파리 버섯체로 실제 냄새 12개 구분 (학습 루프 한 줄)

  python examples/quickstart.py                 # 약 30초 (GPU)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
ap.add_argument("--epochs", type=int, default=12)
args = ap.parse_args()

mb = fd.Circuit.from_flywire()                                      # 오른쪽 버섯체 (FlyWire v783)
Xtr, ytr, Xte, yte = fd.door_task(n_odors=12)                       # DoOR 실제 냄새 (사구체 활성)
model = fd.Pathway(
    fd.GlomerularEncoder(mb),                                       # 사구체 → PN 발화율
    fd.ConnectomeLayer(mb, "PN", "MBON", t_ms=50, dt=0.5,           # PN → KC → MBON 스파이킹, 실제 배선
                       gains={"PN>KC": 3.0, "KC>MBON": 3.0}, input_mode="regular", trainable=["PN>KC", "KC>MBON"]),
    fd.Homeostasis(),                                               # MBON 발화율 크기 맞추기
    fd.Projection(48, 12),                                          # 리드아웃
)
hist = fd.train(model, Xtr, ytr, val=(Xte, yte), epochs=args.epochs)
print(f"평가 정확도 {hist['val_acc'][-1]:.3f} (찍기 {1 / 12:.3f}), 냄새: {', '.join(fd.door_task.names[:4])} ...")
