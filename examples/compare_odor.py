"""
fd.compare 예: 버섯체 배선이 냄새 분류에 중요한가 (torch 없이)

  python examples/compare_odor.py                 # 빠른 앞먹임 KC 확장 (KCExpansion)
  python examples/compare_odor.py --model lif     # 스파이킹 시뮬레이션 (ConnectomeLayer)

과제: 합성 냄새 (사구체 56개, 클래스 30개, 클래스마다 원형 2개, 잡음 큼 → 상한에 걸리지 않게 어렵게)
모델: 사구체 → PN (GlomerularEncoder) → KC (실제 또는 대조군 배선) → 로지스틱 회귀
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="kc", choices=["kc", "lif"])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--classes", type=int, default=30)
ap.add_argument("--noise", type=float, default=0.8)
ap.add_argument("--timing", default="brian", choices=["brian", "legacy"], help="LIF 한 스텝 (legacy = 0.1.15까지, REPORT ⑫)")
args = ap.parse_args()

mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)


def run(circuit, seed):
    Xtr, ytr, Xte, yte = fd.synthetic_odors(args.classes, enc.n_glomeruli, 20, 20, protos_per_class=2,
                                             noise=args.noise, seed=seed)
    if args.model == "kc":
        kc = fd.KCExpansion(circuit, k_frac=0.05)                     # PN 활동을 바로 받음 (투영 없음)
        f = lambda X: kc(enc(X)).numpy()
    else:
        layer = fd.ConnectomeLayer(circuit, "PN", "KC", t_ms=50, dt=0.5, gains={"PN>KC": 3.0}, input_mode="regular",
                                   timing=args.timing)
        f = lambda X: fd.extract(layer, X, enc, batch=200)
    return fd.train_linear(f(Xtr), ytr, f(Xte), yte, epochs=60, seed=seed)["test_acc"]


report = fd.compare(run, mb, controls=["shuffled", "randomized", "shuffled_weights"], seeds=args.seeds,
                    chance=1 / args.classes)
print()
print(report)
