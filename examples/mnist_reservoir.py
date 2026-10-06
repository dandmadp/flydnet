"""
실험 ①: 버섯체를 고정 '저장소(reservoir)'로 쓰고 리드아웃만 학습 — MNIST

  python examples/mnist_reservoir.py                 # 전체 (train 60k / test 10k)
  python examples/mnist_reservoir.py --n-train 10000 # 빠르게

비교 (모두 같은 로지스틱 회귀 리드아웃)
  pixels        : 원본 픽셀 784차원                         (기준선)
  PN            : 인코더 출력 = PN 발화율 344차원            (커넥톰 전 단계)
  KC real       : 실제 FlyWire 배선의 KC 발화율 2,597차원
  KC shuffled k : 연결 수는 같고 배선만 무작위인 회로의 KC   (실제 배선이 의미 있는지 확인, 여러 개)
  MBON real     : 실제 배선 MBON 발화율 48차원
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd
from _mnist import mnist

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=60000)
ap.add_argument("--n-test", type=int, default=10000)
ap.add_argument("--t-ms", type=float, default=100)
ap.add_argument("--pn-kc-gain", type=float, default=2.0, help="PN→KC 시냅스 배율")
ap.add_argument("--max-rate", type=float, default=100)
ap.add_argument("--input-mode", choices=["regular", "poisson"], default="regular")
ap.add_argument("--n-shuffles", type=int, default=3, help="무작위 배선 대조군 개수")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
args = ap.parse_args()

Xtr, ytr, Xte, yte = mnist(args.data)
Xtr, ytr, Xte, yte = Xtr[:args.n_train], ytr[:args.n_train], Xte[:args.n_test], yte[:args.n_test]

mb = fd.Circuit.from_flywire()
print(mb)
enc = fd.RateEncoder(784, len(mb.groups["PN"]), max_rate=args.max_rate)
n_kc = len(mb.groups["KC"])

feats = {"pixels": (Xtr, Xte), "PN": (fd.extract(enc, Xtr), fd.extract(enc, Xte))}
circuits = [("real", mb)] + [(f"shuffled {k}", mb.shuffled(seed=k)) for k in range(args.n_shuffles)]
tag = f"{args.n_train}_{args.n_test}_{args.t_ms:g}ms_g{args.pn_kc_gain:g}_r{args.max_rate:g}_{args.input_mode}_brian"
for name, circ in circuits:
    cache = Path(args.data) / f"feat_{tag}_{name.replace(' ', '')}.npz"      # 자체 엔진(timing="brian") 특징
    if cache.exists():
        with np.load(cache) as z:
            Ftr, Fte = z["Ftr"], z["Fte"]
    else:
        layer = fd.ConnectomeLayer(circ, "PN", ("KC", "MBON"), t_ms=args.t_ms,
                                   gains={"PN>KC": args.pn_kc_gain}, input_mode=args.input_mode)
        t = time.time()
        Ftr = fd.extract(layer, Xtr, enc, batch=args.batch, seed=0)
        Fte = fd.extract(layer, Xte, enc, batch=args.batch, seed=10**6)
        np.savez(cache, Ftr=Ftr, Fte=Fte)
        print(f"  {name}: 시뮬레이션 {time.time() - t:.0f}s", flush=True)
    print(f"  {name}: KC 활성 {(Ftr[:, :n_kc] > 0).mean() * 100:.1f}%", flush=True)
    feats[f"KC {name}"] = (Ftr[:, :n_kc], Fte[:, :n_kc])
    feats[f"MBON {name}"] = (Ftr[:, n_kc:], Fte[:, n_kc:])

print(f"\n{'특징':<16}{'차원':>6}{'train':>9}{'test':>9}")
for name in ["pixels", "PN"] + [f"{g} {c}" for g in ("KC", "MBON") for c, _ in circuits]:
    Ftr, Fte = feats[name]
    r = fd.train_linear(Ftr, ytr, Fte, yte)
    print(f"{name:<16}{Ftr.shape[1]:>6}{r['train_acc'] * 100:>8.2f}%{r['test_acc'] * 100:>8.2f}%", flush=True)
