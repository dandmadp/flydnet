"""
fd.genetics.mosaic 예: 세포 유형 드롭아웃으로 학습하면 후각 수용체(사구체) 하나를 잃어도 버티는가

  python examples/mosaic_odor.py              # 약 6분 (RTX 5070)

과제: DoOR 2.0 실제 냄새 K개 (잡음 섞인 시료) 구분
모델: 사구체 → PN → KC (스파이킹 ConnectomeLayer, 실제 배선) → 선형 리드아웃 (층을 통과해 학습)
조건 (seed마다 짝지음):
  none    드롭아웃 없음
  neuron  PN 뉴런마다 따로 끔 (보통 드롭아웃, p 같음)
  type    PN 세포 유형(사구체)마다 통째로 끔 (mosaic)
평가: 깨끗한 정확도 + 냄새 구분에 가장 중요한 사구체 10개를 하나씩 없앴을 때의 정확도 (수용체 결손 모사)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd
from flydnet.ganglion import backend as B

ap = argparse.ArgumentParser()
ap.add_argument("--odors", type=int, default=20)
ap.add_argument("--samples", type=int, default=30, help="냄새마다 학습·평가 시료 수 (각각)")
ap.add_argument("--noise", type=float, default=0.7)
ap.add_argument("--p", type=float, default=0.2, help="드롭아웃 확률")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--lesions", type=int, default=10, help="하나씩 없앨 사구체 수")
args = ap.parse_args()
B.limit_gpu_memory(0.75)
t0 = time.time()

mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)
G = fd.genetics
door = fd.door_odors(enc.glomeruli)
pick = np.argsort(door["X"].sum(1))[::-1][:args.odors]
proto = door["X"][pick] / door["X"][pick].max()
pn = G.driver(mb, group="PN")
types = mb.meta.cell_type.astype(str).to_numpy()
# 냄새 구분에 중요한 사구체 = 고른 냄새들 사이에서 반응이 가장 많이 달라지는 사구체
informative = [enc.glomeruli[j] for j in np.argsort(proto.std(0))[::-1][:args.lesions]]
lesion_lines = {g: G.Line(mb, [i for i in pn.idx if types[i].split("_")[0] == g], g) for g in informative}
print(f"냄새 {args.odors}개, 없앨 사구체 {informative}")


def make(n, rng):
    y = np.repeat(np.arange(args.odors), n)
    X = proto[y] * rng.lognormal(0, args.noise, (len(y), proto.shape[1])) + rng.uniform(0, 0.15, (len(y), proto.shape[1]))
    return np.clip(X, 0, 1).astype(np.float32), y


def run(seed, cond):
    rng = np.random.default_rng(seed)
    Xtr, ytr = make(args.samples, rng)
    Xte, yte = make(args.samples, rng)
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=50, dt=0.5, gains={"PN>KC": 3.0}, input_mode="regular")
    F = fd.extract(layer, Xtr, enc, batch=200)                       # 정규화 기준 (드롭아웃 없이)
    mu, sd = B.to(F.mean(0), layer.device), max(float((F - F.mean(0)).std()), 1e-6)
    readout = fd.Projection(F.shape[1], args.odors, seed=seed)
    rule = fd.AdaptivePlasticity(readout.synapses(), rate=1e-2, decay=1e-4)
    if cond != "none":
        G.mosaic(layer, p=args.p, by="cell_type" if cond == "type" else "neuron", within=pn)
    order = np.random.default_rng(seed + 100)
    step = 0
    for ep in range(args.epochs):
        rule.rate = 1e-2 * 0.5 * (1 + np.cos(np.pi * ep / args.epochs))
        perm = order.permutation(len(Xtr))
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            Fb = (layer(enc(Xtr[j]), seed=seed * 100000 + step) - mu) * (1.0 / sd)
            loss = fd.surprise(readout(Fb), ytr[j])
            rule.clear(); loss.retrograde(); rule.step()
            step += 1
    G.clear(layer)

    def acc(X, y):
        with fd.quiescent():
            Fe = (layer(enc(X), seed=7) - mu) * (1.0 / sd)
            return float((readout(Fe).numpy().argmax(1) == y).mean())
    clean = acc(Xte, yte)
    les = []
    for g, line in lesion_lines.items():
        with G.silence(layer, line):
            les.append(acc(Xte, yte))
    return clean, float(np.mean(les)), float(np.min(les))


conds = ["none", "neuron", "type"]
R = {c: [] for c in conds}
for s in range(args.seeds):
    for c in conds:
        R[c].append(run(s, c))
    print(f"seed {s}: " + "  ".join(f"{c} 깨끗 {R[c][-1][0]:.3f} 결손 평균 {R[c][-1][1]:.3f}" for c in conds)
          + f"  ({time.time() - t0:.0f}s)", flush=True)

R = {c: np.array(v) for c, v in R.items()}
print(f"\n정확도 (seed {args.seeds}개 평균, 찍기 {1 / args.odors:.3f})")
print(f"{'조건':<8}{'깨끗':>8}{'사구체 1개 결손 (평균)':>24}{'(최악)':>9}")
for c in conds:
    print(f"{c:<8}{R[c][:, 0].mean():>8.3f}{R[c][:, 1].mean():>24.3f}{R[c][:, 2].mean():>9.3f}")
print("\n짝지은 차이 (부호 뒤집기 검정)")
for a, b in [("type", "none"), ("type", "neuron"), ("neuron", "none")]:
    for k, name in [(0, "깨끗"), (1, "결손 평균")]:
        d = R[a][:, k] - R[b][:, k]
        print(f"  {a} - {b:<7} {name:<6} {d.mean():+.3f}  p={fd.sign_flip_p(d):.3f}  ({(d > 0).sum()}/{len(d)} 우세)")
print(f"\n총 {time.time() - t0:.0f}초")
