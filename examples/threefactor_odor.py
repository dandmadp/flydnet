"""
fd.ThreeFactor 예: 역전파 없이 버섯체 연결을 학습할 수 있는가, 실제 피드백 경로(MBON → DAN → KC)가 학습 신호로 쓸모 있는가

  python examples/threefactor_odor.py              # 약 10분 (GPU)

과제: DoOR 2.0 실제 냄새 K개 (잡음 섞인 시료) 구분
모델: 사구체 → PN → KC → MBON (스파이킹, 실제 배선 + DAN) → 선형 리드아웃 (리드아웃은 모든 조건에서 경사 하강)
학습하는 연결: PN > KC (숨은 연결: 오차가 피드백으로 와야 함), KC > MBON (출력으로 들어오는 연결)
조건 (seed마다 짝지음):
  readout     커넥톰 고정, 리드아웃만
  bptt        시간 역전파 (기준)
  none        3요소 규칙, 피드백 없음 → KC > MBON만 학습
  random      3요소 규칙, 무작위 피드백 (표준 e-prop)
  connectome  3요소 규칙, 실제 연결을 따라 MBON에서 퍼지는 오차 (MBON → KC, MBON → DAN → KC 등 2단계)
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
ap.add_argument("--odors", type=int, default=12)
ap.add_argument("--samples", type=int, default=24, help="냄새마다 학습·평가 시료 수 (각각)")
ap.add_argument("--noise", type=float, default=0.8)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--rate", type=float, default=0.02, help="커넥톰 연결 학습률 (리드아웃은 0.01)")
ap.add_argument("--conds", default="readout,bptt,none,random,connectome")
args = ap.parse_args()
B.limit_gpu_memory(0.75)
t0 = time.time()

mb = fd.Circuit.from_flywire(dict(fd.MUSHROOM_BODY, DAN=("cell_class", "DAN")))
enc = fd.GlomerularEncoder(mb)
door = fd.door_odors(enc.glomeruli)
pick = np.argsort(door["X"].sum(1))[::-1][:args.odors]
proto = door["X"][pick] / door["X"][pick].max()
print(mb)


def make(n, rng):
    y = np.repeat(np.arange(args.odors), n)
    X = proto[y] * rng.lognormal(0, args.noise, (len(y), proto.shape[1])) + rng.uniform(0, 0.15, (len(y), proto.shape[1]))
    return np.clip(X, 0, 1).astype(np.float32), y


def run(seed, cond):
    rng = np.random.default_rng(seed)
    Xtr, ytr = make(args.samples, rng)
    Xte, yte = make(args.samples, rng)
    layer = fd.ConnectomeLayer(mb, "PN", "MBON", t_ms=50, dt=0.5, gains={"PN>KC": 3.0, "KC>MBON": 3.0},
                               input_mode="regular", trainable=["PN>KC", "KC>MBON"])
    with fd.quiescent():
        F = layer(enc(Xtr), seed=0).numpy()
    mu, sd = B.to(F.mean(0), layer.device), float(F.std() + 1e-6)
    readout = fd.Projection(F.shape[1], args.odors, seed=seed)
    r_rule = fd.AdaptivePlasticity(readout.synapses(), rate=1e-2, decay=1e-4)
    c_rule = fd.AdaptivePlasticity(layer.synapses(), rate=args.rate, clip=1.0) if cond != "readout" else None
    tf = fd.ThreeFactor(layer, feedback=cond, hops=2, seed=seed) if cond in ("none", "random", "connectome") else None
    order, step, t_step = np.random.default_rng(seed + 100), 0, []
    for ep in range(args.epochs):
        cos = 0.5 * (1 + np.cos(np.pi * ep / args.epochs))
        r_rule.rate = 1e-2 * cos
        if c_rule:
            c_rule.rate = args.rate * cos
        perm = order.permutation(len(Xtr))
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            ts = time.perf_counter()
            s = seed * 100000 + step
            if tf is not None:
                out = tf(enc(Xtr[j]), seed=s)
            elif cond == "bptt":
                out = layer(enc(Xtr[j]), seed=s)
            else:
                with fd.quiescent():
                    out = layer(enc(Xtr[j]), seed=s)
            loss = fd.surprise(readout((out - mu) * (1.0 / sd)), ytr[j])
            r_rule.clear(); layer.clear_retro(); loss.retrograde()
            if tf is not None:
                tf.assign(out)
            r_rule.step()
            if c_rule:
                c_rule.step()
            t_step.append(time.perf_counter() - ts)
            step += 1
    with fd.quiescent():
        pred = readout((layer(enc(Xte), seed=7) - mu) * (1.0 / sd)).numpy().argmax(1)
    return float((pred == yte).mean()), float(np.median(t_step))


conds = args.conds.split(",")
R = {c: [] for c in conds}
T = {c: [] for c in conds}
for s in range(args.seeds):
    for c in conds:
        a, t = run(s, c)
        R[c].append(a); T[c].append(t)
    print(f"seed {s}: " + "  ".join(f"{c} {R[c][-1]:.3f}" for c in conds) + f"  ({time.time() - t0:.0f}s)", flush=True)

print(f"\n평가 정확도 (seed {args.seeds}개, 찍기 {1 / args.odors:.3f})")
for c in conds:
    print(f"  {c:<11} {np.mean(R[c]):.3f} ± {np.std(R[c], ddof=1) / np.sqrt(len(R[c])):.3f}   학습 1스텝 {np.mean(T[c]) * 1000:.0f} ms")
print("\n짝지은 차이 (부호 뒤집기 검정)")
pairs = [("bptt", "readout"), ("random", "readout"), ("connectome", "readout"), ("connectome", "random"),
         ("connectome", "none"), ("random", "none"), ("bptt", "connectome")]
for a, b in pairs:
    if a in R and b in R:
        d = np.array(R[a]) - np.array(R[b])
        print(f"  {a:>10} - {b:<10} {d.mean():+.3f}  p={fd.sign_flip_p(d):.3f}  ({(d > 0).sum()}/{len(d)} 우세)")
print(f"\n총 {time.time() - t0:.0f}초")
