"""
추가 연결 (구조적 가소성) 예: 실제 버섯체 배선은 그대로 두고, 학습하며 PN→KC 시냅스를 덧붙이고 없앰

  python lab/growth_odor.py              # 약 20분 (GPU)

과제: DoOR 2.0 실제 냄새 (어렵게: 냄새 24개, 잡음 큼)
모델: 사구체 → PN → KC (스파이킹 ConnectomeLayer, 실제 배선 + 추가 연결) → 정규화 → 선형
조건 (seed마다 짝지음):
  fixed     실제 배선만 (추가 연결 없음)
  random    에폭마다 약한 추가 연결 20%를 없애고, 허용된 그룹 쌍(PN>KC) 안에서 무작위로 다시 채움
  coactive  같은 일정, 새 연결은 함께 많이 발화한 PN·KC 쌍에 (헤브 규칙)
  gradient  같은 일정, 새 연결은 세기 0으로 넣었을 때 손실이 가장 줄 자리에 (RigL과 같은 생각)
  contrastive 같은 일정, 정답 없이: 시료 특징 30%를 다른 시료 값으로 바꾼 보기 두 벌이 서로 짝을 찾도록 하는 손실
            (InfoNCE + SCARF 보기 만들기)의 기울기로
추가 연결은 실제 PN>KC 연결 하나 크기(시냅스 수 중앙값 10개)로 생겨 학습으로 커지거나 작아짐 (부호는 PN의 실제 부호 = 흥분).
처음 연결을 채운 뒤 KC 평균 발화율을 다시 5 Hz로 보정 - 조건끼리 KC 활동량이 같아서, 차이는 연결 자리에서만 옴
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))  # 예제 공용 도우미 (_mnist 등)
import flydnet as fd
from flydnet import lab
from flydnet.lab import corrupt

ap = argparse.ArgumentParser()
ap.add_argument("--odors", type=int, default=24)
ap.add_argument("--noise", type=float, default=1.2)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--budget", type=int, default=4000, help="추가 연결 상한 (실제 PN>KC 연결 13,485개)")
ap.add_argument("--turnover", type=float, default=0.2, help="에폭마다 없애고 다시 채우는 비율")
ap.add_argument("--seeds", type=int, default=6, help="6 이상이어야 부호 뒤집기 검정 p < 0.05가 가능")
ap.add_argument("--conds", default="fixed,random,coactive,gradient,contrastive")
args = ap.parse_args()
fd.ganglion.limit_gpu_memory(0.75)
t0 = time.time()

mb = fd.flywire()
enc = fd.Glomeruli(mb)


def run(seed, cond):
    Xtr, ytr, Xte, yte = fd.door_task(n_odors=args.odors, noise=args.noise, seed=seed)
    growth = None if cond == "fixed" else dict(allow=["PN>KC"], budget=args.budget)
    layer = fd.Connectome(mb, "PN", "KC", t_ms=50, dt=0.5, input_mode="regular", trainable=["PN>KC"])
    if growth is not None:
        lab.Growth(layer, **growth)                                       # 추가 연결 부품 (layer.growth)
    layer.calibrate(enc(Xtr[::3]), {"KC": 5})
    head = fd.Pathway(fd.Homeostasis(), fd.Projection(layer.n_out, args.odors, seed=seed))
    model = fd.Pathway(enc, layer, head)
    rule = fd.Adaptive(model.named_synapses(), rate=3e-3, clip=1.0)
    rng = np.random.default_rng(seed)
    rates = enc(Xtr[::3]).numpy()

    def loss_on(Xb, yb):
        return lambda L: fd.surprise(head(L(enc(Xb), seed=0)), yb)

    if growth is not None:                                                # 처음에 예산을 채움
        if cond == "coactive":
            layer.growth.grow(args.budget, rule="coactive", rates=rates)
        elif cond == "contrastive":
            lab.grow_contrastive(layer, args.budget, inputs=Xtr, encoder=enc, augment=corrupt(0.3), seed=seed)
        elif cond == "gradient":
            j = rng.permutation(len(Xtr))[:64]
            layer.growth.grow(args.budget, rule="gradient", loss=loss_on(Xtr[j], ytr[j]), seed=seed)
        else:
            layer.growth.grow(args.budget, seed=seed)
        layer.calibrate(enc(Xtr[::3]), {"KC": 5})                         # 추가 연결이 더한 입력만큼 다시 맞춤
    step = 0
    for ep in range(args.epochs):
        rule.rate = 3e-3 * 0.5 * (1 + np.cos(np.pi * ep / args.epochs))
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            loss = fd.surprise(model(Xtr[j], seed=seed * 10 ** 6 + step), ytr[j])
            rule.clear(); loss.retrograde(); rule.step()
            step += 1
        if growth is not None and ep < args.epochs - 1:                   # 구조적 가소성: 약한 것을 없애고 다시 채움
            layer.growth.prune(args.turnover)
            room = args.budget - layer.growth.n
            if cond == "coactive":
                layer.growth.grow(room, rule="coactive", rates=rates)
            elif cond == "contrastive":
                lab.grow_contrastive(layer, room, inputs=Xtr, encoder=enc, augment=corrupt(0.3), seed=seed * 100 + ep)
            elif cond == "gradient":
                j = rng.permutation(len(Xtr))[:64]
                layer.growth.grow(room, rule="gradient", loss=loss_on(Xtr[j], ytr[j]), seed=seed * 100 + ep)
            else:
                layer.growth.grow(room, seed=seed * 100 + ep)
    acc = fd.evaluate(model, Xte, yte)
    extra = layer.growth.extra_edges() if growth is not None else None
    return acc, (0 if extra is None else len(extra)), (float(extra.synapses.median()) if extra is not None and len(extra) else 0.0)


conds = args.conds.split(",")
R = {c: [] for c in conds}
for s in range(args.seeds):
    for c in conds:
        acc, n, med = run(s, c)
        R[c].append(acc)
        print(f"seed {s} {c:<9} 정확도 {acc:.3f}   추가 연결 {n:,}개 (세기 중앙값 시냅스 {med:.2f}개)  [{time.time() - t0:.0f}s]",
              flush=True)

print(f"\n평가 정확도 (냄새 {args.odors}개, 찍기 {1 / args.odors:.3f}, seed {args.seeds}개)")
for c in conds:
    print(f"  {c:<9} {np.mean(R[c]):.3f}")
if "fixed" in R:
    print("\n실제 배선만 대비 짝지은 차이 (부호 뒤집기 검정)")
    for c in conds:
        if c != "fixed":
            d = np.array(R[c]) - np.array(R["fixed"])
            print(f"  {c:<9} {d.mean():+.3f}  p={fd.sign_flip_p(d):.3f}  ({(d > 0).sum()}/{len(d)} 우세)")
print(f"\n총 {time.time() - t0:.0f}초")
