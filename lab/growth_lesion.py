"""
손상 회복 (구조적 가소성): 버섯체 PN→KC 연결의 80%를 끊은 뒤, 추가 연결이 잃은 정확도를 되찾는지 -
정답 없이 되찾는지, 정답으로 고른 연결이 다른 과제에도 통하는지까지

  python lab/growth_lesion.py              # 약 17분 (GPU)

과제: DoOR 2.0 실제 냄새 48개를 번갈아 두 묶음으로 - B(24개)로 학습·평가, A(24개)는 '다른 과제'
모델: 사구체 → PN → KC (스파이킹, 실제 배선) → 정규화 → 선형
조건 (seed마다 같은 손상을 짝지음, 추가 연결은 에폭마다 약한 20% 없애고 다시 채움):
  intact            손상 없음 (회복 목표)
  lesion            PN→KC 연결 80%를 무작위로 끊음 (남은 연결은 보정으로 세짐 - 시냅스 스케일링)
  random b          손상 + 추가 연결 b개, 허용 쌍 PN>KC 안에서 무작위
  homeostatic b     정답 없이: 대표 입력에서 KC 평균보다 덜 발화하는 KC가 모자란 만큼 새 입력을 받음 (보내는 PN은 무작위)
  gradient b        과제 B의 정답으로: 손실이 가장 줄 자리 (RigL과 같은 생각) - 평가 과제에 맞춘 상한
  gradient-A b      다른 과제 A의 정답으로 고른 자리 (A용 선형 층을 그때마다 1에폭 학습해 기울기) → 과제 B로 평가.
                    B에서도 회복되면 '회로가 좋아진 것', 아니면 'A에 맞춘 것'
  contrastive-corrupt b  같은 대조 학습, 보기 만들기만 SCARF 방식 (특징 30%를 다른 시료의 값으로) - 이 데이터를 만든
                    잡음(곱 로그 정규)과 같은 방식을 쓰면 '정답을 아는' 셈이 되므로, 무관한 방식으로도 되는지 (순환 확인)
  규칙-nocap b      받는 뉴런 상한 없이 (per_neuron=None - 기본 "auto" 전과 같음)
  intact+규칙 b     손상 없는 회로에 같은 추가 연결 (대조군 - 회복인지, 어느 회로든 좋아지는 것인지)
  contrastive b     정답 없이 기울기로: 같은 냄새 시료에 잡음을 두 번 따로 넣어 KC 반응 두 벌을 만들고, 각 시료가 자기 짝을
                    다른 시료들 사이에서 찾도록 (대조 학습, InfoNCE - SimCLR과 같은 목표). 라벨을 전혀 쓰지 않음.
                    flydnet.lab.grow_contrastive(layer, n, inputs=X, encoder=enc)
모든 조건은 연결을 정한 뒤 KC 평균 5 Hz로 보정 (조건끼리 KC 활동량이 같음 → 차이는 연결에서만).
80% 손상 회로는 반응이 가팔라 보정이 진동하기 쉬움 - calibrate가 진동하는 그룹의 보폭을 줄여 맞춤 (0.1.18)
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
ap.add_argument("--cut", type=float, default=0.8, help="끊을 PN→KC 연결 비율")
ap.add_argument("--conds", default="random 1350,random 5400,homeostatic 1350,homeostatic 5400,"
                                  "gradient 1350,gradient-A 1350,gradient-A 5400,contrastive 1350,contrastive 5400,"
                                  "contrastive-corrupt 1350,contrastive-corrupt 5400,intact+contrastive 5400",
                help="'규칙 개수'를 쉼표로 (끊은 연결 10,788개의 1/8 = 1350, 1/2 = 5400)")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--seeds", type=int, default=6, help="6 이상이어야 부호 뒤집기 검정 p < 0.05가 가능")
args = ap.parse_args()
fd.ganglion.limit_gpu_memory(0.75)
t0 = time.time()

mb = fd.flywire()
enc = fd.Glomeruli(mb)
g = mb.group_of()
pk = np.nonzero((g[mb.pre] == "PN") & (g[mb.post] == "KC"))[0]
CAL = dict(iters=60)


def lesioned(seed):
    cut = np.random.default_rng(1000 + seed).choice(pk, int(round(args.cut * len(pk))), replace=False)
    keep = np.ones(len(mb.pre), bool)
    keep[cut] = False
    return fd.Circuit(mb.root_ids, mb.groups, mb.pre[keep], mb.post[keep], mb.weight[keep], meta=mb.meta), len(cut)


def split(seed):
    """냄새 48개 → 번갈아 A(짝수 번)·B(홀수 번), 라벨은 0~23으로"""
    Xtr, ytr, Xte, yte = fd.door_task(n_odors=48, noise=1.2, seed=seed)
    out = {}
    for name, r in (("A", 0), ("B", 1)):
        i, j = ytr % 2 == r, yte % 2 == r
        out[name] = (Xtr[i], ytr[i] // 2, Xte[j], yte[j] // 2)
    return out


def train(model, rule, X, y, rng, epochs, seed, step0=0, cosine=True):
    step = step0
    for ep in range(epochs):
        if cosine:
            rule.rate = 3e-3 * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            loss = fd.surprise(model(X[j], seed=seed * 10 ** 6 + step), y[j])
            rule.clear(); loss.retrograde(); rule.step()
            step += 1
    return step


def run(seed, data, circuit, rule=None, budget=0):
    Xtr, ytr, Xte, yte = data["B"]
    XA, yA = data["A"][0], data["A"][1]
    growth = dict(allow=["PN>KC"], budget=budget) if rule else None
    if rule and rule.endswith("-nocap"):                                  # 받는 뉴런 상한 없이 (0.1.18 기본 "auto" 전과 같음)
        growth["per_neuron"] = None
        rule = rule.removesuffix("-nocap")
    layer = fd.Connectome(circuit, "PN", "KC", t_ms=50, dt=0.5, input_mode="regular", trainable=["PN>KC"])
    if growth is not None:
        lab.Growth(layer, **growth)                                       # 추가 연결 부품 (layer.growth)
    R = enc(Xtr[::3])
    layer.calibrate(R, {"KC": 5}, **CAL)
    head = fd.Pathway(fd.Homeostasis(), fd.Projection(layer.n_out, 24, seed=seed))
    model = fd.Pathway(enc, layer, head)
    rng = np.random.default_rng(seed)
    headA = fd.Pathway(fd.Homeostasis(), fd.Projection(layer.n_out, 24, seed=seed + 7))
    ruleA = fd.Adaptive(headA.named_synapses(), rate=3e-3, clip=1.0)

    def grow(n, s):
        if rule == "gradient":
            j = rng.permutation(len(Xtr))[:64]
            layer.growth.grow(n, rule="gradient", loss=lambda L: fd.surprise(head(L(enc(Xtr[j]), seed=0)), ytr[j]), seed=s)
        elif rule == "gradient-A":                                           # A용 선형 층을 지금 회로에 맞춰 1에폭 → A의 기울기
            train(fd.Pathway(enc, layer, headA), ruleA, XA, yA, rng, 1, seed + 99, cosine=False)
            j = rng.permutation(len(XA))[:64]
            layer.growth.grow(n, rule="gradient", loss=lambda L: fd.surprise(headA(L(enc(XA[j]), seed=0)), yA[j]), seed=s)
        elif rule == "homeostatic":
            layer.growth.grow(n, rule="homeostatic", rates=R, seed=s)
        elif rule.startswith("contrastive"):                                # 라벨 없이: 과제 B 입력만
            aug = corrupt(0.3) if rule == "contrastive-corrupt" else 0.3
            lab.grow_contrastive(layer, n, inputs=Xtr, encoder=enc, augment=aug, seed=s)
        else:
            layer.growth.grow(n, seed=s)

    if rule:
        grow(budget, seed)
        layer.calibrate(R, {"KC": 5}, **CAL)                             # 추가 연결이 더한 입력만큼 다시 맞춤
    opt = fd.Adaptive(model.named_synapses(), rate=3e-3, clip=1.0)
    step = 0
    for ep in range(args.epochs):
        opt.rate = 3e-3 * 0.5 * (1 + np.cos(np.pi * ep / args.epochs))
        step = train(model, opt, Xtr, ytr, rng, 1, seed, step, cosine=False)
        if rule and ep < args.epochs - 1:                                  # 약한 것을 없애고 다시 채움
            layer.growth.prune(0.2)
            grow(budget - layer.growth.n, seed * 100 + ep)
    return fd.evaluate(model, Xte, yte)


conds = ["intact", "lesion"] + [c.strip() for c in args.conds.split(",")]
Racc = {c: [] for c in conds}
for s in range(args.seeds):
    lc, n_cut = lesioned(s)
    data = split(s)
    for c in conds:
        if c == "intact":
            acc = run(s, data, mb)
        elif c == "lesion":
            acc = run(s, data, lc)
        else:                                                            # "intact+규칙 b": 손상 없는 회로에 추가 연결 (대조군)
            rule, b = c.split()
            base = mb if rule.startswith("intact+") else lc
            acc = run(s, data, base, rule.removeprefix("intact+"), int(b))
        Racc[c].append(acc)
        print(f"seed {s} {c:<18} 정확도 {acc:.3f}  [{time.time() - t0:.0f}s]", flush=True)

print(f"\nPN→KC 연결 {len(pk):,}개 중 {n_cut:,}개({args.cut:.0%}) 끊음, 과제 B 냄새 24개 (찍기 0.042), seed {args.seeds}개")
lo, hi = np.mean(Racc["lesion"]), np.mean(Racc["intact"])
print(f"{'조건':<20}{'정확도':>7}{'회복':>7}   손상만 대비              손상 없음 대비")
for c in conds:
    m = np.mean(Racc[c])
    rec = (m - lo) / (hi - lo) if hi > lo else float("nan")
    line = f"{c:<20}{m:>7.3f}{rec:>7.0%}"
    for ref in ("lesion", "intact"):
        if c in ("lesion", "intact"):
            line += " " * 26
            continue
        d = np.array(Racc[c]) - np.array(Racc[ref])
        line += f"   {d.mean():+.3f} p={fd.sign_flip_p(d):.3f} ({(d > 0).sum()}/{len(d)})"
    print(line)
print("회복 = (조건 - 손상) / (손상 없음 - 손상)")
print(f"\n총 {time.time() - t0:.0f}초")
