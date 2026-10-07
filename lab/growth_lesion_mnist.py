"""
손상 회복 재현 (MNIST): growth_lesion.py의 냄새 결과가 다른 종류의 데이터에서도 나오는지 - 버섯체 PN→KC 연결 80%를 끊고
정답 없는 대조 학습(flydnet.lab.grow_contrastive)으로 추가 연결을 고름

  python lab/growth_lesion_mnist.py        # 약 27분 (GPU)

과제: MNIST 손글씨 숫자 (seed마다 학습 3,000 / 평가 1,000장을 다르게 뽑음)
모델: 픽셀 → 무작위 투영으로 PN 344개 (RateEncoder) → KC (스파이킹, 실제 배선) → 정규화 → 선형
조건 (seed마다 같은 손상을 짝지음, 추가 연결은 에폭마다 약한 20% 없애고 다시 채움):
  intact                  손상 없음
  lesion                  PN→KC 연결 80%를 무작위로 끊음
  random b                손상 + 추가 연결 b개 (허용 쌍 PN>KC 안에서 무작위)
  contrastive b           손상 + 정답 없이: 같은 숫자 그림을 두 번 따로 흔들어 (최대 2픽셀 이동 + 밝기 잡음) KC 반응 두 벌,
                          각 그림이 자기 짝을 찾도록 (InfoNCE) 하는 기울기로 고름. 라벨을 전혀 쓰지 않음
  intact+contrastive b    손상 없는 회로에 같은 규칙 (대조군)
원인 찾기 (대조 학습이 무작위보다 못한 이유 - 한 번에 하나만 바꿈):
  contrastive-noise b     보기 만들기에서 이동을 빼고 밝기 잡음만 (곱 로그 정규 0.3, 냄새와 같은 방식)
  contrastive-t0.5 b      온도 tau 0.5 (기본 0.1) - 다른 시료를 덜 세게 밀어냄
  contrastive-s256 b      시료 256개 (기본 64)
  contrastive-onehot b    진단용 (방법 아님): 숫자마다 한 장씩 10장 - 같은 숫자끼리 서로 밀어내는 일이 없게 (고를 때만 라벨)
  contrastive-cap3 b      받는 KC마다 추가 연결 3개까지 (per_neuron=3) - 소수 KC에 몰리지 않게
  random-cap3 b           무작위 + 같은 상한 (비교)
  contrastive-corrupt b   보기 만들기를 SCARF로 (픽셀 30%를 다른 그림의 같은 픽셀로)
  *-nocap b               받는 뉴런 상한 없이 (per_neuron=None - 0.1.18 기본 "auto" 전과 같음)
모든 조건은 연결을 정한 뒤 KC 평균 5 Hz로 보정
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
from _mnist import mnist

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
ap.add_argument("--cut", type=float, default=0.8)
ap.add_argument("--conds", default="random 1350,random 5400,contrastive 1350,contrastive 5400,contrastive-corrupt 5400,"
                                  "contrastive-nocap 5400,intact+contrastive 5400")
ap.add_argument("--n-train", type=int, default=3000)
ap.add_argument("--n-test", type=int, default=1000)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--seeds", type=int, default=6, help="6 이상이어야 부호 뒤집기 검정 p < 0.05가 가능")
args = ap.parse_args()
fd.ganglion.limit_gpu_memory(0.75)
t0 = time.time()

Xall, yall, Xte_all, yte_all = mnist(args.data)
mb = fd.flywire()
enc = fd.RateEncoder(784, len(mb.groups["PN"]), max_rate=100.0)       # 픽셀 0~1 → PN Hz (무작위 투영, seed 0)
g = mb.group_of()
pk = np.nonzero((g[mb.pre] == "PN") & (g[mb.post] == "KC"))[0]
CAL = dict(iters=60)


def shift_noise(X, rng, px=2, sd=0.2):
    """그림마다 가로·세로 최대 px픽셀 이동 + 곱 로그 정규 밝기 잡음 ([0, 1]로 자름) - 같은 숫자의 다른 '보기'"""
    img = X.reshape(-1, 28, 28)
    out = np.empty_like(img)
    dx, dy = rng.integers(-px, px + 1, len(img)), rng.integers(-px, px + 1, len(img))
    for i in range(len(img)):
        out[i] = np.roll(np.roll(img[i], dy[i], 0), dx[i], 1)
    out = out.reshape(len(X), -1) * np.exp(sd * rng.standard_normal((len(X), 784)))
    return np.clip(out, 0, 1).astype(np.float32)


def lesioned(seed):
    cut = np.random.default_rng(1000 + seed).choice(pk, int(round(args.cut * len(pk))), replace=False)
    keep = np.ones(len(mb.pre), bool)
    keep[cut] = False
    return fd.Circuit(mb.root_ids, mb.groups, mb.pre[keep], mb.post[keep], mb.weight[keep], meta=mb.meta), len(cut)


def run(seed, circuit, rule=None, budget=0):
    r = np.random.default_rng(seed)
    i, j = r.permutation(len(Xall))[:args.n_train], r.permutation(len(Xte_all))[:args.n_test]
    Xtr, ytr, Xte, yte = Xall[i], yall[i], Xte_all[j], yte_all[j]
    growth = dict(allow=["PN>KC"], budget=budget) if rule else None
    if rule and rule.endswith("-cap3"):                                    # 받는 뉴런마다 추가 연결 상한
        growth["per_neuron"] = 3
        rule = rule.removesuffix("-cap3")
    if rule and rule.endswith("-nocap"):
        growth["per_neuron"] = None
        rule = rule.removesuffix("-nocap")
    layer = fd.Connectome(circuit, "PN", "KC", t_ms=50, dt=0.5, input_mode="regular", trainable=["PN>KC"])
    if growth is not None:
        lab.Growth(layer, **growth)                                       # 추가 연결 부품 (layer.growth)
    R = enc(Xtr[::10])
    layer.calibrate(R, {"KC": 5}, **CAL)
    model = fd.Pathway(enc, layer, fd.Homeostasis(), fd.Projection(layer.n_out, 10, seed=seed))
    rng = np.random.default_rng(seed)

    def grow(n, s):
        if rule.startswith("contrastive"):
            kw = dict(inputs=Xtr, encoder=enc, augment=shift_noise, seed=s)
            variant = rule.partition("-")[2]
            if variant == "noise":
                kw["augment"] = 0.3
            elif variant.startswith("t"):
                kw["tau"] = float(variant[1:])
            elif variant.startswith("s"):
                kw["samples"] = int(variant[1:])
            elif variant == "corrupt":                                       # SCARF: 픽셀 30%를 다른 그림의 같은 픽셀로
                kw["augment"] = corrupt(0.3)
            elif variant == "onehot":
                pick = np.array([rng.choice(np.nonzero(ytr == d)[0]) for d in range(10)])
                kw.update(inputs=Xtr[pick], samples=10)
            lab.grow_contrastive(layer, n, **kw)
        else:
            layer.growth.grow(n, seed=s)

    if rule:
        grow(budget, seed)
        layer.calibrate(R, {"KC": 5}, **CAL)
    opt = fd.Adaptive(model.named_synapses(), rate=3e-3, clip=1.0)
    step = 0
    for ep in range(args.epochs):
        opt.rate = 3e-3 * 0.5 * (1 + np.cos(np.pi * ep / args.epochs))
        perm = rng.permutation(len(Xtr))
        for k in range(0, len(perm), 32):
            b = perm[k:k + 32]
            loss = fd.surprise(model(Xtr[b], seed=seed * 10 ** 6 + step), ytr[b])
            opt.clear(); loss.retrograde(); opt.step()
            step += 1
        if rule and ep < args.epochs - 1:
            layer.growth.prune(0.2)
            grow(budget - layer.growth.n, seed * 100 + ep)
    return fd.evaluate(model, Xte, yte)


conds = ["intact", "lesion"] + [c.strip() for c in args.conds.split(",")]
Racc = {c: [] for c in conds}
for s in range(args.seeds):
    lc, n_cut = lesioned(s)
    for c in conds:
        if c == "intact":
            acc = run(s, mb)
        elif c == "lesion":
            acc = run(s, lc)
        else:
            rule, b = c.split()
            acc = run(s, mb if rule.startswith("intact+") else lc, rule.removeprefix("intact+"), int(b))
        Racc[c].append(acc)
        print(f"seed {s} {c:<24} 정확도 {acc:.3f}  [{time.time() - t0:.0f}s]", flush=True)

print(f"\nPN→KC 연결 {len(pk):,}개 중 {n_cut:,}개({args.cut:.0%}) 끊음, MNIST 학습 {args.n_train} / 평가 {args.n_test}, "
      f"seed {args.seeds}개")
lo, hi = np.mean(Racc["lesion"]), np.mean(Racc["intact"])
print(f"{'조건':<26}{'정확도':>7}{'회복':>7}   손상만 대비              손상 없음 대비")
for c in conds:
    m = np.mean(Racc[c])
    rec = (m - lo) / (hi - lo) if hi > lo else float("nan")
    line = f"{c:<26}{m:>7.3f}{rec:>7.0%}"
    for ref in ("lesion", "intact"):
        if c in ("lesion", "intact"):
            continue
        d = np.array(Racc[c]) - np.array(Racc[ref])
        line += f"   {d.mean():+.3f} p={fd.sign_flip_p(d):.3f} ({(d > 0).sum()}/{len(d)})"
    print(line)
print("회복 = (조건 - 손상) / (손상 없음 - 손상)")
print(f"\n총 {time.time() - t0:.0f}초")
