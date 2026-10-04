"""
어떤 그래프든 회로로: 초파리가 아닌 커넥톰과 합성 그래프에 같은 도구 쓰기

  python -m flydnet download worm
  python examples/any_graph.py                 # 약 8분 (GPU; --part 1 은 약 1분)

1) 예쁜꼬마선충 (Cook et al. 2019): 감각 뉴런 24개(amphid)에 들어온 패턴을 체벽 근육 95개의 활동으로 구분.
   fd.compare로 실제 배선 대 대조군 - 초파리용 도구가 다른 동물에서 그대로 동작
2) 그래프 종류 비교 (AI 연구 예): 노드 수·평균 연결 수를 맞춘 무작위망 / 작은 세상망 / 척도 없는 망 / 블록 구조를
   같은 과제의 스파이킹 저장소(reservoir)로 - 어떤 구조가 정보를 잘 전달하는가.
   공정하게: 그래프마다 연결 세기를 골라 전체 평균 발화율을 같게 (약 20 Hz) 맞춘 뒤 비교
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--classes", type=int, default=8)
ap.add_argument("--noise", type=float, default=0.6)
ap.add_argument("--part", default="12", help="돌릴 부분 (1, 2, 12)")
args = ap.parse_args()
fd.ganglion.limit_gpu_memory(0.75)
t0 = time.time()


def task(n_in, seed, n=30):
    """클래스마다 원형 하나 (입력 뉴런별 발화율), 곱 잡음"""
    rng = np.random.default_rng(seed)
    proto = rng.uniform(0, 1, (args.classes, n_in)) ** 2
    def make():
        y = np.repeat(np.arange(args.classes), n)
        X = proto[y] * rng.lognormal(0, args.noise, (len(y), n_in))
        return (np.clip(X, 0, 1.5) * 100).astype(np.float32), y
    return make(), make()


def score(circuit, seed, inp, out, **kw):
    (Xtr, ytr), (Xte, yte) = task(len(circuit.groups[inp]), seed)
    layer = fd.ConnectomeLayer(circuit, inp, out, t_ms=200, **kw)
    f = lambda X: fd.extract(layer, None, X, batch=256, seed=seed)
    return fd.train_linear(f(Xtr), ytr, f(Xte), yte, epochs=60, seed=seed)["test_acc"]


if "1" in args.part:
    worm = fd.Circuit.celegans()
    amphid = [f"{c}{s}" for c in ["ASE", "AWC", "AWA", "ASH", "ADL", "ASK", "ASI", "ASJ", "AWB", "ADF", "ASG", "AFD"]
              for s in "LR"]
    sens = fd.genetics.driver(worm, node=amphid)
    worm = worm.regroup({"sensory": sens.idx, "muscle": worm.groups["body_muscle"]}, rest="neuron_or_other")
    print(worm)
    rep = fd.compare(lambda c, s: score(c, s, "sensory", "muscle", params={"w_syn": 0.825}), worm,
                     controls=["shuffled", "randomized", "shuffled_weights"], seeds=args.seeds,
                     chance=1 / args.classes)
    print("\n1) 예쁜꼬마선충: 감각 뉴런 → 근육 활동으로 패턴 구분\n")
    print(rep)

if "2" in args.part:
    print(f"\n2) 그래프 종류 비교: 노드 400 (입력 20, 출력 40), 평균 연결 수 약 12, 억제 20%  ({time.time() - t0:.0f}s)")
    n, k = 400, 12
    io = {"in": np.arange(20), "out": np.arange(360, 400)}
    families = {
        "무작위 (Erdos-Renyi)": lambda s, w: fd.graphs.erdos_renyi(n, k / (n - 1), weight=w, seed=s, groups=io),
        "작은 세상 (Watts-Strogatz)": lambda s, w: fd.graphs.watts_strogatz(n, k, 0.1, weight=w, seed=s, groups=io),
        "척도 없음 (Barabasi-Albert)": lambda s, w: fd.graphs.barabasi_albert(n, k // 2, weight=w, seed=s, groups=io,
                                                                           reciprocal=1.0),
        "블록 (4 모듈)": lambda s, w: fd.graphs.stochastic_block(
            {"in": 20, "m1": 85, "m2": 85, "m3": 85, "m4": 85, "out": 40},
            {**{(a, a): 0.07 for a in ("m1", "m2", "m3", "m4")},
             **{(a, b): 0.02 for a in ("m1", "m2", "m3", "m4") for b in ("m1", "m2", "m3", "m4") if a != b},
             ("in", "m1"): 0.1, ("in", "m2"): 0.1, ("m3", "out"): 0.1, ("m4", "out"): 0.1},
            weight=w, seed=s),
    }
    probe = np.random.default_rng(99).uniform(0, 150, (16, 20)).astype(np.float32)

    def matched(make, s, target=20.0):
        """전체 평균 발화율이 target Hz에 가장 가까운 연결 세기"""
        best = None
        for w in (10, 15, 20, 25, 30, 35, 40, 50, 60):
            c = make(s, w)
            with fd.quiescent(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*출력.*모두 0")      # 약한 세기도 일부러 시험하는 탐색
                r = float(fd.ConnectomeLayer(c, "in", "out", t_ms=200)(probe, seed=0, return_all=True).numpy().mean())
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (c, r, w)
        return best
    acc = {name: [] for name in families}
    used = {name: [] for name in families}
    for s in range(args.seeds):
        for name, make in families.items():
            c, rate, w = matched(make, s)
            used[name].append((w, rate))
            acc[name].append(score(c, s, "in", "out"))
    base = np.array(acc["무작위 (Erdos-Renyi)"])
    print(f"{'구조':<30}{'연결 수':>8}{'세기':>6}{'발화율':>8}{'정확도':>9}   무작위와 짝지은 차이")
    for name, make in families.items():
        a = np.array(acc[name])
        d = a - base
        w_, r_ = np.mean([u[0] for u in used[name]]), np.mean([u[1] for u in used[name]])
        extra = "" if name.startswith("무작위") else f"   {d.mean():+.3f}  p={fd.sign_flip_p(d):.3f} ({(d > 0).sum()}/{len(d)})"
        print(f"{name:<30}{make(0, 1).n_edges:>8,}{w_:>6.0f}{r_:>7.1f}Hz{a.mean():>9.3f}{extra}")

print(f"\n총 {time.time() - t0:.0f}초")
