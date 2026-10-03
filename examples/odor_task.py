"""
실험 ③: 합성 냄새 과제 — 실제 버섯체 배선이 무작위 배선보다 나은가?

  python examples/odor_task.py
  python examples/odor_task.py --task-seeds 3 --shuffles 3   # 빠르게

냄새 = 사구체 56개 활성 벡터 → GlomerularEncoder(같은 사구체 PN = 같은 발화율) → 버섯체 → KC 발화율
클래스 = 냄새 원형 여러 개의 묶음 → 사구체 공간에선 선형 분리가 어려움 (확장 층이 필요한 과제)

비교 (같은 로지스틱 회귀 리드아웃)
  glomeruli   : 사구체 값 그대로 56차원
  KC real     : 실제 FlyWire PN→KC 배선
  KC shuffled : 뉴런별 입출력 연결 수는 같고 '누가 누구에게'만 무작위 (여러 개)
과제 seed마다 '실제 − 무작위 평균' 차이와, 실제가 무작위 몇 개를 이겼는지 집계
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--task-seeds", type=int, default=5)
ap.add_argument("--shuffles", type=int, default=5)
ap.add_argument("--pn-kc-gain", type=float, default=3.0, help="3.0 ≈ KC 6%% 활성")
ap.add_argument("--t-ms", type=float, default=100)
args = ap.parse_args()

# (클래스 수, 클래스당 원형 수, 잡음, 클래스당 학습 샘플)
CONFIGS = [(20, 5, 0.5, 20), (20, 10, 0.5, 20), (10, 20, 0.5, 40), (50, 5, 0.8, 20)]
N_TEST = 50

mb = fd.Circuit.from_flywire()
enc = fd.torch.GlomerularEncoder(mb)
G = enc.n_glomeruli
mk = lambda c: fd.torch.ConnectomeLayer(c, "PN", "KC", t_ms=args.t_ms, gains={"PN>KC": args.pn_kc_gain},
                                  input_mode="regular")
layers = {"real": mk(mb)} | {f"shuffled{k}": mk(mb.shuffled(seed=k)) for k in range(args.shuffles)}
print(f"{mb}\n사구체 {G}개 | 무작위 배선 {args.shuffles}개 | 과제 seed {args.task_seeds}개\n")

acc = lambda Ftr, ytr, Fte, yte: fd.torch.train_linear(Ftr, ytr, Fte, yte)["test_acc"] * 100
summary = []
for C, K, noise, ntr in CONFIGS:
    t = time.time()
    rows = []
    for s in range(args.task_seeds):
        Xtr, ytr, Xte, yte = fd.torch.synthetic_odors(C, G, ntr, N_TEST, protos_per_class=K, noise=noise, seed=s)
        r = {"glomeruli": acc(Xtr, ytr, Xte, yte)}
        for name, L in layers.items():
            r[name] = acc(fd.torch.extract(L, enc, Xtr), ytr, fd.torch.extract(L, enc, Xte), yte)
        rows.append(r)
    glo = torch.tensor([r["glomeruli"] for r in rows])
    real = torch.tensor([r["real"] for r in rows])
    shuf = torch.tensor([[r[f"shuffled{k}"] for k in range(args.shuffles)] for r in rows])  # (seeds, shuffles)
    diff = real - shuf.mean(1)
    wins = (real[:, None] > shuf).float().mean() * 100
    print(f"클래스 {C} × 원형 {K}, 잡음 {noise}, 학습 {ntr}/클래스  ({time.time() - t:.0f}s)")
    print(f"  glomeruli {glo.mean():5.1f} | KC real {real.mean():5.1f} | KC shuffled {shuf.mean():5.1f} "
          f"(무작위끼리 범위 {shuf.mean(0).min():.1f}~{shuf.mean(0).max():.1f})")
    print(f"  실제 − 무작위: seed별 " + " ".join(f"{d:+.1f}" for d in diff)
          + f" | 평균 {diff.mean():+.2f} ± {diff.std():.2f} | 실제가 이긴 비율 {wins:.0f}%", flush=True)
    summary.append(diff)

all_diff = torch.cat(summary)
print(f"\n전체 {len(all_diff)}개 (설정×seed): 실제 − 무작위 평균 {all_diff.mean():+.2f}%p, "
      f"양수 {int((all_diff > 0).sum())}/{len(all_diff)}")
