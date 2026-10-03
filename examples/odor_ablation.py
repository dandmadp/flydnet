"""
실험 ③-b: 실제 배선의 이점은 어느 연결에서 나오나? — 연결 종류별로 하나씩만 무작위화

  python examples/odor_ablation.py
  python examples/odor_ablation.py --task-seeds 3 --shuffles 2   # 빠르게

조건 (odor_task.py와 같은 과제·설정, 섞을 때 뉴런별 연결 수는 그대로)
  전체    : 모든 연결
  PN→KC   : 투사 뉴런 → Kenyon 세포만
  KC→KC   : Kenyon 세포끼리만
  APL     : APL → KC, KC → APL
  나머지  : 위 셋을 뺀 전부 (MBON 관련, PN끼리 등)
'손실' = 실제 배선 정확도 − 해당 조건 정확도. 손실이 클수록 그 연결 구조가 이점의 출처
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
ap.add_argument("--shuffles", type=int, default=3)
ap.add_argument("--pn-kc-gain", type=float, default=3.0)
ap.add_argument("--t-ms", type=float, default=100)
args = ap.parse_args()

CONFIGS = [(20, 5, 0.5, 20), (20, 10, 0.5, 20), (10, 20, 0.5, 40)]
N_TEST = 50
APL = ["APL>KC", "KC>APL"]
CONDITIONS = {
    "전체": dict(),
    "PN→KC": dict(pairs=["PN>KC"]),
    "KC→KC": dict(pairs=["KC>KC"]),
    "APL": dict(pairs=APL),
    "나머지": dict(exclude=["PN>KC", "KC>KC"] + APL),
}

mb = fd.Circuit.from_flywire()
enc = fd.torch.GlomerularEncoder(mb)
G = enc.n_glomeruli
mk = lambda c: fd.torch.ConnectomeLayer(c, "PN", "KC", t_ms=args.t_ms, gains={"PN>KC": args.pn_kc_gain},
                                  input_mode="regular")
layers = {("실제", 0): mk(mb)}
for cname, kw in CONDITIONS.items():
    for k in range(args.shuffles):
        layers[(cname, k)] = mk(mb.shuffled(seed=k, **kw))
print(f"{mb}\n조건 {len(CONDITIONS)}개 × 무작위 {args.shuffles}개 | 과제 seed {args.task_seeds}개\n")

acc = lambda Ftr, ytr, Fte, yte: fd.torch.train_linear(Ftr, ytr, Fte, yte)["test_acc"] * 100
loss = {c: [] for c in CONDITIONS}                      # 조건별 (설정×seed) 손실
for C, K, noise, ntr in CONFIGS:
    t = time.time()
    res = {key: [] for key in layers}
    for s in range(args.task_seeds):
        Xtr, ytr, Xte, yte = fd.torch.synthetic_odors(C, G, ntr, N_TEST, protos_per_class=K, noise=noise, seed=s)
        for key, L in layers.items():
            res[key].append(acc(fd.torch.extract(L, enc, Xtr), ytr, fd.torch.extract(L, enc, Xte), yte))
    real = torch.tensor(res[("실제", 0)])
    line = f"  실제 {real.mean():5.1f}"
    for cname in CONDITIONS:
        sh = torch.tensor([res[(cname, k)] for k in range(args.shuffles)]).mean(0)   # seed별 무작위 평균
        d = real - sh
        loss[cname].append(d)
        line += f" | {cname} {sh.mean():5.1f} (손실 {d.mean():+.2f})"
    print(f"클래스 {C} × 원형 {K}, 학습 {ntr}/클래스 ({time.time() - t:.0f}s)\n{line}", flush=True)

print(f"\n━━ 조건별 손실 (실제 − 무작위, {len(CONFIGS)}설정 × {args.task_seeds}seed) ━━")
for cname, ds in loss.items():
    d = torch.cat(ds)
    print(f"  {cname:<6} {d.mean():+.2f} ± {d.std() / len(d) ** 0.5:.2f} (표준오차) | 양수 {int((d > 0).sum())}/{len(d)}")
