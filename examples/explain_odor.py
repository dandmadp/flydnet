"""
fd.explain 예: 실제 냄새를 구분하도록 학습한 버섯체 모델은 어떤 세포 유형에 기대는가

  python examples/explain_odor.py              # 약 1분 (GPU)

과제: DoOR 2.0 실제 냄새 반응 (사구체별) 중 냄새 K개를 잡음 섞인 시료로 만들어 구분
모델: 사구체 → PN (GlomerularEncoder) → KC (스파이킹 ConnectomeLayer, 실제 배선) → 학습한 선형 리드아웃
설명: 정답 로짓이 어떤 세포 유형(PN 사구체, KC 하위 유형)에 기대는지 → 실제로 꺼서 확인
정답 맞추기: 냄새마다 "모델이 기대는 사구체"와 "그 냄새에 실제로 반응하는 사구체 (DoOR)"의 상관
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--odors", type=int, default=8, help="구분할 냄새 수")
ap.add_argument("--samples", type=int, default=40, help="냄새마다 학습·평가 시료 수 (각각)")
ap.add_argument("--noise", type=float, default=0.6, help="곱 잡음 (로그 정규 시그마)")
ap.add_argument("--verify", type=int, default=5, help="실제로 꺼서 확인할 유형 수 (큰 쪽·작은 쪽 각각)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
fd.ganglion.limit_gpu_memory(0.75)
rng = np.random.default_rng(args.seed)

mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)
door = fd.door_odors(enc.glomeruli)
strength = door["X"].sum(1)
pick = np.argsort(strength)[::-1][:args.odors]                    # 반응이 큰 냄새 K개
proto = door["X"][pick] / door["X"][pick].max()
print(f"냄새 {args.odors}개: {', '.join(door['names'][i] for i in pick)}")


def make(n):
    y = np.repeat(np.arange(args.odors), n)
    X = proto[y] * rng.lognormal(0, args.noise, (len(y), proto.shape[1])) + rng.uniform(0, 0.15, (len(y), proto.shape[1]))
    return np.clip(X, 0, 1).astype(np.float32), y


Xtr, ytr = make(args.samples)
Xte, yte = make(args.samples)
layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=50, dt=0.5, gains={"PN>KC": 3.0}, input_mode="regular")
Ftr, Fte = fd.extract(layer, enc, Xtr, batch=200), fd.extract(layer, enc, Xte, batch=200)
mu, sd = Ftr.mean(0), max(float((Ftr - Ftr.mean(0)).std()), 1e-6)
res = fd.train_linear(Ftr, ytr, Fte, yte, epochs=60, seed=args.seed)
readout = res["model"]
print(f"평가 정확도 {res['test_acc']:.3f} (찍기 {1 / args.odors:.3f})")
mu_, sd_ = fd.ganglion.backend.to(mu, layer.device), sd


def correct_logit(X, y):
    """정답 클래스 로짓의 합 - 모델이 맞히는 근거를 설명할 값"""
    def score(L, seed):
        F = (L(enc(X), seed=seed) - mu_) * (1.0 / sd_)              # train_linear과 같은 정규화
        return readout(F)[np.arange(len(y)), y].sum()
    return score


print("\n=== 전체: 정답 로짓이 기대는 세포 유형 ===")
rep = fd.explain(correct_logit(Xte, yte), layer, by="cell_type", pathways=True, verify=args.verify)
print(rep)

print("\n=== 냄새마다: 모델이 기대는 사구체 대 실제로 반응하는 사구체 (DoOR) ===")
types = mb.meta.cell_type.astype(str).to_numpy()
glom_of = {t: t.split("_")[0] for t in set(types[mb.groups["PN"]])}
rs = []
for k, i in enumerate(pick):
    m = yte == k
    r = fd.explain(correct_logit(Xte[m], yte[m]), layer, by="cell_type")
    g = r.groups[r.groups.name.isin(glom_of)].copy()
    g["glom"] = g.name.map(glom_of)
    per = g.groupby("glom").pred_drop.sum()
    per = per[per.index.isin(enc.glomeruli)]                             # 단일 사구체형 PN만 (입력을 받는 PN)
    resp = proto[k][[enc.glomeruli.index(x) for x in per.index]]
    c = float(np.corrcoef(per.to_numpy(), resp)[0, 1])
    rs.append(c)
    top = per.sort_values(ascending=False).index[:3].tolist()
    true = [enc.glomeruli[j] for j in np.argsort(proto[k])[::-1][:3]]
    print(f"  {door['names'][i][:28]:<28} 상관 {c:+.2f}   모델이 기대는 사구체 {top}   가장 크게 반응하는 사구체 {true}")
print(f"평균 상관 {np.mean(rs):+.2f} (냄새 {len(rs)}개 중 양수 {sum(c > 0 for c in rs)}개)")
