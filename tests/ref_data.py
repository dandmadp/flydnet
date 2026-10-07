"""회로·데이터 대 원본 파일 직접 집계: fd.flywire()·fd.brain()·visual_circuit·worm의 뉴런·연결·세기·부호·번호,
사구체 대응, DoOR 반응 (자발 발화 빼기). FlyWire·DoOR 데이터가 필요

  python tests/ref_data.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import flydnet as fd
from flydnet.data import missing

bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


if missing("flywire"):
    print("FlyWire 데이터 없음 - 건너뜀")
    sys.exit(0)
d = fd.data_dir("flywire")
ids = pd.read_csv(d / "Completeness_783.csv", index_col=0).index.values.astype(np.int64)
ann = pd.read_csv(d / "flywire_annotations.tsv", sep="\t", low_memory=False)
try:                                                                     # 원본 parquet와 직접 비교 (npz는 flydnet이 만든 것이라 기준이 못 됨)
    con = pd.read_parquet(d / "Connectivity_783.parquet")
except (FileNotFoundError, ImportError) as e:
    print(f"원본 연결 parquet를 읽을 수 없음 ({type(e).__name__}) - 건너뜀. 비교하려면 pip install pyarrow 후 "
          "python -m flydnet download flywire")
    sys.exit(0)

# 1) 전체 뇌: 번호 = Completeness 순서, 연결 = 원본 그대로, 세기 = 시냅스 수 x 부호
br = fd.brain()
ok("brain 뉴런 수", br.N == len(ids) == 138639, br.N)
ok("brain 번호 순서 = Completeness", np.array_equal(br.root_ids, ids))
ok("brain 연결 수", br.n_edges == len(con))
ok("brain 연결 = 원본", np.array_equal(br.pre, con.Presynaptic_Index.values) and np.array_equal(br.post, con.Postsynaptic_Index.values))
ok("brain 세기 = 시냅스 수 x 부호", np.array_equal(br.weight, (con.Connectivity * con.Excitatory).values.astype(np.float32)))
ok("brain ID 대응 (Presynaptic_ID)", np.array_equal(ids[con.Presynaptic_Index.values[:1000]], con.Presynaptic_ID.values[:1000]))
a = ann.drop_duplicates("root_id").set_index("root_id").reindex(ids)
sc = a.super_class.astype("string").fillna("unannotated")
ok("brain 그룹 = super_class", all(len(br.groups[k]) == int((sc == k).sum()) for k in br.groups))

# 2) 버섯체 (오른쪽): 그룹 = 주석 조건, 연결 = 그 뉴런들 사이 원본 연결 전부
mb = fd.flywire()
right = ann[ann.root_id.isin(ids) & (ann.side == "right")]
for g, (col, val) in fd.MUSHROOM_BODY.items():
    want = set(right.root_id[right[col].isin([val] if isinstance(val, str) else val)])
    ok(f"flywire 그룹 {g} 뉴런", set(mb.root_ids[mb.groups[g]]) == want, (len(mb.groups[g]), len(want)))
pos = pd.Series(np.arange(len(ids)), index=ids)
glob = pos[mb.root_ids].values
m = np.isin(con.Presynaptic_Index.values, glob) & np.isin(con.Postsynaptic_Index.values, glob)
ok("flywire 연결 수 = 원본 유도 부분 그래프", mb.n_edges == int(m.sum()), (mb.n_edges, int(m.sum())))
ok("flywire 세기 합", np.isclose(mb.weight.sum(), (con.Connectivity * con.Excitatory).values[m].sum()))
ok("flywire 연결의 ID", set(zip(mb.root_ids[mb.pre], mb.root_ids[mb.post])) ==
   set(zip(con.Presynaptic_ID.values[m], con.Postsynaptic_ID.values[m])))
ok("flywire meta 순서 = 뉴런 순서", np.array_equal(mb.meta.root_id.values, mb.root_ids))

# 3) 사구체 대응: 단일 사구체형 PN마다 사구체 하나, 같은 사구체 PN은 같은 발화율
enc = fd.Glomeruli(mb, device="cpu")
P = np.asarray(enc.P)
pm = mb.meta.iloc[mb.groups["PN"]]
uni = pm.cell_sub_class.astype(str).eq("uniglomerular").values
ok("사구체: 단일 사구체형 PN은 정확히 하나", np.all(P[uni].sum(1) == 1) and np.all(P[~uni].sum(1) == 0))
gl = pm.cell_type.astype(str).str.split("_").str[0].values
ok("사구체: 이름 대응", all(enc.glomeruli[P[i].argmax()] == gl[i] for i in np.nonzero(uni)[0]))

# 4) 시각계: 광수용체가 보내는 연결은 모두 억제, 나머지 부호는 원본 그대로
vc = fd.visual_circuit()
ph = np.concatenate([vc.groups[g] for g in fd.PHOTORECEPTORS if g in vc.groups])
from_ph = np.isin(vc.pre, ph)
ok("시각계: 광수용체 출력 = 억제", np.all(vc.weight[from_ph] < 0))
raw = fd.Circuit.from_flywire(fd.VISUAL_SYSTEM, side="right", group_by="cell_type")
ok("시각계: 나머지는 원본 부호", np.array_equal(vc.weight[~from_ph], raw.weight[~from_ph]))
ok("시각계: 크기는 그대로", np.array_equal(np.abs(vc.weight), np.abs(raw.weight)))

# 5) DoOR: 반응 = 원본 - 자발 발화(SFR), 0 아래는 0
if not missing("door"):
    dd = fd.data_dir("door")
    R = pd.read_csv(dd / "door_response_matrix.csv", sep=";")
    M = pd.read_csv(dd / "door_mappings.csv", sep=";")
    door = fd.door_odors(enc.glomeruli, min_measured=0)
    rec = M.dropna(subset=["receptor", "code"])
    rec = rec[rec.receptor.isin(R.columns) & rec.code.isin(enc.glomeruli)].iloc[0]
    j = enc.glomeruli.index(rec.code)
    raw_v = R[rec.receptor].drop(index="SFR").values
    exp = np.where(np.isnan(raw_v), 0, np.maximum(raw_v - R.loc["SFR", rec.receptor], 0))
    ok(f"DoOR {rec.receptor}→{rec.code} = 원본 - SFR", np.allclose(door["X"][:, j], exp, atol=1e-6))

# 6) 예쁜꼬마선충: GABA 뉴런의 화학 시냅스만 억제
if not missing("worm"):
    w = fd.worm()
    gaba = set(w.meta.node[w.meta.gaba])
    src = w.meta.node.values[w.pre]
    ok("worm: GABA 출력 = 억제", np.all(w.weight[np.isin(src, list(gaba))] < 0))
    ok("worm: 나머지 = 흥분", np.all(w.weight[~np.isin(src, list(gaba))] > 0))

print(f"회로·데이터: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
