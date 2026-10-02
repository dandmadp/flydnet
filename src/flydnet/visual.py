"""초파리 시각계 회로: 광수용체 → 라미나 → 메둘라 → T4/T5(운동 감지) → 소엽판 뉴런

  circ = fd.visual_circuit()                 # 오른쪽 시각엽, 세포 유형마다 그룹 하나
  xy   = fd.column_map(circ)                 # 뉴런별 시야 좌표 (단위 ≈ 기둥 간격)
  lum  = fd.drifting_grating(xy[circ.groups["R1-6"]], directions, ...)   # (B, T, n) 밝기 −1~1

시야 좌표는 FlyWire 주석의 뉴런 대표 점(pos)을 메둘라 평면에 투영한 뒤, 같은 기둥에 속한 뉴런끼리의
연결을 따라 평균내서 다듬은 것 (대표 점만으로는 격자가 고르지 않음). 정밀한 기둥 배정
(Matsliah et al. 2024)보다 거칠다는 점에 주의.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .circuit import Circuit

PHOTORECEPTORS = ["R1-6", "R7", "R8"]
# 광수용체 + 시각엽 내부 뉴런 + 시각엽에서 중앙 뇌로 가는 뉴런(소엽판 HS/VS 포함)
VISUAL_SYSTEM = {
    "photoreceptor": ("cell_type", PHOTORECEPTORS),
    "optic": ("super_class", ["optic", "visual_projection"]),
}
# 기둥마다 하나씩 있는 세포 유형 (시야 지도를 다듬을 때 씀)
COLUMNAR = PHOTORECEPTORS + ["L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
                             "Tm20", "Tm21", "Mi15", "C2", "C3", "T1", "T2", "T2a", "T3"] + \
           [f"T{k}{d}" for k in "45" for d in "abcd"]
# 운동 감지 경로만: 광수용체 → 라미나 → T4/T5 입력 메둘라 뉴런 → T4/T5 (Circuit.subset으로 회로를 좁혀 빠르게)
MOTION_PATHWAY = PHOTORECEPTORS + ["L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2", "Lai", "C2", "C3", "T1", "CT1",
                                   "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9"] + \
                 [f"T{k}{d}" for k in "45" for d in "abcd"]
# 큰 운동 감지 뉴런 (소엽판 접선 세포): 수평 HS, 수직 VS, H1/H2
LPTC = ["HSE", "HSN", "HSS", "H1", "H2"] + [f"VS{i}" for i in range(1, 9)] + ["VSm", "VST1", "VST2"]


def visual_circuit(side: str = "right", data_dir: str | Path | None = None) -> Circuit:
    """오른쪽(또는 왼쪽) 시각계. 그룹 = 세포 유형 (약 700개), 광수용체 출력은 억제로 바꿈
    (히스타민은 받는 뉴런을 과분극시킴. 빛 → L1/L2 막전위 감소)"""
    c = Circuit.from_flywire(VISUAL_SYSTEM, side=side, data_dir=data_dir, group_by="cell_type")
    return c.with_sign([g for g in PHOTORECEPTORS if g in c.groups], -1)


def column_map(circuit: Circuit, anchor: str = "Mi1", smooth: int = 3, columnar=None) -> np.ndarray:
    """뉴런별 시야 좌표 (N, 2). 기둥 세포가 아니면 NaN.
    1) anchor 세포들의 대표 점으로 메둘라 평면(주성분 2개)을 잡고 모든 기둥 세포를 그 평면에 투영
    2) 유형마다 평균 위치를 anchor 평균에 맞춤 (대표 점이 유형마다 다른 깊이·신경망에 찍혀 생기는 일정한 어긋남 제거.
       기둥 세포 유형은 모두 같은 시야를 덮으므로)
    3) 기둥 세포끼리 연결(시냅스 수 가중)을 따라 smooth번 이웃 평균 → 고르지 않은 대표 점을 다듬음
    4) anchor 좌표를 등방(공분산 = 단위행렬 배수)으로 맞추고, anchor 하나가 넓이 1을 차지하도록 크기 조정
       → 단위 ≈ 기둥 간격"""
    if circuit.pos is None:
        raise ValueError("회로에 뉴런 위치(pos)가 없음 (Circuit.from_flywire로 만든 회로 필요)")
    columnar = COLUMNAR if columnar is None else columnar
    col_groups = [g for g in columnar if g in circuit.groups]
    is_col = np.zeros(circuit.N, bool)
    for g in col_groups:
        is_col[circuit.groups[g]] = True
    anc = circuit.groups[anchor]
    X = circuit.pos.astype(np.float64)
    mu = X[anc].mean(0)
    _, _, Vt = np.linalg.svd(X[anc] - mu, full_matrices=False)
    Z = (X - mu) @ Vt[:2].T

    def center(Z):
        Z = Z.copy()
        for gname in col_groups:
            idx = circuit.groups[gname]
            Z[idx] -= Z[idx].mean(0) - Z[anc].mean(0)
        return Z
    Z = torch.tensor(center(Z))

    m = is_col[circuit.pre] & is_col[circuit.post]
    i, j, w = circuit.pre[m], circuit.post[m], np.abs(circuit.weight[m]).astype(np.float64)
    A = torch.sparse_coo_tensor(np.stack([np.r_[i, j], np.r_[j, i]]), np.r_[w, w], (circuit.N, circuit.N),
                                check_invariants=False).coalesce()
    deg = torch.tensor(np.bincount(np.r_[i, j], weights=np.r_[w, w], minlength=circuit.N)).clamp_min(1e-9)
    for _ in range(smooth):
        Z = 0.5 * Z + 0.5 * torch.sparse.mm(A, Z) / deg[:, None]
    Z = center(Z.numpy())

    Za = Z[anc] - Z[anc].mean(0)                                 # 등방으로 + 기둥 간격 단위로
    evals, evecs = np.linalg.eigh(np.cov(Za.T))
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T
    Z = (Z - Z[anc].mean(0)) @ W.T
    area = 4 * np.pi                                            # 공분산 I인 균일 타원의 넓이 = 4π·√det
    Z *= np.sqrt(len(anc) / area)                               # → 넓이 = anchor 개수 (하나당 1)
    Z[~is_col] = np.nan
    return Z.astype(np.float32)


def drifting_grating(xy, directions, t_ms: float, frames: int, wavelength: float = 8.0, temporal_hz: float = 5.0,
                     phase=None, contrast: float = 1.0, onset_ms: float = 0.0) -> torch.Tensor:
    """움직이는 사인파 격자. 밝기 (B, frames, n), −contrast ~ +contrast (onset 전은 0 = 회색)
    xy: (n, 2) 시야 좌표 (기둥 간격 단위) / directions: (B,) 도 (0 = +x 방향)
    wavelength: 기둥 수 / temporal_hz: 한 점에서 밝기가 바뀌는 빈도 (속도 = wavelength × temporal_hz 기둥/s)"""
    xy = torch.as_tensor(np.nan_to_num(np.asarray(xy, np.float32)))
    th = torch.as_tensor(directions, dtype=torch.float32) * np.pi / 180
    B = len(th)
    phase = torch.zeros(B) if phase is None else torch.as_tensor(phase, dtype=torch.float32)
    t = (torch.arange(frames) + 0.5) * t_ms / frames                            # ms, 프레임 중앙
    k = torch.stack([torch.cos(th), torch.sin(th)], 1) * (2 * np.pi / wavelength)  # (B, 2)
    proj = k @ xy.T                                                              # (B, n)
    tt = (t - onset_ms).clamp_min(0) / 1000.0
    lum = contrast * torch.sin(proj[:, None, :] - 2 * np.pi * temporal_hz * tt[None, :, None] + phase[:, None, None])
    return lum * (t >= onset_ms).float()[None, :, None]


def direction_offsets(circuit: Circuit, xy: np.ndarray, post_types=("T4a", "T4b", "T4c", "T4d"),
                      a: str = "Mi9", b: str = "Mi4"):
    """배선 속 방향 정보: 받는 뉴런마다 a 입력 위치 → b 입력 위치 벡터(시냅스 가중 평균).
    유형별 (평균 방향 도, 일관성 R 0~1). 무작위 배선이면 R ≈ 0"""
    import pandas as pd
    g = circuit.group_of()
    df = pd.DataFrame({"pre": circuit.pre, "post": circuit.post, "w": np.abs(circuit.weight),
                       "pt": g[circuit.pre], "qt": g[circuit.post]})
    out = {}
    for t in post_types:
        s = df[(df.qt == t) & df.pt.isin([a, b])]
        cen = s.assign(x=xy[s.pre, 0] * s.w, y=xy[s.pre, 1] * s.w).groupby(["post", "pt"])[["x", "y", "w"]].sum()
        cen = cen[["x", "y"]].div(cen.w, axis=0).unstack("pt").dropna()
        v = np.stack([cen[("x", b)] - cen[("x", a)], cen[("y", b)] - cen[("y", a)]], 1)
        u = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
        m = u.mean(0)
        out[t] = (float(np.degrees(np.arctan2(m[1], m[0]))), float(np.linalg.norm(m)))
    return out
