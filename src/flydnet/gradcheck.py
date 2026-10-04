"""스파이킹 층의 역전파 기울기가 믿을 만한지: 연결 종류별 유한 차분과 비교, 대리 기울기 감쇠 고르기

  r = fd.gradcheck(score, layer)                    # 연결 종류(pre > post)마다: 역전파 기울기 대 실제 변화
  print(r)                                           # 방향 일치(cos)와 크기 비율
  best = fd.tune_surrogate(score, layer)             # surrogate_damp 후보 중 가장 잘 맞는 값 (layer에 적용)

score(layer, seed) -> 값 하나인 Signal. 입력은 무작위성이 없게 (input_mode="regular" 권장) - 아니면 seed를 고정한
같은 순전파끼리 비교하므로 차분에 포아송 잡음이 섞임.

방법: 연결 종류마다 그 종류의 모든 연결 배율(log_scale)을 함께 ±eps 바꿔 score의 실제 변화(중심 차분)를 재고,
같은 방향의 역전파 기울기(그 종류 연결들의 기울기 합)와 비교. 스파이킹은 미분할 수 없으므로 차분은 "작은 변화에
대한 실제 반응"이고 대리 기울기가 이것을 얼마나 따라가는지를 본다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ganglion import backend as B
from .ganglion.signal import Signal, quiescent


class GradCheck:
    def __init__(self, table: pd.DataFrame, damp: float):
        self.table, self.damp = table, damp

    @property
    def cos(self) -> float:
        g, f = self.table.bptt.to_numpy(), self.table.finite_diff.to_numpy()
        return float(g @ f / (np.linalg.norm(g) * np.linalg.norm(f) + 1e-30))

    @property
    def ratio(self) -> float:
        """|역전파| / |차분| (1이면 크기까지 맞음)"""
        return float(np.linalg.norm(self.table.bptt) / (np.linalg.norm(self.table.finite_diff) + 1e-30))

    def __str__(self):
        verdict = ("믿을 만함" if self.cos > 0.9 and 0.2 < self.ratio < 5 else
                   "방향은 맞음, 크기가 다름 (학습률·clip으로 보정 가능)" if self.cos > 0.9 else
                   "대략 맞음" if self.cos > 0.5 else "믿으면 안 됨 - surrogate_damp를 낮추거나 truncate를 쓸 것")
        head = (f"역전파 기울기 확인 (surrogate_damp {self.damp:g}): 방향 일치 cos {self.cos:+.3f}, "
                f"크기 비율 {self.ratio:.3g} → {verdict}")
        t = self.table.reindex(self.table.finite_diff.abs().sort_values(ascending=False).index).head(8)
        return head + "\n" + t.to_string(index=False, float_format=lambda v: f"{v:.3g}")

    __repr__ = __str__


def gradcheck(score, layer, eps: float = 0.05, seed: int = 0, min_edges: int = 1) -> GradCheck:
    """연결 종류마다 역전파 기울기 대 유한 차분. layer는 trainable인 ConnectomeLayer"""
    if not getattr(layer, "trainable", False):
        raise ValueError("학습하는 연결이 있는 ConnectomeLayer에서만 (trainable=...)")
    dev = layer.device
    which = B.numpy(layer.train_which)
    code = layer._edge_code[B.numpy(layer.train_pos)]
    names = np.array(layer._edge_names(code), dtype=object)
    kinds, inv = np.unique(names, return_inverse=True)
    # 연결 종류 → log_scale 칸 (share="pair"면 칸 하나, "edge"면 여러 칸)
    slots = [np.unique(which[inv == k]) for k in range(len(kinds))]
    layer.log_scale.retro = None
    val = score(layer, seed)
    if not isinstance(val, Signal) or val.data.size != 1:
        raise TypeError("score(layer, seed)는 값 하나인 Signal")
    val.retrograde()
    g_all = B.numpy(layer.log_scale.retro)
    layer.log_scale.retro = None
    base = layer.log_scale.data.copy()
    rows = []
    try:
        with quiescent():
            for k, sl in enumerate(slots):
                if len(sl) < min_edges:
                    continue
                out = []
                for sgn in (1.0, -1.0):
                    b = base.copy()
                    b[B.to(sl, dev)] += sgn * eps
                    layer.log_scale.data = b
                    out.append(float(score(layer, seed).data.reshape(-1)[0]))
                rows.append(dict(pathway=kinds[k], slots=len(sl), bptt=float(g_all[sl].sum()),
                                 finite_diff=(out[0] - out[1]) / (2 * eps)))
    finally:
        layer.log_scale.data = base
    return GradCheck(pd.DataFrame(rows), layer.surrogate_damp)


def tune_surrogate(score, layer, candidates=(1.0, 0.3, 0.1, 0.03), eps: float = 0.05, seed: int = 0,
                   apply: bool = True, verbose: bool = True) -> float:
    """surrogate_damp 후보마다 gradcheck → 방향 일치(cos)가 가장 높은 값 (같으면 크기 비율이 1에 가까운 것).
    apply면 layer.surrogate_damp에 적용"""
    from ._console import say
    old = layer.surrogate_damp
    best, key = None, None
    for d in candidates:
        layer.surrogate_damp = float(d)
        r = gradcheck(score, layer, eps=eps, seed=seed)
        k = (round(r.cos, 2), -abs(np.log(max(r.ratio, 1e-30))))
        if verbose:
            say(f"  surrogate_damp {d:<6g} cos {r.cos:+.3f}  크기 비율 {r.ratio:.3g}", flush=True)
        if key is None or k > key:
            best, key = float(d), k
    layer.surrogate_damp = best if apply else old
    if apply:
        layer.config["surrogate_damp"] = best
    return best
