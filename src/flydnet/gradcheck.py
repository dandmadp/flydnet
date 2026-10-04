"""스파이킹 층의 역전파 기울기가 믿을 만한지: 연결 종류별 유한 차분과 비교, 대리 기울기 감쇠 고르기

  r = fd.gradcheck(score, layer)                    # 연결 종류(pre > post)마다: 역전파 기울기 대 실제 변화
  print(r)                                           # 방향 일치(cos)와 크기 비율
  best = fd.tune_surrogate(score, layer)             # surrogate_damp 후보 중 가장 잘 맞는 값 (layer에 적용)

score(layer, seed) -> 값 하나인 Signal.

기준(차분)의 신뢰도도 함께 잼: 변화 폭 eps와 eps/2의 차분이 같은 방향이어야 기준이 성립 (fd_consistency).
스파이킹 회로는 무작위성이 없으면 출력이 연결 세기에 대해 울퉁불퉁할 수 있음 (혼돈적 되먹임 - 예: 예쁜꼬마선충).
그러면 "진짜 기울기"가 정의되지 않으므로, 무작위성이 있는 설정(input_mode="poisson")에서 seeds 여러 개의 평균 출력
(= 매끄러운 기대값)의 기울기로 비교할 것: gradcheck(score, layer, seeds=16).

방법: 연결 종류마다 그 종류의 모든 연결 배율(log_scale)을 함께 ±eps 바꿔 score의 실제 변화(중심 차분)를 재고,
같은 방향의 역전파 기울기(그 종류 연결들의 기울기 합)와 비교. 스파이킹은 미분할 수 없으므로 차분은 "작은 변화에
대한 실제 반응"이고 대리 기울기가 이것을 얼마나 따라가는지를 본다.
"""
from __future__ import annotations

from . import _check as _C
import numpy as np
import pandas as pd

from .ganglion import backend as B
from .ganglion.signal import Signal, quiescent


class GradCheck:
    def __init__(self, table: pd.DataFrame, damp: float, fd_consistency: float, seeds: int):
        self.table, self.damp, self.fd_consistency, self.seeds = table, damp, fd_consistency, seeds

    @property
    def flat(self) -> bool:
        """연결 세기를 바꿔도 출력이 변하지 않음 (출력 뉴런이 발화하지 않는 등)"""
        return not np.any(np.abs(self.table.finite_diff.to_numpy()) > 0) if len(self.table) else True

    @property
    def reliable_reference(self) -> bool:
        """차분 기준이 성립하는가 (eps와 eps/2의 차분 방향 일치 0.8 이상, 출력이 변함)"""
        return not self.flat and self.fd_consistency >= 0.8

    @property
    def cos(self) -> float:
        g, f = self.table.bptt.to_numpy(), self.table.finite_diff.to_numpy()
        return float(g @ f / (np.linalg.norm(g) * np.linalg.norm(f) + 1e-30))

    @property
    def ratio(self) -> float:
        """|역전파| / |차분| (1이면 크기까지 맞음)"""
        return float(np.linalg.norm(self.table.bptt) / (np.linalg.norm(self.table.finite_diff) + 1e-30))

    def __str__(self):
        if self.flat:
            verdict = ("판단 불가 - 연결 세기를 바꿔도 출력이 변하지 않음: 출력 뉴런이 발화하는지 확인 "
                       "(layer(x, return_all=True)로 그룹별 발화율, 약하면 calibrate나 gains로 키우기)")
        elif not self.reliable_reference:
            verdict = (f"판단 불가 - 기준(차분) 자체가 불안정 (eps·eps/2 일치 {self.fd_consistency:+.2f}): 출력이 연결 세기에 대해 "
                       "울퉁불퉁함. input_mode='poisson'으로 seeds를 늘려 평균 출력으로 비교할 것")
        else:
            verdict = ("믿을 만함" if self.cos > 0.9 and 0.2 < self.ratio < 5 else
                       "방향은 맞음, 크기가 다름 (학습률·clip으로 보정 가능)" if self.cos > 0.9 else
                       "대략 맞음" if self.cos > 0.5 else "믿으면 안 됨 - surrogate_damp를 바꾸거나(tune_surrogate) truncate")
        head = (f"역전파 기울기 확인 (surrogate_damp {self.damp}, seed {self.seeds}개 평균): 방향 일치 cos {self.cos:+.3f}, "
                f"크기 비율 {self.ratio:.3g}, 기준 신뢰도 {self.fd_consistency:+.2f} → {verdict}")
        t = self.table.reindex(self.table.finite_diff.abs().sort_values(ascending=False).index).head(8)
        return head + "\n" + t.to_string(index=False, float_format=lambda v: f"{v:.3g}")

    __repr__ = __str__


def gradcheck(score, layer, eps: float = 0.05, seed: int = 0, min_edges: int = 1, seeds=None) -> GradCheck:
    """연결 종류마다 역전파 기울기 대 유한 차분 (seeds면 그 seed들의 평균 출력으로). layer는 trainable인 ConnectomeLayer"""
    _C.pos('eps', eps)
    seed_list = [seed] if seeds is None else (list(range(seeds)) if isinstance(seeds, int) else list(seeds))
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
    for sd in seed_list:                                            # 평균 출력의 기울기 = seed마다 기울기의 평균
        val = score(layer, sd)
        if not isinstance(val, Signal) or val.data.size != 1:
            raise TypeError("score(layer, seed)는 값 하나인 Signal")
        (val * (1.0 / len(seed_list))).retrograde()
    g_all = B.numpy(layer.log_scale.retro)
    layer.log_scale.retro = None

    def mean_score():
        return float(np.mean([float(score(layer, sd).data.reshape(-1)[0]) for sd in seed_list]))
    base = layer.log_scale.data.copy()
    rows = []
    try:
        with quiescent():
            for k, sl in enumerate(slots):
                if len(sl) < min_edges:
                    continue
                diff = []
                for e in (eps, eps / 2):                             # 기준 신뢰도: 변화 폭을 절반으로
                    out = []
                    for sgn in (1.0, -1.0):
                        b = base.copy()
                        b[B.to(sl, dev)] += sgn * e
                        layer.log_scale.data = b
                        out.append(mean_score())
                    diff.append((out[0] - out[1]) / (2 * e))
                rows.append(dict(pathway=kinds[k], slots=len(sl), bptt=float(g_all[sl].sum()),
                                 finite_diff=diff[0], finite_diff_half=diff[1]))
    finally:
        layer.log_scale.data = base
    t = pd.DataFrame(rows)
    a, b_ = t.finite_diff.to_numpy(), t.finite_diff_half.to_numpy()
    cons = float(a @ b_ / (np.linalg.norm(a) * np.linalg.norm(b_) + 1e-30)) if len(t) else 0.0
    damp = layer.surrogate_damp
    label = f"auto={layer.damp_value():.3g}" if damp == "auto" else f"{damp:g}"
    return GradCheck(t, label, cons, len(seed_list))


def tune_surrogate(score, layer, candidates=("auto", 1.0, 0.3, 0.1, 0.03), eps: float = 0.05, seed: int = 0,
                   apply: bool = True, verbose: bool = True, seeds=None):
    """surrogate_damp 후보마다 gradcheck → 방향 일치(cos)가 가장 높은 값 (같으면 크기 비율이 1에 가까운 것).
    apply면 layer에 적용. 기준(차분)이 어느 후보에서도 성립하지 않으면 고르지 않고 None (층은 그대로) -
    그때는 input_mode="poisson"과 seeds=16 등으로 평균 출력의 기울기를 기준으로 다시"""
    for d in candidates:                                       # "auto" 또는 (0, 1]의 감쇠 값
        if d != "auto":
            _C.unit("candidates의 값", d, lo_open=True)
    from ._console import say
    old = layer.surrogate_damp
    best, key, reliable = None, None, False
    for d in candidates:
        layer.surrogate_damp = d if d == "auto" else float(d)
        r = gradcheck(score, layer, eps=eps, seed=seed, seeds=seeds)
        if verbose:
            say(f"  surrogate_damp {r.damp:<10} cos {r.cos:+.3f}  크기 비율 {r.ratio:.3g}  기준 신뢰도 {r.fd_consistency:+.2f}",
                flush=True)
        if not r.reliable_reference:
            continue
        reliable = True
        k = (round(r.cos, 2), -abs(np.log(max(r.ratio, 1e-30))))
        if key is None or k > key:
            best, key = (d if d == "auto" else float(d)), k
    if not reliable:
        layer.surrogate_damp = old
        if verbose:
            say("  기준(차분)이 어느 후보에서도 불안정 - 고르지 않음. input_mode='poisson'과 seeds=16 등으로 다시", flush=True)
        return None
    layer.surrogate_damp = best if apply else old
    if apply:
        layer.config["surrogate_damp"] = best
    return best
