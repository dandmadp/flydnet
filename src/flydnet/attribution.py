"""회로 기여도: 학습된 모델이 답을 낼 때 어떤 세포 유형·경로에 기대는지, 실제로 꺼 봐서 확인까지

  rep = fd.explain(score, layer, by="cell_type", pathways=True, verify=5)
  print(rep)              # 세포 유형별 예측 기여 + 경로별 기여 + 실제로 끈 결과와의 일치

score(layer, seed) -> 스칼라 Signal: 설명할 값 (예: 정답 클래스 로짓의 합, 어떤 뉴런의 발화율).

방법 (가상 손상의 1차 근사):
  뉴런마다 발화에 곱하는 배율 m(=1)을 두고 기울기 d score / d m 을 구한다. m을 1 → 0으로 내리는 것이 그 뉴런을 끄는 것
  (fd.genetics.silence와 같은 조작)이므로, 기울기는 "끄면 score가 얼마나 줄어드는가"의 1차 예측(pred_drop)이다.
  세포 유형의 예측 = 그 유형 뉴런들의 합 (모두 함께 끄는 것의 1차 예측). 연결도 같은 방법 (연결마다 배율 → 경로별 합).
  verify=k: 예측이 큰 유형 k개와 나머지에서 고르게 k개를 실제로 꺼서(silence) 측정한 감소(actual_drop)와 비교 →
  순위 상관. 스파이킹 뉴런은 대리 기울기라 1차 예측이 빗나갈 수 있어서, 확인 결과를 함께 본다.
  쓰는 법: 기울기(pred_drop)는 모든 유형을 역전파 한 번으로 훑어 후보를 고르는 용도, 결론은 실제로 끈 값(actual_drop).
  유형 하나를 실제로 끄는 것은 순전파 한 번이라, 기울기를 여러 번 적분하는 것보다 싸고 정확함.
  1차 예측은 끄는 도중 반응이 크게 휘는 유형(되먹임 억제 뉴런 APL 등)에서 방향까지 틀릴 수 있음 → 확인 결과에 표시.
  모든 유형의 정확한 값이 필요하면 fd.genetics.screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ganglion import backend as B
from .ganglion.signal import Signal, quiescent


def default_by(circuit) -> str:
    """기본 묶음 기준: 주석에 cell_type이 있으면 그것 (커넥톰), 없으면 회로 그룹 (일반 그래프)"""
    return "cell_type" if circuit.meta is not None and "cell_type" in circuit.meta.columns else "group"


def _labels(circuit, by: str | None) -> np.ndarray:
    """뉴런별 이름: by = "group" 또는 주석 열 (값이 없으면 "<그룹>?"). None이면 default_by"""
    by = by or default_by(circuit)
    group = circuit.group_of()
    if by == "group":
        return group.astype(object)
    if circuit.meta is None or by not in circuit.meta.columns:
        raise KeyError(f"주석 열이 없음: {by} (by='group'이나 {list(circuit.meta.columns) if circuit.meta is not None else []})")
    v = circuit.meta[by].astype("string").to_numpy(dtype=object, na_value=None)
    return np.array([x if x is not None else f"{g}?" for x, g in zip(v, group)], dtype=object)


class Explanation:
    """explain()의 결과. groups / pathways / verified 표와 뉴런별 예측(neurons)"""

    def __init__(self, base, by, neurons, groups, pathways, verified, seeds, effects):
        self.base, self.by, self.neurons = base, by, neurons
        self.groups, self.pathways, self.verified = groups, pathways, verified
        self.seeds, self.effects = seeds, effects

    @property
    def agreement(self) -> float | None:
        """확인한 유형들에서 예측과 실제 감소의 순위 상관 (Spearman)"""
        v = self.verified
        if v is None or len(v) < 3 or v.pred_drop.nunique() < 2 or v.actual_drop.nunique() < 2:
            return None
        return float(v.pred_drop.rank().corr(v.actual_drop.rank()))

    def __str__(self):
        out = [f"설명할 값 (seed {len(self.seeds)}개 평균): {self.base:.4g}"]
        if self.effects:
            out.append(f"켜진 효과기 아래에서 계산: {self.effects}")
        g = self.groups.head(10)
        out.append(f"\n{self.by}별 기여 (끄면 줄어들 것으로 예측되는 양, 큰 순 10개)")
        out.append(g[["name", "n", "pred_drop", "share"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        if self.pathways is not None:
            out.append("\n경로별 기여 (연결 종류 pre > post, 큰 순 10개)")
            out.append(self.pathways.head(10)[["pathway", "edges", "pred_drop", "share"]].to_string(
                index=False, float_format=lambda v: f"{v:.3g}"))
        if self.verified is not None:
            out.append("\n확인: 실제로 끄기 (silence)")
            out.append(self.verified[["name", "pred_drop", "actual_drop"]].to_string(
                index=False, float_format=lambda v: f"{v:.3g}"))
            v = self.verified
            big = v.actual_drop.abs() > 0.05 * (v.actual_drop.abs().max() or 1)
            flip = v[big & (np.sign(v.pred_drop) != np.sign(v.actual_drop))]
            if len(flip):
                out.append(f"  ! 1차 예측이 방향을 틀린 유형: {', '.join(flip.name)} - 끄는 도중 반응이 크게 휘는 유형 "
                           "(되먹임 억제 등). 이 유형들은 actual_drop을 쓸 것")
            a = self.agreement
            if a is None:
                out.append("  예측·실제 순위 상관: 계산 불가 (확인한 유형이 너무 적거나 값이 모두 같음)")
            else:
                note = ("예측을 믿을 만함" if a >= 0.7 else "예측은 대략적 - 실제로 끈 결과를 기준으로" if a >= 0.3
                        else "예측이 빗나감 - 이 모델에서는 실제로 끈 결과만 믿을 것")
                out.append(f"  예측·실제 순위 상관 {a:.2f}: {note}")
        return "\n".join(out)

    __repr__ = __str__


def explain(score, layer, by: str | None = None, pathways: bool = False, verify: int = 0, seeds=1,
            verbose: bool = False) -> Explanation:
    """score(layer, seed)가 어떤 세포 유형(by)·경로에 기대는지. layer는 ConnectomeLayer

    by:       "cell_type" 등 회로 주석 열, 또는 "group" (회로 그룹). None이면 cell_type 주석이 있으면 그것, 없으면 group
    pathways: 연결 종류(by 이름 pre > post)별 기여도 (연결마다 기울기 - 전체 뇌면 메모리가 더 듦)
    verify:   예측이 큰 유형 k개 + 나머지(예측 0 제외)에서 순위를 고르게 k개를 실제로 꺼서 확인 (k x 2 x seed 수 만큼 순전파)
    seeds:    정수(개수) 또는 목록. 예측·확인 모두 seed 평균"""
    from . import genetics as G
    if not hasattr(layer, "_probe"):
        raise TypeError("ConnectomeLayer에서만 (fd.ConnectomeLayer)")
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    if not seeds:
        raise ValueError("seeds는 1개 이상")
    circuit, dev = layer.circuit, layer.device
    by = by or default_by(circuit)
    xp = B.xp(dev)
    labels = _labels(circuit, by)
    names, inv = np.unique(labels.astype(str), return_inverse=True)

    # 1) 기울기 = 가상 손상의 1차 예측
    g_n = np.zeros(circuit.N)
    g_e = None
    base = []
    mos = [e for e in layer._effects if e.kind == "mosaic"]               # 세포 유형 드롭아웃은 학습용 - 설명할 때는 끔
    for e in mos:
        e.remove()
    try:
        for s in seeds:
            pn = Signal(xp.ones((circuit.N, 1), dtype=xp.float32), plastic=True)
            layer._probe = {"neuron": pn}
            if pathways:
                pe = Signal(xp.ones(len(layer.w_base), dtype=xp.float32), plastic=True)
                layer._probe["edge"] = pe
            val = score(layer, s)
            if not isinstance(val, Signal) or val.data.size != 1:
                raise TypeError("score(layer, seed)는 값 하나인 Signal을 돌려줘야 함 (예: out[:, k].sum())")
            base.append(float(val.data.reshape(-1)[0]))
            val.retrograde()
            g_n += B.numpy(pn.retro).reshape(-1) if pn.retro is not None else 0
            if pathways:
                ge = B.numpy(pe.retro) if pe.retro is not None else np.zeros(len(layer.w_base))
                g_e = ge if g_e is None else g_e + ge
            if verbose:
                from ._console import say
                say(f"  기울기 seed {s}: score {base[-1]:.4g}", flush=True)
    finally:
        layer._probe = {}
        layer._effects.extend(m for m in mos if m not in layer._effects)
    g_n /= len(seeds)
    base_mean = float(np.mean(base))
    drop = np.bincount(inv, weights=g_n, minlength=len(names))
    count = np.bincount(inv, minlength=len(names))
    tot = np.abs(drop).sum() or 1.0
    groups = pd.DataFrame({"name": names, "n": count, "pred_drop": drop, "share": drop / tot})
    groups = groups.reindex(groups.pred_drop.abs().sort_values(ascending=False).index).reset_index(drop=True)

    # 2) 경로
    paths = None
    if pathways:
        g_e = g_e / len(seeds)
        pre, post = B.numpy(layer.wiring.pre), B.numpy(layer.wiring.post)
        code = inv[pre] * len(names) + inv[post]
        uniq, ci = np.unique(code, return_inverse=True)
        pdrop = np.bincount(ci, weights=g_e)
        ptot = np.abs(pdrop).sum() or 1.0
        paths = pd.DataFrame({"pathway": [f"{names[c // len(names)]} > {names[c % len(names)]}" for c in uniq],
                              "edges": np.bincount(ci), "pred_drop": pdrop, "share": pdrop / ptot})
        paths = paths.reindex(paths.pred_drop.abs().sort_values(ascending=False).index).reset_index(drop=True)

    # 3) 실제로 꺼 보기
    verified = None
    if verify:
        # 예측이 큰 k개 + 나머지 중 예측이 0이 아닌 유형에서 순위를 고르게 k개 (0인 유형은 맞히기 쉬워 상관을 부풀림)
        top = list(groups.name[:verify])
        rest = groups[~groups.name.isin(top) & (groups.pred_drop.abs() > 1e-12 * (groups.pred_drop.abs().max() or 1))]
        spread = rest.name.iloc[np.unique(np.linspace(0, len(rest) - 1, min(verify, len(rest))).round().astype(int))]             if len(rest) else []
        pick = top + list(spread)
        rows = []
        with quiescent():
            b0 = np.mean([float(score(layer, s).data.reshape(-1)[0]) for s in seeds])
            for name in pick:
                line = G.Line(circuit, np.nonzero(labels.astype(str) == name)[0], name)
                with G.silence(layer, line):
                    v = np.mean([float(score(layer, s).data.reshape(-1)[0]) for s in seeds])
                rows.append(dict(name=name, n=len(line),
                                 pred_drop=float(groups.pred_drop[groups.name == name].iloc[0]), actual_drop=b0 - v))
                if verbose:
                    from ._console import say
                    say(f"  확인 {name}: 예측 {rows[-1]['pred_drop']:.3g}, 실제 {rows[-1]['actual_drop']:.3g}", flush=True)
        verified = pd.DataFrame(rows)
    return Explanation(base_mean, by, g_n, groups, paths, verified, seeds, G.active(layer))
