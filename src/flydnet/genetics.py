"""가상 유전학: 초파리 실험실의 방법 그대로 뉴런 집단을 골라(드라이버) 끄고·켜고·막고·없앤다

  line = fd.genetics.driver(brain, cell_type="MBON01")      # GAL4 드라이버: 주석으로 뉴런 집단 고르기
  line = fd.genetics.driver(brain, root_ids=[...])           # 뉴런 ID로 / group="KC" 회로 그룹으로
  a & b   (split-GAL4: 두 드라이버 모두에서 켜진 뉴런만),  a | b,  a - b

효과기 - ConnectomeLayer에 발현. with 블록 안에서만, 또는 .remove()를 부를 때까지:
  silence(layer, line)              Kir2.1: 발화하지 않음 (내보내는 신호·발화율 모두 0)
  block(layer, line)                Shibire-ts: 발화는 하지만 시냅스 전달이 막힘 (발화율은 그대로 보임).
                                    Shiu et al. 2024의 "silence"(나가는 시냅스 세기 0)가 이것
  activate(layer, line, hz=100)     CsChrimson·P2X2: 포아송 자극. Shiu et al. 2024와 같은 방식 (자극 하나 = 전위
                                    w_syn·f_poi 더하기, 자극받는 뉴런은 불응기 없음).
                                    연속값 뉴런(neuron="graded")은 level로 활동을 고정
  ablate(circuit, line)             세포 제거: 그 뉴런들의 연결을 모두 뺀 새 회로 (학습 비교용 - fd.compare와 함께)
  mosaic(layer, p=0.1, by=...)      세포 유형 드롭아웃 (유전 모자이크): 학습 중에만 시료마다 세포 유형을 통째로 무작위로 끔

  active(layer), clear(layer)       지금 켜져 있는 효과기 / 모두 끄기 (print(layer)에도 보임)
  lines(circuit, by="cell_type")    주석 값마다 드라이버 (GAL4 모음)
  screen(measure, layer, lines)     유전자 스크린: 집단마다 효과기를 발현해 측정값 변화·짝지은 p값 표

효과기는 시뮬레이션에만 적용되고 저장되지 않는다 (layer.save는 배선·학습값만).
"""
from __future__ import annotations

from . import _check as _C
import contextlib
import re

import numpy as np
import pandas as pd

from .ganglion import backend as B


# ─────────────── 드라이버 (뉴런 집단) ───────────────
class Line:
    """회로 안의 뉴런 집단 (회로 번호 idx). 드라이버 계통처럼 &, |, - 로 조합"""

    def __init__(self, circuit, idx, name: str):
        self.circuit = circuit
        self.idx = np.unique(np.asarray(idx, dtype=np.int64))
        self.name = name
        if len(self.idx) and (self.idx[0] < 0 or self.idx[-1] >= circuit.N):
            raise ValueError(f"{name}: 회로 밖 번호")

    def __len__(self):
        return len(self.idx)

    @property
    def root_ids(self) -> np.ndarray:
        return self.circuit.root_ids[self.idx]

    def _same(self, other: "Line"):
        if not isinstance(other, Line):
            return NotImplemented
        if not _same_circuit(self.circuit, other.circuit):
            raise ValueError("다른 회로의 드라이버끼리는 조합할 수 없음")

    def __and__(self, other):
        self._same(other)
        return Line(self.circuit, np.intersect1d(self.idx, other.idx), f"({self.name} ∩ {other.name})")

    def __or__(self, other):
        self._same(other)
        return Line(self.circuit, np.union1d(self.idx, other.idx), f"({self.name} ∪ {other.name})")

    def __sub__(self, other):
        self._same(other)
        return Line(self.circuit, np.setdiff1d(self.idx, other.idx), f"({self.name} - {other.name})")

    def __repr__(self):
        return f"<Line {self.name}: 뉴런 {len(self)}개>"


def _same_circuit(a, b) -> bool:
    return a is b or (a.N == b.N and np.array_equal(a.root_ids, b.root_ids))


def _match(values: pd.Series, want) -> np.ndarray:
    """주석 값 비교: 문자열·목록은 정확히 같은 값, re.compile(...)이면 정규식 전체 일치"""
    s = values.astype("string")
    if isinstance(want, re.Pattern):
        return s.str.fullmatch(want).fillna(False).to_numpy(bool)
    wants = list(want) if isinstance(want, (list, tuple, set, frozenset, np.ndarray, pd.Index, pd.Series)) else [want]
    raw = values.isin(wants).to_numpy(bool)                              # 같은 자료형끼리 (True, 3 == 3.0 등)
    text = s.isin([str(w) for w in wants]).fillna(False).to_numpy(bool)  # 문자열로 저장된 값 ('True', 예전 파일)
    return raw | text


def driver(circuit, group: str | None = None, root_ids=None, name: str | None = None, missing: str = "error",
           **annotation) -> Line:
    """뉴런 집단 고르기 (조건을 여러 개 주면 모두 만족하는 뉴런 = 교집합)

    group:      회로 그룹 이름 (예: "KC")
    root_ids:   FlyWire 뉴런 ID 목록. 회로에 없는 ID는 missing = "error" / "warn" / "ignore"
    annotation: 주석 열=값 (예: cell_type="MBON01", cell_sub_class=["sugar", "sugar/low_salt"], side="left").
                값이 re.compile("MBON.*")이면 정규식"""
    if group is None and root_ids is None and not annotation:
        raise ValueError("group, root_ids, 주석 조건 중 하나는 필요")
    keep = np.ones(circuit.N, bool)
    parts = []
    if group is not None:
        if group not in circuit.groups:
            raise KeyError(f"회로에 없는 그룹: {group} (있는 것: {list(circuit.groups)[:20]})")
        m = np.zeros(circuit.N, bool); m[circuit.groups[group]] = True
        keep &= m; parts.append(group)
    if root_ids is not None:
        ids = np.asarray(root_ids, dtype=np.int64)
        found = pd.Index(circuit.root_ids).isin(ids)
        lost = np.setdiff1d(ids, circuit.root_ids)
        if len(lost):
            msg = f"회로에 없는 뉴런 ID {len(lost)}개 (예: {lost[:3].tolist()}) - 다른 FlyWire 버전의 ID일 수 있음"
            if missing == "error":
                raise KeyError(msg + " (missing='warn'이면 빼고 진행)")
            if missing == "warn":
                import warnings
                warnings.warn(msg)
        keep &= found; parts.append(f"ID {len(ids) - len(lost)}개")
    if annotation:
        if circuit.meta is None:
            raise ValueError("주석이 없는 회로 (Circuit.from_flywire / whole_brain으로 만든 회로에서)")
        for col, want in annotation.items():
            if col not in circuit.meta.columns:
                raise KeyError(f"주석 열이 없음: {col} (있는 것: {list(circuit.meta.columns)})")
            keep &= _match(circuit.meta[col], want)
            parts.append(f"{col}={want.pattern if isinstance(want, re.Pattern) else want}")
    line = Line(circuit, np.nonzero(keep)[0], name or " & ".join(map(str, parts)))
    if not len(line):
        raise ValueError(f"조건에 맞는 뉴런이 없음: {line.name}")
    return line


def lines(circuit, by: str | None = None, min_size: int = 1, within: Line | None = None) -> dict[str, Line]:
    """주석 열 by의 값마다 드라이버 (GAL4 모음). by="group"이면 회로 그룹마다, None이면 cell_type 주석이 있으면 그것,
    없으면 group. within을 주면 그 집단 안에서만"""
    _C.integer('min_size', min_size)
    from .attribution import default_by
    by = by or default_by(circuit)
    if by == "group":
        vals = pd.Series(circuit.group_of(), dtype="string")
    elif circuit.meta is None or by not in circuit.meta.columns:
        raise KeyError(f"주석 열이 없음: {by}")
    else:
        vals = circuit.meta[by].astype("string")
    if within is not None:
        mask = np.zeros(circuit.N, bool); mask[within.idx] = True
        vals = vals.where(mask)
    out = {}
    for v, idx in vals.dropna().groupby(vals.dropna()).groups.items():
        if len(idx) >= min_size:
            out[str(v)] = Line(circuit, np.asarray(idx), str(v))
    return out


# ─────────────── 효과기 ───────────────
class Expression:
    """층에 발현된 효과기 하나. with 블록이 끝나거나 remove()를 부르면 사라짐"""

    def __init__(self, layer, kind: str, line: Line, hz: float | None = None, level: float | None = None):
        if not _same_circuit(layer.circuit, line.circuit):
            raise ValueError(f"{line.name}: 층의 회로와 다른 회로에서 고른 드라이버")
        if not len(line):
            raise ValueError(f"{line.name}: 뉴런이 없음")
        self.layer, self.kind, self.line, self.hz, self.level = layer, kind, line, hz, level
        if kind == "activate":
            ins = np.intersect1d(line.idx, B.numpy(layer.in_idx))
            if len(ins):
                raise ValueError(f"{line.name}: 입력 그룹 뉴런 {len(ins)}개는 활성화 대신 입력 발화율로 조절")
            for e in layer._effects:
                if e.kind == "activate" and len(np.intersect1d(e.line.idx, line.idx)):
                    raise ValueError(f"{line.name}: 이미 활성화한 뉴런과 겹침 ({e.line.name})")
        layer._effects.append(self)

    def remove(self):
        if self in self.layer._effects:
            self.layer._effects.remove(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __repr__(self):
        extra = f", {self.hz} Hz" if self.hz is not None else (f", level {self.level}" if self.level is not None else "")
        if self.kind == "mosaic":
            extra = f", p {self.p}, by {self.by}"
        return f"<{self.kind} {self.line.name} ({len(self.line)}개{extra})>"


def active(layer) -> list:
    """층에 지금 켜져 있는 효과기 목록"""
    return list(getattr(layer, "_effects", ()))


def clear(layer):
    """층의 효과기를 모두 끔"""
    layer._effects.clear()


def silence(layer, line: Line) -> Expression:
    """Kir2.1: 발화하지 않음 - 내보내는 신호와 발화율 모두 0 (입력 그룹도 끌 수 있음)"""
    return Expression(layer, "silence", line)


def block(layer, line: Line) -> Expression:
    """Shibire-ts: 발화는 그대로(발화율에 보임), 시냅스 전달만 막힘. Shiu et al. 2024 모델의 silence와 같음"""
    return Expression(layer, "block", line)


def activate(layer, line: Line, hz: float = 100.0, level: float | None = None) -> Expression:
    """CsChrimson·P2X2: 스파이킹 뉴런에 hz의 포아송 자극 (자극 하나 = 입력 스파이크 하나, 불응기 없음 - Shiu et al. 2024).
    연속값 뉴런(graded)은 활동을 level로 고정 (기본 1.0)"""
    _C.optional(_C.finite, 'level', level)
    _C.pos("hz", hz)                                                 # NaN·무한대면 자극이 조용히 사라짐
    if layer.neuron == "graded":
        level = 1.0 if level is None else level
    return Expression(layer, "activate", line, hz=float(hz), level=level)


def mosaic(layer, p: float = 0.1, by: str | None = None, within: Line | None = None, rescale: bool = True) -> Expression:
    """세포 유형 드롭아웃 (유전 모자이크): 학습 중에만, 시료마다 세포 유형(by)을 확률 p로 통째로 끔 (Kir2.1과 같은 조작).
    한 세포 유형에만 기대지 않게 → 세포 유형 손상(수용체 결손 등)에 강한 모델. 평가(quiescent) 때는 꺼짐

    by:      주석 열 (예: "cell_type"), "group", 또는 "neuron" (뉴런마다 따로 = 보통 드롭아웃, 비교용).
             None이면 cell_type 주석이 있으면 그것, 없으면 group
    within:  이 집단 안에서만 (예: driver(mb, group="PN")). None이면 회로 전체
    rescale: 남은 세포의 출력을 1/(1-p)배 (보통 드롭아웃처럼 학습·평가 때의 평균 입력을 맞춤)
    난수는 순전파의 seed로 정해짐 (같은 seed = 같은 모자이크, 체크포인팅으로 다시 계산해도 같음)"""
    if not 0 <= p < 1:
        raise ValueError(f"p는 0 이상 1 미만: {p}")
    c = layer.circuit
    line = within if within is not None else Line(c, np.arange(c.N), "전체")
    from .attribution import default_by
    by = by or default_by(c)
    if by == "neuron":
        inv = np.arange(c.N)
    else:
        from .attribution import _labels
        _, inv = np.unique(_labels(c, by).astype(str), return_inverse=True)
    e = Expression(layer, "mosaic", line)
    e.p, e.by, e.rescale = float(p), by, rescale
    member = np.zeros(c.N, bool); member[line.idx] = True
    _, e.inv = np.unique(inv[member], return_inverse=True)        # 집단 안의 유형만 번호 매김
    e.member = np.nonzero(member)[0]
    e.n_types = int(e.inv.max()) + 1 if len(e.inv) else 0
    return e


_TRAINING = [0]


@contextlib.contextmanager
def training():
    """역행성 경로 없이(quiescent) 학습하는 규칙(ThreeFactor·STDP)과 gradcheck의 수치 미분이 mosaic을 켜 두게.
    없으면 mosaic이 학습 중인지 learning_enabled()로만 판단해 이 경우 조용히 꺼짐"""
    _TRAINING[0] += 1
    try:
        yield
    finally:
        _TRAINING[0] -= 1


def mosaic_mask(layer, seed: int, batch: int):
    """순전파용 모자이크 마스크 (N, batch) 또는 None (학습 중이 아니거나 mosaic이 없으면)"""
    from .ganglion.signal import learning_enabled
    mos = [e for e in getattr(layer, "_effects", ()) if e.kind == "mosaic" and e.p > 0]
    if not mos or not (learning_enabled() or _TRAINING[0]):
        return None
    rng = np.random.default_rng([int(seed) % (1 << 63), 0x3051C])
    m = np.ones((layer.circuit.N, batch), np.float32)
    for e in mos:
        keep = (rng.random((e.n_types, batch)) >= e.p).astype(np.float32)
        if e.rescale:
            keep /= (1 - e.p)
        m[e.member] *= keep[e.inv]
    return B.to(m, layer.device)


def ablate(circuit, line: Line):
    """세포 제거: line 뉴런이 주고받는 연결을 모두 뺀 새 회로 (뉴런 번호·그룹은 그대로 → 같은 층 설정으로 비교)"""
    if not _same_circuit(circuit, line.circuit):
        raise ValueError("다른 회로에서 고른 드라이버")
    gone = np.zeros(circuit.N, bool); gone[line.idx] = True
    keep = ~(gone[circuit.pre] | gone[circuit.post])
    from .circuit import Circuit
    return Circuit(circuit.root_ids, circuit.groups, circuit.pre[keep], circuit.post[keep], circuit.weight[keep],
                   name=f"{circuit.name} [제거: {line.name}]", meta=circuit.meta, pos=circuit.pos)


def effects(layer) -> dict:
    """forward가 쓰는 효과기 요약 (내부용): 뉴런별 발화 마스크·전달 마스크, 활성화 뉴런·확률·수준"""
    eff = list(getattr(layer, "_effects", ()))
    if not eff:
        return {}
    N = layer.circuit.N
    out = {}
    for kind in ("silence", "block"):
        idx = [e.line.idx for e in eff if e.kind == kind]
        if idx:
            m = np.ones((N, 1), np.float32); m[np.concatenate(idx)] = 0
            out[kind] = B.to(m, layer.device)
    acts = [e for e in eff if e.kind == "activate"]
    if acts:
        idx = np.concatenate([e.line.idx for e in acts])
        out["act_idx"] = B.to(idx, layer.device)
        out["act_hz"] = B.to(np.concatenate([np.full(len(e.line), e.hz, np.float32) for e in acts])[:, None], layer.device)
        out["act_level"] = B.to(np.concatenate([np.full(len(e.line), e.level if e.level is not None else 1.0, np.float32)
                                                for e in acts])[:, None], layer.device)
    return out


# ─────────────── 유전자 스크린 ───────────────
def screen(measure, layer, lines_: dict, effector: str = "silence", seeds=5, hz: float = 100.0,
           verbose: bool = True) -> pd.DataFrame:
    """집단마다 효과기를 발현하고 측정값이 얼마나 바뀌는지 (같은 seed끼리 짝지음)

    measure(layer, seed) -> float   예: lambda L, s: L(None, seed=s, return_all=True)[:, mn9.idx].mean()
    lines_:   {이름: Line} (lines(...)의 결과 등)
    effector: "silence" / "block" / "activate"
    반환 표: 집단, 뉴런 수, 기준 평균, 조작 평균, 변화, 변화 비율, p (부호 뒤집기 순열 검정),
            p_holm (집단 수만큼 여러 번 시험한 것을 보정), 변화가 큰 순. seed 6개 미만이면 p < 0.05가 불가능해 경고"""
    if hasattr(seeds, "__len__") and not len(seeds):
        raise ValueError("seeds 목록이 비어 있음")
    if not hasattr(seeds, "__len__"):
        _C.integer("seeds", seeds)
    _C.pos('hz', hz)
    from ._console import say
    from .controls import sign_flip_p
    make = {"silence": silence, "block": block, "activate": lambda L, l: activate(L, l, hz=hz)}
    if effector not in make:
        raise ValueError(f"effector는 {list(make)} 중 하나")
    import warnings
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    min_p = 2 / 2 ** len(seeds)                                     # 부호 뒤집기 검정이 낼 수 있는 가장 작은 p
    if min_p > 0.05:
        warnings.warn(f"seed {len(seeds)}개로는 p가 {min_p:.3g} 아래로 내려갈 수 없음 - 효과가 커도 유의하지 않게 나옴. "
                      "seeds=6 이상 (p < 0.05가 가능한 최소)", stacklevel=2)
    for name, line in lines_.items():                               # 오래 돌기 전에 붙일 수 있는지 (입력 뉴런 활성화 등)
        try:
            make[effector](layer, line).remove()
        except ValueError as e:
            raise ValueError(f"screen: 집단 '{name}'에 {effector}를 발현할 수 없음 - {e}") from None
    if active(layer):
        warnings.warn(f"층에 이미 켜진 효과기가 있음 ({active(layer)}) - 기준·조작 모두에 적용됨", stacklevel=2)
    base = np.array([float(measure(layer, s)) for s in seeds])
    if not np.isfinite(base).all():
        raise ValueError("기준 측정값에 NaN·무한대")
    if len(seeds) > 1 and np.ptp(base) == 0 and base[0] == 0:
        warnings.warn("기준 측정값이 모두 0 - 줄이는 조작의 효과는 볼 수 없음 (자극이 충분한지 확인)", stacklevel=2)
    rows = []
    for i, (name, line) in enumerate(lines_.items()):
        with make[effector](layer, line):
            val = np.array([float(measure(layer, s)) for s in seeds])
        d = val - base
        rows.append(dict(line=name, n=len(line), baseline=base.mean(), manipulated=val.mean(), change=d.mean(),
                         rel_change=d.mean() / base.mean() if base.mean() else np.nan,
                         p=sign_flip_p(d) if len(seeds) > 1 else np.nan))
        if verbose:
            say(f"  [{i + 1}/{len(lines_)}] {name} ({len(line)}개): {base.mean():.3g} → {val.mean():.3g}", flush=True)
    df = pd.DataFrame(rows)
    df = df.reindex(df.change.abs().sort_values(ascending=False).index).reset_index(drop=True)
    df.attrs["min_p"] = min_p
    if len(df) > 1:                                                 # 여러 집단을 시험하면 우연히 작은 p가 나옴
        df["p_holm"] = _holm(df.p.to_numpy())
    return df


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm 보정 (여러 집단을 한꺼번에 시험할 때의 p)"""
    adj = np.full(len(p), np.nan)
    ok = np.nonzero(np.isfinite(p))[0]                               # p가 없는(NaN) 집단은 빼고 보정
    run = 0.0
    for k, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, min(1.0, (len(ok) - k) * p[i]))
        adj[i] = run
    return adj
