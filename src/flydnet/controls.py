"""대조 실험: "이 배선이 정말 중요한가?"를 통계로 묻기

    def run(circuit, seed):                    # 회로 하나로 모델을 만들고 학습해서 점수 하나 (높을수록 좋음)
        ...
        return accuracy

    report = fd.compare(run, mb, controls=["shuffled", "randomized"], seeds=5, chance=0.125)
    print(report)                              # 조건별 평균 ± 95% 신뢰구간, 실제 - 대조군 차이, 효과 크기, p값, 경고
    report.table()                             # pandas 표

같은 seed끼리 짝을 지음: seed i에서 실제 회로와, seed i로 만든 대조군 회로를 같은 학습 seed i로 돌림.
검정은 분포 가정이 없는 부호 뒤집기 순열 검정 (짝 차이의 부호를 무작위로 뒤집은 평균과 비교, 양측).

대조군 (fd.controls):
  "shuffled"          Shuffled()        그룹 쌍별 연결 수·뉴런별 연결 수 유지, 상대만 섞음 → "누가 누구와"가 중요한가
  "randomized"        Randomized()      그룹 쌍별 연결 수만 유지, 차수 분포도 무작위 → 허브·차수 구조가 중요한가
  "shuffled_weights"  ShuffledWeights() 배선 그대로, 시냅스 세기만 섞음 → 세기 분포가 중요한가
  (객체만)            Local(xy, r)      시야 위치(칸 r) 안에서만 섞음 → 큰 구조 말고 세부 배선이 중요한가
  함수 f(circuit, seed) → Circuit       직접 만든 대조군
"""
from __future__ import annotations

from . import _check as _C
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ._console import say


# ─────────────── 대조군 ───────────────
class Control:
    """대조군 기본 클래스: build(circuit, seed) → Circuit"""
    name = "control"
    question = ""

    def build(self, circuit, seed: int):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Shuffled(Control):
    question = "누가 누구와 연결되는가 (연결 수·차수는 같게)"

    def __init__(self, pairs=None, exclude=None, name: str = "shuffled"):
        self.pairs, self.exclude, self.name = pairs, exclude, name

    def build(self, circuit, seed):
        return circuit.shuffled(seed=seed, pairs=self.pairs, exclude=self.exclude)


class Randomized(Control):
    name = "randomized"
    question = "연결 수 분포(허브·차수 구조)까지"

    def build(self, circuit, seed):
        return circuit.randomized(seed=seed)


class ShuffledWeights(Control):
    name = "shuffled_weights"
    question = "시냅스 세기(시냅스 수) 분포"

    def build(self, circuit, seed):
        return circuit.shuffled_weights(seed=seed)


class Local(Control):
    """위치 대응을 유지한 국소 무작위: 받는 뉴런 위치 xy를 radius 칸으로 나눠 같은 칸 안에서만 섞음.
    merge={그룹: 별칭}이면 아형을 합쳐 섞음 (예: T4a~d → T4: 아형별 방향 구조까지 지움)"""
    question = "세부 배선 (큰 위치 구조는 같게)"

    def __init__(self, xy, radius: float = 2.0, merge=None, name: str | None = None):
        _C.pos('radius', radius)
        self.xy, self.radius, self.merge = np.asarray(xy), radius, merge
        self.name = name or f"local(r={radius:g}{', merge' if merge else ''})"

    def build(self, circuit, seed):
        if self.xy.shape != (circuit.N, 2):
            raise ValueError(f"Local: xy 모양 {self.xy.shape} ≠ (회로 뉴런 수 {circuit.N}, 2) - "
                             "이 회로로 만든 시야 좌표(fd.column_map(circuit))를 줄 것")
        return circuit.shuffled(seed=seed, local=(self.xy, self.radius), merge=self.merge)


class Custom(Control):
    question = "사용자 정의"

    def __init__(self, fn, name: str | None = None, question: str | None = None):
        """fn(circuit, seed) → 대조군 회로. question: 결과표에 나올 '묻는 것' 설명"""
        self.fn, self.name = fn, name or getattr(fn, "__name__", "custom")
        if question is not None:
            self.question = question

    def build(self, circuit, seed):
        return self.fn(circuit, seed)


_NAMED = {"shuffled": Shuffled, "randomized": Randomized, "random_sparse": Randomized,
          "shuffled_weights": ShuffledWeights}


def _as_control(c) -> Control:
    if isinstance(c, Control):
        return c
    if isinstance(c, str):
        if c == "local":
            raise ValueError("'local'은 시야 좌표가 필요함: fd.controls.Local(xy, radius=2)")
        if c not in _NAMED:
            raise ValueError(f"모르는 대조군: {c} (가능: {sorted(_NAMED)}, 또는 Local(xy, r), 함수)")
        return _NAMED[c]()
    if callable(c):
        return Custom(c)
    raise TypeError(f"대조군은 이름·Control·함수: {c!r}")


# ─────────────── 통계 ───────────────
def sign_flip_p(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """짝 차이 d의 평균이 0인지 부호 뒤집기 순열 검정 (양측). n ≤ 14면 모든 경우를 셈 (정확한 p)"""
    d = np.asarray(d, float)
    if d.ndim == 2 and d.shape[1] == 1:                                  # (n, 1) 열 벡터
        d = d[:, 0]
    if d.ndim != 1:
        raise ValueError(f"짝 차이 d는 1차원 (seed마다 하나): 모양 {d.shape}")
    if not np.isfinite(d).all():                                         # 예전: NaN이 있으면 평균이 NaN → 비교가 모두 거짓 →
        raise ValueError(f"짝 차이에 NaN·무한대: {d[~np.isfinite(d)][:5].tolist()} - 실패한 seed를 확인할 것")   # p = 0 (유의)
    n = len(d)
    if n == 0 or not np.any(d):                                     # 단위와 상관없이 (allclose는 1e-8보다 작은 점수를 0으로 봄)
        return 1.0
    obs = abs(d.mean())
    if n <= 14:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
    null = np.abs((signs * d).mean(1))
    hits = int((null >= obs * (1 - 1e-9)).sum())                         # 같은 값의 반올림 차이만 허용 (상대)
    if n <= 14:                                                          # 모든 경우 (관측한 부호 그대로도 포함) → 정확한 p
        return hits / len(null)
    return (hits + 1) / (len(null) + 1)                                  # 표본: 관측값도 한 경우로 셈 - 예전에는 p = 0이 나올 수 있었음


# t 분포 97.5% 분위수: 자유도 1 ~ 30은 표 (소수 여섯째 자리 - 셋째 자리 표는 차이가 최대 5e-4라 신뢰구간의 셋째 자리가
# 반올림 경계에서 바뀌었음), 30 넘으면 Cornish-Fisher 전개 (scipy.stats.t.ppf와 차이 2e-6 이하)
_T975 = [12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
         2.200985, 2.178813, 2.160369, 2.144787, 2.131450, 2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
         2.079614, 2.073873, 2.068658, 2.063899, 2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272]


def t975(df: int) -> float:
    """t 분포의 97.5% 분위수 (scipy.stats.t.ppf(0.975, df) 대신 - scipy 없이)"""
    if df <= 30:
        return _T975[df - 1]
    z = 1.959963984540054
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def _ci95(x: np.ndarray):
    """평균의 95% 신뢰구간 (t 분포). 값이 하나면 (nan, nan)"""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (math.nan, math.nan)
    h = t975(len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


# ─────────────── 보고서 ───────────────
@dataclass
class CompareReport:
    scores: dict                                   # 조건 이름 → seed별 점수 (numpy)
    seeds: list
    controls: list
    chance: float | None = None
    higher_is_better: bool = True
    seconds: float = 0.0
    warnings: list = field(default_factory=list)

    def table(self):
        """조건별 요약 표 (pandas): 평균, 표준편차, 95% CI, 실제 - 대조군 차이와 그 CI, 효과 크기 d, p, 실제가 이긴 seed 수"""
        import pandas as pd
        real = self.scores["real"]
        rows = []
        for name, s in self.scores.items():
            lo, hi = _ci95(s)
            row = dict(condition=name, mean=s.mean(), sd=s.std(ddof=1) if len(s) > 1 else math.nan,
                       ci_low=lo, ci_high=hi)
            if name != "real":
                d = real - s if self.higher_is_better else s - real
                dlo, dhi = _ci95(d)
                sd = d.std(ddof=1) if len(d) > 1 else math.nan
                row.update(diff=d.mean(), diff_ci_low=dlo, diff_ci_high=dhi,
                           effect_d=(d.mean() / sd if sd > 0 else math.copysign(math.inf, d.mean()) if d.mean() else 0.0)
                           if not math.isnan(sd) else math.nan,
                           p=sign_flip_p(d), real_wins=f"{int((d > 0).sum())}/{len(d)}")
            rows.append(row)
        return pd.DataFrame(rows).set_index("condition")

    def verdict(self, name: str, alpha: float = 0.05) -> str:
        r = self.table().loc[name]
        if r.p < alpha and r["diff"] > 0:
            return "실제 배선이 더 좋음"
        if r.p < alpha and r["diff"] < 0:
            return "대조군이 더 좋음"
        return "차이를 확인하지 못함"

    # 대조군 사다리: (종류, 이 단계가 바로 아래 단계보다 더 유지하는 구조)
    _LADDER = (("randomized", "그룹 쌍별 연결 수"),
               ("shuffled", "뉴런별 연결 수 분포 (차수·허브)"),
               ("local", "시야 위치 대응 같은 큰 공간 구조"),
               ("real", "세부 배선 (위치 안에서 누가 정확히 누구와)"))

    def interpret(self, alpha: float = 0.05) -> list:
        """대조군의 포함 관계로 원인 좁히기: randomized ⊂ shuffled ⊂ local ⊂ 실제 배선.
        실제 배선이 아래 단계는 이기고 위 단계와는 차이가 없으면, 그 사이 단계들이 더한 구조가 원인.
        비교하지 않은 단계가 사이에 있으면 그 구조들도 후보 - 예전에는 바로 위 단계의 구조만 말해서, shuffled만
        비교하고 이기면 위치 구조일 수도 있는데 '세부 배선'이 원인이라고 단정했음"""
        t = self.table()
        kind = {Randomized: "randomized", Local: "local"}
        present = {}
        for c in self.controls:
            k = "shuffled" if (type(c) is Shuffled and c.pairs is None and c.exclude is None) else kind.get(type(c))
            if k and k not in present:
                present[k] = c.name
        present["real"] = "실제 배선"
        level = [k for k, _ in self._LADDER]
        adds = dict(self._LADDER)
        steps = [k for k in level if k in present]
        lose = lambda k: k != "real" and t.loc[present[k]].p < alpha and t.loc[present[k]]["diff"] > 0
        out = []
        for lo, hi in zip(steps, steps[1:]):
            if not lose(lo) or lose(hi):
                continue
            between = [adds[k] for k in level[level.index(lo) + 1:level.index(hi) + 1]]
            what = between[0] if len(between) == 1 else "다음 중 하나 이상 - " + " / ".join(between)
            if hi == "real":                                                 # 가장 많이 유지한 대조군도 짐
                out.append(f"실제 배선이 {present[lo]}도 이김 → 중요한 구조: {what}")
            else:
                out.append(f"실제 배선이 {present[lo]}는 이기고 {present[hi]}와는 차이가 없음 → 중요한 구조: {what}")
        win = [c for c in self.controls if t.loc[c.name].p < alpha and t.loc[c.name]["diff"] < 0]
        for c in win:                                                      # 대조군이 실제 배선보다 좋음
            out.append(f"{c.name}가 실제 배선보다 좋음 → 실제 배선의 이 구조({c.question})가 이 과제에는 오히려 불리 "
                       "(과제가 그 회로가 실제로 하는 일과 다르면 흔함)")
        if len(steps) > 1 and not any(lose(k) for k in steps) and not win:
            out.append("어느 대조군과도 차이를 확인하지 못함 → 이 과제에서 배선 구조의 이점은 보이지 않음")
        return out

    def __str__(self):
        t = self.table()
        lines = [f"대조 실험: seed {len(self.seeds)}개, {self.seconds:.0f}초" +
                 (f", 찍기 수준 {self.chance:g}" if self.chance is not None else ""), ""]
        lines.append(f"{'조건':<26}{'평균':>9}{'95% CI':>20}{'실제 - 대조':>13}{'d':>7}{'p':>8}{'실제 우세':>9}")
        for name, r in t.iterrows():
            ci = f"[{r.ci_low:.4g}, {r.ci_high:.4g}]" if not math.isnan(r.ci_low) else "-"
            if name == "real":
                lines.append(f"{'real (실제 배선)':<26}{r['mean']:>9.4g}{ci:>20}")
            else:
                lines.append(f"{name:<26}{r['mean']:>9.4g}{ci:>20}{r['diff']:>+13.4g}{r.effect_d:>7.2f}"
                             f"{r.p:>8.3f}{r.real_wins:>9}")
        lines.append("")
        for c in self.controls:
            lines.append(f"  · {c.name}: {self.verdict(c.name)}  - 묻는 것: {c.question}")
        interp = self.interpret()
        if interp:
            lines += ["", "해석 (대조군 포함 관계):"] + [f"  → {x}" for x in interp]
        if self.warnings:
            lines += ["", "경고:"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)

    __repr__ = __str__

    def to_dict(self) -> dict:
        return dict(scores={k: v.tolist() for k, v in self.scores.items()}, seeds=list(self.seeds),
                    controls=[c.name for c in self.controls], chance=self.chance, warnings=list(self.warnings),
                    table=self.table().reset_index().to_dict(orient="records"))


def _diagnose(rep: CompareReport, check_repeat, ceiling, floor_margin):
    """실험에서 실제로 빠졌던 함정을 경고로"""
    w = rep.warnings
    n = len(rep.seeds)
    min_p = 2 / 2 ** n if n else 1.0                                      # 부호 뒤집기 검정이 낼 수 있는 가장 작은 p
    if min_p > 0.05:
        w.append(f"seed {n}개로는 p가 {min_p:.3g} 아래로 내려갈 수 없음 - 효과가 아무리 커도 '차이를 확인하지 못함'으로 "
                 "나옴. p < 0.05를 보려면 seed 6개 이상")
    elif n < 5:
        w.append(f"seed가 {n}개뿐 - 우연과 구별하기 어려움. 6개 이상 권장")
    all_scores = np.concatenate(list(rep.scores.values()))
    if not rep.higher_is_better:                                         # 손실처럼 낮을수록 좋은 점수: 상한·찍기 경고 안 함
        ceiling = None
    if ceiling is not None and ((all_scores < 0) | (all_scores > 1)).any():   # 정확도(0~1)가 아닌 점수 (발화율 Hz·보상 등):
        ceiling = None                                                   # 0.99 상한이 늘 넘어 '너무 쉬움'으로 잘못 경고했음
    if ceiling is not None and (all_scores >= ceiling).all():
        w.append(f"실제·대조군 모두 {ceiling:g} 이상 - 과제가 너무 쉬워 배선 차이가 드러나지 않음 "
                 "(예: 수천 뉴런 평균 같은 지름길). 더 어려운·국소적인 과제로")
    elif ceiling is not None and rep.scores["real"].mean() >= ceiling - 0.05:
        t0 = rep.table()
        if all(t0.loc[c.name].p >= 0.05 for c in rep.controls):
            w.append(f"실제 배선 점수가 상한({ceiling:g})에 가까움 ({rep.scores['real'].mean():.3g}) - 차이가 없다고 나왔어도 "
                     "상한에 가려졌을 수 있음. 더 어려운 과제(클래스 늘리기·잡음 키우기·학습 데이터 줄이기)로 다시")
    if rep.higher_is_better and rep.chance is not None and (all_scores <= rep.chance + floor_margin).all():
        w.append(f"모든 조건이 찍기 수준({rep.chance:g}) - 아무것도 학습되지 않음. 학습 설정부터 확인")
    real = rep.scores["real"]
    if n >= 2 and np.ptp(real) == 0 and not (rep.chance is not None and real[0] <= rep.chance):
        w.append("실제 배선 점수가 seed마다 똑같음 - run이 seed를 쓰지 않는지 확인 (반복의 의미가 없음)")
    if check_repeat is not None and not np.isclose(check_repeat[0], check_repeat[1], rtol=1e-6, atol=1e-9):
        w.append(f"같은 회로·같은 seed를 다시 돌렸더니 점수가 다름 ({check_repeat[0]:.6g} → {check_repeat[1]:.6g}) - "
                 "결과가 재현되지 않음 (GPU 연산 순서·전역 난수). 학습이 불안정하면 차이가 크게 벌어질 수 있음")
    t = rep.table()
    names = [c.name for c in rep.controls]
    win = lambda c: t.loc[c.name].p < 0.05 and t.loc[c.name]["diff"] > 0
    kinds = {type(c) for c in rep.controls}
    shuffled_wins = [c.name for c in rep.controls if isinstance(c, Shuffled) and c.pairs is None
                     and c.exclude is None and win(c)]
    if shuffled_wins and Local not in kinds:
        w.append(f"실제 배선이 {', '.join(shuffled_wins)}를 이겼지만, 원인이 시야 위치 대응 같은 큰 공간 구조일 수 있음 - "
                 "위치를 유지한 대조군(fd.controls.Local)으로 세부 배선을 따로 확인할 것 (실험 ⑩)")
    rand_wins = [c.name for c in rep.controls if isinstance(c, Randomized) and win(c)]
    if rand_wins and Shuffled not in kinds:
        w.append(f"실제 배선이 {', '.join(rand_wins)}를 이겼지만, 연결 수 분포(차수) 때문인지 연결 상대 때문인지 모름 - "
                 "차수를 유지한 'shuffled'도 같이 비교할 것")
    big = [nm for nm in names if t.loc[nm].p >= 0.05 and abs(t.loc[nm].effect_d) >= 1 and n < 10]
    if big:
        w.append(f"{', '.join(big)}: 효과 크기는 큰데(|d| ≥ 1) 유의하지 않음 - seed를 늘리면 확인될 수 있음")
    elif len(names) and all(t.loc[nm].p >= 0.05 for nm in names) and n < 6:
        w.append("차이를 확인하지 못했지만 seed가 적음 - '차이 없음'의 근거로는 약함")


def compare(run, circuit, controls=("shuffled",), seeds=5, chance: float | None = None,
            higher_is_better: bool = True, ceiling: float | None = 0.99, floor_margin: float = 0.02,
            check_repeat: bool = True, verbose: bool = True) -> CompareReport:
    """실제 배선 대 대조군 배선을 같은 학습 절차로 여러 seed 비교

    run:       run(circuit, seed) → 점수 (float). 학습 난수에 seed를 쓸 것
    controls:  이름("shuffled", "randomized", "shuffled_weights"), Control 객체(Local 등), 함수 f(circuit, seed)
    seeds:     정수 n (0..n-1) 또는 seed 목록
    chance:    찍기 수준 (주면 '아무것도 못 배움' 경고)
    ceiling:   모든 점수가 이 이상이면 '과제가 너무 쉬움' 경고 (None이면 안 함)
    check_repeat: 실제 회로·첫 seed를 한 번 더 돌려 재현되는지 확인 (실행 한 번 추가)
    """
    _C.optional(_C.finite, 'chance', chance)
    _C.optional(_C.unit, 'ceiling', ceiling)
    _C.nonneg('floor_margin', floor_margin)
    controls = [_as_control(c) for c in controls]
    if not controls:
        raise ValueError("대조군이 없음 - 예: controls=['shuffled', 'randomized']")
    names = [c.name for c in controls]
    if len(set(names)) != len(names):
        raise ValueError(f"대조군 이름이 겹침: {names} (name=으로 구분)")
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    if len(set(seeds)) != len(seeds):                               # 같은 seed = 같은 짝을 두 번 세어 p가 작아짐 (유사 반복)
        raise ValueError(f"seeds에 같은 값이 있음: {seeds} - 짝마다 다른 seed")
    if len(seeds) < 2:
        raise ValueError(f"seeds는 2개 이상 (짝지은 검정) - p < 0.05가 가능하려면 6개 이상: {len(seeds)}")
    t0 = time.time()
    scores = {"real": []}
    scores.update({c.name: [] for c in controls})
    def call(name, circ, s):
        try:
            v = float(run(circ, s))
        except Exception as e:
            raise RuntimeError(f"compare: 조건 '{name}', seed {s}에서 run이 실패함 ({type(e).__name__}: {e})") from e
        if not np.isfinite(v):
            raise ValueError(f"compare: 조건 '{name}', seed {s}에서 run이 {v}를 돌려줌 (학습 발산 또는 점수 계산 확인)")
        return v

    for i, s in enumerate(seeds):
        scores["real"].append(call("real", circuit, s))
        for c in controls:
            try:
                circ = c.build(circuit, s)
            except Exception as e:
                msg = f"compare: 대조군 '{c.name}'을 seed {s}로 만들지 못함 ({type(e).__name__}: {e})"
                kind = type(e) if isinstance(e, (ValueError, TypeError, KeyError, IndexError)) else RuntimeError
                raise kind(msg) from e                       # 인자 오류는 종류를 유지 (except ValueError로 잡히게)
            scores[c.name].append(call(c.name, circ, s))
        if verbose:
            parts = ", ".join(f"{k} {v[-1]:.4g}" for k, v in scores.items())
            say(f"  seed {s} ({i + 1}/{len(seeds)}): {parts}  [{time.time() - t0:.0f}s]", flush=True)
    rep_pair = None
    if check_repeat and seeds:
        rep_pair = (scores["real"][0], call("real (재현 확인)", circuit, seeds[0]))
    rep = CompareReport(scores={k: np.array(v) for k, v in scores.items()}, seeds=seeds, controls=controls,
                        chance=chance, higher_is_better=higher_is_better, seconds=time.time() - t0)
    _diagnose(rep, rep_pair, ceiling, floor_margin)
    return rep
