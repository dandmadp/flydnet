"""시험용 데이터 (자체 엔진판, torch 없음, 결과는 numpy). torch판(flydnet.torch)과 난수 생성기가 달라 값은 다름"""
from __future__ import annotations

from . import _check as _C
import numpy as np


def synthetic_odors(n_classes: int, n_glomeruli: int, n_train: int, n_test: int, protos_per_class: int = 1,
                    active_frac: float = 0.2, noise: float = 0.5, add_noise: float = 0.1, seed: int = 0):
    """합성 냄새 분류 과제

    냄새 원형: 사구체마다 active_frac 확률로 켜짐, 세기 U(0.2, 1)
    클래스:   원형 protos_per_class개의 묶음. 2개 이상이면 사구체 공간에서 선형 분리가 어려워짐
    샘플:     클래스의 원형 하나 × exp(N(0, noise)) + |N(0, add_noise)|
    반환: (Xtr, ytr, Xte, yte), X는 (n, n_glomeruli) float32, 클래스당 n_train / n_test 개
    """
    _C.integer('n_classes', n_classes)
    _C.integer('n_glomeruli', n_glomeruli)
    _C.nonneg('noise', noise)
    rng = np.random.default_rng(seed)
    P = n_classes * protos_per_class
    on = rng.random((P, n_glomeruli)) < active_frac
    protos = on * (0.2 + 0.8 * rng.random((P, n_glomeruli)))

    def sample(n):
        y = np.repeat(np.arange(n_classes), n)
        k = y * protos_per_class + rng.integers(0, protos_per_class, len(y))
        x = protos[k] * np.exp(noise * rng.normal(size=(len(y), n_glomeruli)))
        x = x + np.abs(add_noise * rng.normal(size=(len(y), n_glomeruli)))
        return x.astype(np.float32), y

    Xtr, ytr = sample(n_train)
    Xte, yte = sample(n_test)
    return Xtr, ytr, Xte, yte


# 같은 계열인데 이름이 다른 DoOR 화학 계열
_CLASS_ALIASES = {"arom": "aromatics", "terpenes": "terpene", "sulfid": "sulfide"}


def door_odors(glomeruli, data_dir=None, min_measured: int = 20):
    """DoOR 2.0 실제 냄새 반응 → 사구체 벡터 (Münch & Galizia 2016, CC BY-SA 4.0)

    glomeruli: 사구체 이름 순서 (예: GlomerularEncoder.glomeruli)
    반환 dict: X (n_odors, n_glomeruli) 반응 - 자발 발화(SFR), 0 아래는 0 / measured 측정 여부 /
              names, classes (화학 계열), inchikey. min_measured 이상 사구체가 측정된 냄새만
    """
    import pandas as pd
    from .data import require

    d = require("door", data_dir)
    R = pd.read_csv(d / "door_response_matrix.csv", sep=";")
    M = pd.read_csv(d / "door_mappings.csv", sep=";")
    O = pd.read_csv(d / "odor.csv", sep=";").drop_duplicates("InChIKey").set_index("InChIKey")
    sfr = R.loc["SFR"]
    R = R.drop(index="SFR")
    col = {g: j for j, g in enumerate(glomeruli)}
    X = np.zeros((len(R), len(glomeruli)), np.float32)
    meas = np.zeros_like(X, dtype=bool)
    for rec, glo in M[["receptor", "code"]].dropna().itertuples(index=False):
        if rec in R.columns and glo in col:
            v = R[rec].values
            ok = ~np.isnan(v)
            X[ok, col[glo]] = np.maximum(v[ok] - sfr[rec], 0)
            meas[ok, col[glo]] = True
    keep = meas.sum(1) >= min_measured
    info = O.reindex(R.index[keep])
    classes = info["Class"].fillna("other").replace(_CLASS_ALIASES).values
    return dict(X=X[keep], measured=meas[keep], names=info["Name"].fillna("").values,
                classes=classes, inchikey=R.index[keep].values)


def biconditional_mixtures(X0, sets, n: int, noise: float = 0.5, add_noise: float = 0.05, sat: float = 0.3,
                           seed: int | None = None, rng=None):
    """조건부 구별 과제: 냄새 4개 (a, b, c, d) 묶음마다 AB, CD → 1 (보상), AC, BD → 0 (무보상)

    혼합물 = 포화(Σ 성분 × exp(N(0, noise)) + |N(0, add_noise)|), 포화(x) = x / (x + sat)
    반환: (x, y, 묶음 번호), 묶음·혼합물마다 n개. rng: numpy Generator (없으면 seed로 만듦)
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    X0 = np.asarray(X0, np.float32)
    xs, ys, ps = [], [], []
    G = X0.shape[1]
    for p, (a, b, c, d) in enumerate(sets):
        for comp, label in (((a, b), 1), ((c, d), 1), ((a, c), 0), ((b, d), 0)):
            base = sum(X0[k] * np.exp(noise * rng.normal(size=(n, G))) for k in comp)
            x = base + np.abs(add_noise * rng.normal(size=(n, G)))
            xs.append(x / (x + sat))
            ys += [label] * n
            ps += [p] * n
    return np.concatenate(xs).astype(np.float32), np.array(ys), np.array(ps)
