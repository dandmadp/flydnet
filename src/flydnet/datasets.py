"""시험용 데이터 생성"""
import torch


def synthetic_odors(n_classes: int, n_glomeruli: int, n_train: int, n_test: int, protos_per_class: int = 1,
                    active_frac: float = 0.2, noise: float = 0.5, add_noise: float = 0.1, seed: int = 0):
    """합성 냄새 분류 과제

    냄새 원형: 사구체마다 active_frac 확률로 켜짐, 세기 U(0.2, 1)
    클래스:   원형 protos_per_class개의 묶음 (예: '먹이' = 사과 냄새 또는 효모 냄새).
              2개 이상이면 사구체 공간에서 선형 분리가 어려워짐 → 확장 층(KC)이 필요한 과제
    샘플:     클래스의 원형 하나를 골라 × exp(N(0, noise)) (사구체별 세기 흔들림) + |N(0, add_noise)| (배경)
    반환: (Xtr, ytr, Xte, yte), X는 (n, n_glomeruli), 클래스당 n_train / n_test 개
    """
    g = torch.Generator().manual_seed(seed)
    P = n_classes * protos_per_class
    on = torch.rand(P, n_glomeruli, generator=g) < active_frac
    protos = on * (0.2 + 0.8 * torch.rand(P, n_glomeruli, generator=g))

    def sample(n):
        y = torch.arange(n_classes).repeat_interleave(n)
        k = y * protos_per_class + torch.randint(0, protos_per_class, (len(y),), generator=g)
        x = protos[k] * torch.exp(noise * torch.randn(len(y), n_glomeruli, generator=g))
        x = x + (add_noise * torch.randn(len(y), n_glomeruli, generator=g)).abs()
        return x, y

    Xtr, ytr = sample(n_train)
    Xte, yte = sample(n_test)
    return Xtr, ytr, Xte, yte


# 같은 계열인데 이름이 다른 DoOR 화학 계열
_CLASS_ALIASES = {"arom": "aromatics", "terpenes": "terpene", "sulfid": "sulfide"}


def door_odors(glomeruli, data_dir=None, min_measured: int = 20):
    """DoOR 2.0 실제 냄새 반응 → 사구체 벡터 (Münch & Galizia 2016, CC BY-SA 4.0)

    data_dir에 door_response_matrix.csv, door_mappings.csv, odor.csv 필요. None이면 flydnet.data_dir("door")
    (없으면 flydnet.download("door"), 출처 https://github.com/ropensci/DoOR.data 의 data/ 폴더)

    glomeruli: 사구체 이름 순서 (예: GlomerularEncoder.glomeruli)
    반환 dict:
      X        (n_odors, n_glomeruli) 반응 − 자발 발화(SFR), 0 아래는 0. 측정 안 된 칸도 0
      measured (n_odors, n_glomeruli) 측정 여부
      names, classes (화학 계열), inchikey
    min_measured: 이만큼 이상의 사구체가 측정된 냄새만
    """
    import numpy as np
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
    return dict(X=torch.tensor(X[keep]), measured=torch.tensor(meas[keep]), names=info["Name"].fillna("").values,
                classes=classes, inchikey=R.index[keep].values)


def biconditional_mixtures(X0, sets, n: int, noise: float = 0.5, add_noise: float = 0.05, sat: float = 0.3,
                           generator=None):
    """조건부 구별 과제 샘플: 냄새 4개 (a, b, c, d) 묶음마다 AB, CD → 1 (보상), AC, BD → 0 (무보상)

    혼합물 = 포화(Σ 성분 × exp(N(0, noise)) + |N(0, add_noise)|), 포화(x) = x / (x + sat)
    반환: (x, y, 묶음 번호), 묶음·혼합물마다 n개
    """
    xs, ys, ps = [], [], []
    G = X0.shape[1]
    for p, (a, b, c, d) in enumerate(sets):
        for comp, label in (((a, b), 1), ((c, d), 1), ((a, c), 0), ((b, d), 0)):
            base = sum(X0[k] * torch.exp(noise * torch.randn(n, G, generator=generator)) for k in comp)
            x = base + (add_noise * torch.randn(n, G, generator=generator)).abs()
            xs.append(x / (x + sat))
            ys += [label] * n
            ps += [p] * n
    return torch.cat(xs), torch.tensor(ys), torch.tensor(ps)
