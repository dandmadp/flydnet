"""flydnet.lab.contrastive - [실험적] 대조 학습 성장. 인터페이스가 바뀌거나 없어질 수 있음

  from flydnet import lab
  g = lab.Growth(layer, allow=["PN>KC"], budget=5000)               # 추가 연결 부품을 층에 붙임 (lab/growth.py)
  lab.grow_contrastive(g, 1350, inputs=X, encoder=enc)              # 정답 없이 추가 연결 자리 고르기 (대조 학습)
  lab.grow_contrastive(g, 1350, inputs=X, encoder=enc, augment=lab.corrupt(0.3))   # 보기 만들기 = SCARF

대조 학습 성장: 라벨 없는 데이터를 두 번 따로 흔들어 (augment) 층 출력 두 벌을 만들고, 각 시료가 자기 짝을 다른 시료들
사이에서 찾도록 하는 손실 (InfoNCE, SimCLR - Chen et al. 2020)의 기울기로 추가 연결 자리를 고름 (Growth.grow(rule="gradient")와
같은 방식, 손실만 정답 대신 대조 손실).
결과 (lab/growth_lesion*.py, seed 6개): 버섯체 PN→KC 80% 손상 회복에서 냄새는 1,350개로 손상 전과 차이 없음 (무작위보다
+0.032, p = 0.031), MNIST는 무작위를 넘지 못함 (0.851 대 0.862) - 입력 채널에 뜻이 있는 데이터에서만 통함
"""
from __future__ import annotations

import numpy as np

from .. import _check as _C


def info_nce(z1, z2, tau: float = 0.1):
    """대조 손실 (InfoNCE): z1·z2 (B, d) Signal, i번째 행끼리가 짝. 코사인 유사도 / tau로 행마다 B개 중 자기 짝 고르기의
    교차 엔트로피 - 같은 시료의 두 보기는 닮고 다른 시료와는 달라지도록. 라벨이 필요 없음 (SimCLR, Chen et al. 2020)"""
    from ..ganglion.physiology import surprise

    def unit(z):
        return z / ((z * z).sum(axis=1, keepdims=True) + 1e-6) ** 0.5
    return surprise((unit(z1) @ unit(z2).T) * (1.0 / tau), np.arange(z1.shape[0]))


def views(X, rng, augment=0.3):
    """데이터 X (B, ...)의 흔든 보기 하나. augment: 숫자 = 곱 로그 정규 잡음 표준편차 ([min(0, 최솟값), max(1, 최댓값)]으로 자름),
    함수 = augment(X, rng) (예: corrupt(0.3) - 특징 일부를 다른 시료 값으로)"""
    X = np.asarray(X, np.float32)
    if callable(augment):
        return np.asarray(augment(X, rng), np.float32)
    _C.nonneg("augment", augment)
    lo = min(0.0, float(X.min())) if X.size else 0.0                    # 음수가 있는 데이터(표준화 등)는 자르지 않음
    hi = max(1.0, float(X.max())) if X.size else 1.0                    # (예전: 늘 0 아래를 잘라 음수 특징이 모두 0이 됨)
    return np.clip(X * np.exp(augment * rng.standard_normal(X.shape)), lo, hi).astype(np.float32)


def corrupt(frac: float = 0.3):
    """표 데이터용 보기 만들기 (SCARF, Bahri et al. 2022): 시료마다 특징의 frac를 골라, 같은 묶음의 다른 시료가 가진 그 특징
    값으로 바꿈 - 값의 분포는 그대로, 데이터를 만든 잡음 방식은 몰라도 됨. augment=corrupt(0.3)처럼 씀"""
    _C.unit("frac", frac)

    def f(X, rng):
        X = np.asarray(X, np.float32)
        n, d = X.shape[0], int(np.prod(X.shape[1:]))
        flat = X.reshape(n, d)
        mask = rng.random((n, d)) < frac
        donor = rng.integers(0, n, (n, d))
        return np.where(mask, flat[donor, np.arange(d)], flat).reshape(X.shape).astype(np.float32)
    return f


def contrastive_loss(inputs, rng, encoder=None, augment=0.3, tau: float = 0.1, samples: int = 64):
    """grow_contrastive의 손실 함수 loss(layer) (Growth.grow(rule="gradient", loss=...)에 그대로 넘겨도 됨): inputs에서 samples개 → 보기 두 벌 (encoder가 있으면 변환) →
    층 출력의 InfoNCE. 두 보기는 층 seed도 다르게 (0, 1)"""
    from ..ganglion.signal import Signal, quiescent
    _C.integer("samples", samples, lo=2)
    _C.pos("tau", tau)
    from ..readout import _np
    X = _np(inputs) if isinstance(inputs, Signal) or hasattr(inputs, "detach") or hasattr(inputs, "__cuda_array_interface__")         else np.asarray(inputs, np.float32)                                # numpy·cupy·Signal·torch·리스트
    if len(X) < 2:
        raise ValueError("contrastive: 시료가 2개 이상 필요 (서로 구별할 것이 있어야 함)")
    X = X[rng.permutation(len(X))[:samples]]
    x1, x2 = views(X, rng, augment), views(X, rng, augment)
    if encoder is not None:
        with quiescent():
            x1, x2 = encoder(x1), encoder(x2)
        x1, x2 = (x.data if isinstance(x, Signal) else x for x in (x1, x2))
    return lambda layer: info_nce(layer(x1, seed=0), layer(x2, seed=1), tau)


def grow_contrastive(growth, n: int, inputs, encoder=None, augment=0.3, tau: float = 0.1, samples: int = 64,
                     candidates: int = 10, seed=None) -> int:
    """정답 없이 추가 연결 n개: 라벨 없는 데이터 inputs (encoder가 있으면 그 입력, 없으면 층 입력 Hz)에서 samples개를 골라
    augment로 두 번 흔들고, 대조 손실(InfoNCE, 코사인 / tau)의 기울기로 Growth.grow(rule="gradient")처럼 고름. growth: lab.Growth 또는 그것이 붙은 층.
    augment: 숫자 = 곱 로그 정규 잡음 세기, 함수 = augment(X, rng) (예: corrupt(0.3) - SCARF, 이미지면 이동 등).
    받는 뉴런 상한(per_neuron)은 층의 growth 설정을 따름 - 상한이 없으면 소수 뉴런에 몰리기 쉬움 (MNIST). 반환: 만든 수"""
    from .growth import Growth
    if not isinstance(growth, Growth):                                   # 층을 주면 그 층에 붙은 Growth
        growth = getattr(growth, "growth", None)
        if not isinstance(growth, Growth):
            raise ValueError("추가 연결 부품(lab.Growth) 또는 그것이 붙은 층이 필요: g = lab.Growth(layer, budget=5000)")
    if inputs is None:
        raise ValueError("grow_contrastive는 라벨 없는 데이터 inputs가 필요")
    _C.integer("n", n, lo=0)
    _C.pos("tau", tau)                                                   # 인자 확인을 먼저 (상한이 차서 아무것도 안 만들 때도)
    _C.integer("samples", samples, lo=2)
    _C.integer("candidates", candidates)
    if not callable(augment):
        _C.nonneg("augment", augment)
    rng = np.random.default_rng(seed)
    if n == 0 or growth.n >= growth.budget:
        return 0
    loss = contrastive_loss(inputs, rng, encoder=encoder, augment=augment, tau=tau, samples=samples)
    return growth.grow(n, rule="gradient", loss=loss, candidates=candidates, seed=rng)   # 같은 난수 흐름 (손실 → 후보)
