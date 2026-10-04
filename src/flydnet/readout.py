"""발화율 특징 위에 학습하는 리드아웃 (자체 엔진판, torch 없음). 결과는 numpy"""
from __future__ import annotations

from . import _check as _C
import numpy as np

from ._console import say

from .ganglion import backend as B
from .ganglion.physiology import surprise
from .ganglion.rules import AdaptivePlasticity
from .ganglion.signal import Signal, quiescent
from .ganglion.tissue import Projection


def _np(x) -> np.ndarray:
    if isinstance(x, Signal):
        return x.numpy()
    if hasattr(x, "detach"):                                         # torch 텐서
        return x.detach().cpu().numpy()
    return B.numpy(x) if B.device_of(x) == "gpu" else np.asarray(x)


def _accepts_seed(fn) -> bool:
    """fn(x, seed=...)를 받는지 (TypeError를 잡아 다시 부르면 층 안의 진짜 오류까지 삼킴)"""
    import inspect
    target = getattr(fn, "forward", fn)
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return "seed" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def extract(layer, encoder, X, batch: int = 256, seed: int = 0, log_every: int = 0) -> np.ndarray:
    """데이터 X 전체를 (encoder →) layer에 통과시켜 출력 특징 (n, n_out)을 numpy로. encoder=None이면 X를 바로"""
    _C.integer('batch', batch)
    if len(X) == 0:
        raise ValueError("데이터가 비어 있음 (시료 0개)")
    out = []
    takes_seed = _accepts_seed(layer)
    with quiescent():
        for i in range(0, len(X), batch):
            x = X[i:i + batch]
            x = encoder(x) if encoder is not None else x
            y = layer(x, seed=seed + i) if takes_seed else layer(x)
            out.append(_np(y))
            if log_every and (i // batch) % log_every == 0:
                say(f"    {i + len(out[-1]):6d}/{len(X)}", flush=True)
    return np.concatenate(out)


def train_linear(Xtr, ytr, Xte, yte, n_classes=None, epochs: int = 30, lr: float = 1e-2, wd: float = 1e-4,
                 batch: int = 512, device: str | None = None, scale: str = "global", seed: int = 0) -> dict:
    """정규화 + 로지스틱 회귀(소프트맥스), 적응형 가소성(AdamW와 같은 감쇠) + 코사인 학습률. 정확도 반환

    scale: "global"  = 특징별 평균 빼고 전체 표준편차 하나로 나눔 (드물게 켜지는 특징이 폭주하지 않음)
           "feature" = 특징별 표준화
    seed:  가중치 초기화와 샘플 순서
    """
    _C.integer('epochs', epochs, lo=0)
    _C.nonneg('lr', lr)
    _C.nonneg('wd', wd)
    _C.integer('batch', batch)
    dev = B.check(device) if device is not None else B.default_device()
    Xtr, Xte = _np(Xtr).astype(np.float32), _np(Xte).astype(np.float32)
    ytr, yte = B.labels(ytr), B.labels(yte)
    if len(Xtr) == 0 or len(Xte) == 0:
        raise ValueError(f"데이터가 비어 있음: 학습 {len(Xtr)}개, 평가 {len(Xte)}개")
    if len(Xtr) != len(ytr) or len(Xte) != len(yte):
        raise ValueError(f"특징과 라벨의 개수가 다름: 학습 {len(Xtr)} 대 {len(ytr)}, 평가 {len(Xte)} 대 {len(yte)}")
    n_classes = n_classes or int(max(ytr.max(), yte.max())) + 1
    B.check_labels(ytr, n_classes, "학습 라벨"); B.check_labels(yte, n_classes, "평가 라벨")
    mu = Xtr.mean(0)
    sd = np.maximum(Xtr.std(0), 1e-6) if scale == "feature" else max(float((Xtr - mu).std()), 1e-6)
    xp = B.xp(dev)
    Xtr_, Xte_ = B.to((Xtr - mu) / sd, dev), B.to((Xte - mu) / sd, dev)
    ytr_ = B.to(ytr, dev)
    model = Projection(Xtr.shape[1], n_classes, seed=seed, device=dev)
    rule = AdaptivePlasticity(model.synapses(), rate=lr, decay=wd)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        rule.rate = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))     # 코사인 감소 (에폭마다)
        perm = B.to(rng.permutation(len(Xtr_)), dev)
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            loss = surprise(model(Signal(Xtr_[j])), ytr_[j])
            rule.clear(); loss.retrograde(); rule.step()
    with quiescent():
        acc = lambda X, y: float((B.numpy(model(Signal(X)).data.argmax(1)) == y).mean())
        return dict(train_acc=acc(Xtr_, ytr), test_acc=acc(Xte_, yte), model=model)
