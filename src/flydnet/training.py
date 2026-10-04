"""학습 편의: 학습 루프 한 줄 (fd.train), 정확도 (fd.evaluate), 실제 냄새 분류 과제 (fd.door_task)

  Xtr, ytr, Xte, yte = fd.door_task(n_odors=12)                       # DoOR 실제 냄새, 사구체 활성 [0, 1]
  enc = fd.GlomerularEncoder(mb)
  model = fd.Pathway(enc, fd.ConnectomeLayer(mb, "PN", "MBON", trainable=True), fd.Projection(48, 12))
  hist = fd.train(model, Xtr, ytr, val=(Xte, yte), epochs=10)          # 배치·섞기·코사인 학습률·clip·평가
  print(hist["val_acc"])

model: Tissue (Pathway 등) 또는 함수 f(x) → 로짓 Signal. seed를 받는 구조물(ConnectomeLayer)이 들어 있으면 배치마다
다른 seed를 넘김 (Pathway는 seed를 받는 자식에게 전달).
"""
from __future__ import annotations

from . import _check as _C
import numpy as np

from ._console import say
from .ganglion import backend as B
from .ganglion.physiology import surprise
from .ganglion.rules import AdaptivePlasticity
from .ganglion.signal import Signal, quiescent


def _call(model, x, seed):
    """seed를 받는 모델이면 seed와 함께 (Pathway는 seed를 받는 자식에게 넘김)"""
    from .ganglion.tissue import _takes_seed
    return model(x, seed=seed) if _takes_seed(model) else model(x)


def _rows(X):
    """배치로 자를 수 있는 형태로 (리스트·pandas → numpy, numpy·cupy·torch·Signal은 그대로)"""
    if isinstance(X, Signal) or hasattr(X, "shape") and not hasattr(X, "iloc"):
        return X
    return np.asarray(X, dtype=np.float32)


def evaluate(model, X, y, batch: int = 256, seed: int = 10 ** 6) -> float:
    """정확도 (역전파 경로 없이, 배치로)"""
    _C.integer('batch', batch)
    X = _rows(X)
    y = B.labels(y)
    if len(X) != len(y):
        raise ValueError(f"X와 y의 개수가 다름: {len(X)} 대 {len(y)}")
    hits = 0
    with quiescent():
        for i in range(0, len(X), batch):
            out = _call(model, X[i:i + batch], seed + i)
            hits += int((B.numpy(out.data if isinstance(out, Signal) else out).argmax(-1) == y[i:i + batch]).sum())
    return hits / max(len(y), 1)


def train(model, X, y, epochs: int = 10, batch: int = 32, rate: float = 3e-3, decay: float = 0.0,
          clip: float | None = 1.0, schedule: str = "cosine", val=None, loss=surprise, synapses=None, seed: int = 0,
          verbose: bool = True) -> dict:
    """학습 루프: 에폭마다 섞어 배치로, 적응형 가소성(AdamW) + 학습률 스케줄 + 기울기 크기 제한.

    schedule: "cosine" (에폭마다 코사인으로 0까지) / "constant"
    val:      (X, y)면 에폭마다 정확도
    loss:     loss(logits, y_batch) → 값 하나인 Signal (기본 교차 엔트로피)
    synapses: 바꿀 시냅스 (기본 model.named_synapses())
    반환: dict(loss=[에폭별 평균 손실], val_acc=[에폭별 정확도], train_acc=마지막, rule=가소성 규칙)"""
    _C.nonneg('rate', rate)
    _C.nonneg('decay', decay)
    _C.optional(_C.pos, 'clip', clip)
    X = _rows(X)
    y = B.labels(y)
    if len(X) != len(y):
        raise ValueError(f"X와 y의 개수가 다름: {len(X)} 대 {len(y)}")
    if not (isinstance(batch, (int, np.integer)) and batch >= 1):
        raise ValueError(f"batch는 1 이상의 정수: {batch}")
    if not (isinstance(epochs, (int, np.integer)) and epochs >= 0):
        raise ValueError(f"epochs는 0 이상의 정수: {epochs}")
    if schedule not in ("cosine", "constant"):
        raise ValueError(f"schedule은 'cosine' 또는 'constant': {schedule!r}")
    if synapses is None:
        if not hasattr(model, "named_synapses"):
            raise TypeError("함수 모델이면 synapses=[...]를 줄 것")
        synapses = list(model.named_synapses())
    rule = AdaptivePlasticity(synapses, rate=rate, decay=decay, clip=clip)
    rng = np.random.default_rng(seed)
    hist = dict(loss=[], val_acc=[])
    step = 0
    for ep in range(epochs):
        rule.rate = rate * (0.5 * (1 + np.cos(np.pi * ep / epochs)) if schedule == "cosine" else 1.0)
        perm = rng.permutation(len(X))
        tot, n = 0.0, 0
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            out = _call(model, X[j], seed * 10 ** 6 + step)
            L = loss(out, y[j])
            rule.clear(); L.retrograde(); rule.step()
            tot += float(L.data) * len(j); n += len(j)
            step += 1
        hist["loss"].append(tot / n)
        if val is not None:
            hist["val_acc"].append(evaluate(model, val[0], val[1], seed=seed * 10 ** 6 + 10 ** 5))
        if verbose:
            extra = f", 평가 정확도 {hist['val_acc'][-1]:.3f}" if val is not None else ""
            say(f"  에폭 {ep + 1}/{epochs}: 손실 {hist['loss'][-1]:.4f}{extra}", flush=True)
    hist["train_acc"] = evaluate(model, X, y)
    hist["rule"] = rule
    return hist


def door_task(n_odors: int = 12, samples: int = 24, noise: float = 0.8, background: float = 0.15, seed: int = 0,
              glomeruli=None, data_dir=None):
    """DoOR 2.0 실제 냄새 구분 과제: 반응이 큰 냄새 n_odors개, 냄새마다 학습·평가 시료 samples개씩.
    시료 = 냄새의 사구체 반응 x 로그 정규 잡음(noise) + 배경 잡음 (0 ~ background), [0, 1]로 자름.
    glomeruli: 사구체 순서 (기본: 오른쪽 버섯체 GlomerularEncoder의 순서 - FlyWire 데이터 필요)
    반환: Xtr, ytr, Xte, yte (X는 (시료, 사구체) float32). 냄새 이름은 door_task.names에"""
    _C.nonneg('noise', noise)
    _C.nonneg('background', background)
    from .datasets import door_odors
    if glomeruli is None:
        from .circuit import Circuit
        from .encoders import GlomerularEncoder
        glomeruli = GlomerularEncoder(Circuit.from_flywire(), device="cpu").glomeruli
    for nm, v, lo in (("n_odors", n_odors, 2), ("samples", samples, 1)):
        if not (isinstance(v, (int, np.integer)) and v >= lo):
            raise ValueError(f"{nm}는 {lo} 이상의 정수: {v}")
    door = door_odors(glomeruli, data_dir=data_dir)
    if len(door["X"]) < 2:
        raise ValueError(f"쓸 수 있는 냄새가 {len(door['X'])}개 - 사구체 {len(glomeruli)}개 중 측정된 것이 적음 "
                         "(door_odors의 min_measured=20 기준). glomeruli를 비우면 버섯체 PN의 사구체 전체를 씀")
    if not 2 <= n_odors <= len(door["X"]):
        raise ValueError(f"n_odors는 2 ~ {len(door['X'])}")
    pick = np.argsort(door["X"].sum(1))[::-1][:n_odors]
    proto = door["X"][pick] / door["X"][pick].max()
    rng = np.random.default_rng(seed)

    def make():
        yy = np.repeat(np.arange(n_odors), samples)
        XX = proto[yy] * rng.lognormal(0, noise, (len(yy), proto.shape[1])) + \
            rng.uniform(0, background, (len(yy), proto.shape[1]))
        return np.clip(XX, 0, 1).astype(np.float32), yy
    Xtr, ytr = make()
    Xte, yte = make()
    door_task.names = [door["names"][i] for i in pick]
    return Xtr, ytr, Xte, yte
