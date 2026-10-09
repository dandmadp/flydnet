"""역전파 없는 도파민 학습 리드아웃 - 버섯체 KC→MBON 시냅스 규칙 (자체 엔진판, torch 없음)

    w[c, k] ← w[c, k] · (1 - lr · a_k · DA_c)        (ltd, 처벌 도파민: 시냅스 약화)
    w[c, k] ← w[c, k] + lr · a_k · DA_c · (w_max - w) (ltp, 보상 도파민: 시냅스 강화)
    w[c, k] ← max(0, w[c, k] + lr · a_k · DA_c)        (bidir, DA_c = +1 정답 / -1 이긴 오답, 틀렸을 때만)
    w[c, k] ← w[c, k] + DA_c · (a_k - w[c, k]) / n_c   (assoc, DA_c = 1 정답만. 정답 출력이 그 클래스 평균 패턴이 됨)

a_k  = KC k의 활동 (샘플 안에서 최대 발화율로 정규화, 0~1)
DA_c = 출력 c를 담당하는 도파민 뉴런의 신호 (정답/오답에서 결정)
출력 c의 점수 = Σ_k w[c, k] · a_k  (가장 큰 출력 = 예측)
시냅스 하나의 변화는 그 시냅스 앞 KC 활동과 그 출력의 도파민만으로 정해짐 (국소 규칙)

입력은 numpy·cupy 배열, Signal, torch 텐서 모두 받음. 예측은 numpy. 저장은 np.savez.
"""
from __future__ import annotations

from . import _check as _C
import json

import numpy as np

from .ganglion import backend as B
from .ganglion.physiology import recall, reinforce_, check_activity
from .ganglion.signal import Signal


def _arr(x, device):
    if isinstance(x, Signal):
        x = x.data
    elif hasattr(x, "detach"):                                         # torch 텐서
        x = x.detach().cpu().numpy()
    elif hasattr(x, "to_numpy"):                                       # pandas (예전: 행 이름으로 고르다 KeyError)
        x = x.to_numpy()
    return B.to(np.asarray(x) if not hasattr(x, "shape") else x, device)


class _Saveable:
    """설정 + 배열을 np.savez 한 파일로"""
    _ARRAYS: tuple = ()

    def _config(self) -> dict:
        raise NotImplementedError

    def state(self) -> dict:
        """복사본 (예전: CPU에선 내부 배열 그대로라 best = r.state()로 남긴 값이 이어 배우면 조용히 바뀌었음)"""
        return {"format": np.array(type(self).__name__), "config": np.array(json.dumps(self._config())),
                **{k: np.array(B.numpy(getattr(self, k)), copy=True) for k in self._ARRAYS}}

    def save(self, path):
        from ._archive import write
        return write(path, type(self).__name__, self.state())

    @classmethod
    def load(cls, path, device: str | None = None):
        from ._archive import read
        d, _ = read(path, cls.__name__)
        if str(d.get("format")) != cls.__name__:
            raise ValueError(f"{cls.__name__} 파일이 아님 (format={d.get('format')})")
        obj = cls(**json.loads(str(d["config"])), device=device)
        for k in cls._ARRAYS:
            setattr(obj, k, B.to(d[k], obj.device))
        return obj


class DopamineReadout(_Saveable):
    MODES = ("bidir", "assoc", "ltd", "ltd_err", "ltp")
    _ARRAYS = ("W", "n_seen")

    def __init__(self, n_in: int, n_classes: int, mode: str = "bidir", lr: float = 0.01,
                 binary: bool = False, homeostasis: bool = False, device: str | None = None):
        """mode
        bidir   : 틀렸을 때만, 정답 출력은 강화 + 이긴 오답 출력은 약화 (양방향 가소성, 권장)
        assoc   : 정답 출력만 강화 (보상 연합 학습) + 출력별 시냅스 총량 정규화(코사인 점수). 연속 학습에서 망각 없음
        ltd     : 정답이 아닌 모든 출력에 처벌 도파민 (항상)
        ltd_err : 정답보다 점수가 높거나 같았던 틀린 출력에만 처벌 도파민 (틀렸을 때만)
        ltp     : 정답 출력에 보상 도파민 (항상)
        binary  : True면 KC 활동을 켜짐/꺼짐(0/1)으로
        """
        _C.integer('n_in', n_in)
        _C.integer('n_classes', n_classes)
        _C.nonneg('lr', lr)
        if mode not in self.MODES:
            raise ValueError(f"mode는 {self.MODES} 중 하나: {mode!r}")
        self.mode, self.lr, self.binary, self.homeostasis = mode, lr, binary, homeostasis
        self.device = B.check(device) if device is not None else B.default_device()
        xp = B.xp(self.device)
        self.n_classes = n_classes
        init = {"ltp": 0.0, "assoc": 0.0, "bidir": 0.5}.get(mode, 1.0)
        self.W = xp.full((n_classes, n_in), init, dtype=xp.float32)
        self.n_seen = xp.zeros(n_classes, dtype=xp.float32)

    @property
    def xp(self):
        return B.xp(self.device)

    def _config(self):
        return dict(n_in=int(self.W.shape[1]), n_classes=self.n_classes, mode=self.mode, lr=self.lr,
                    binary=self.binary, homeostasis=self.homeostasis)

    def activity(self, X):
        X = _arr(X, self.device)
        X = X.reshape(1, -1) if X.ndim == 1 else X.reshape(len(X), int(np.prod(X.shape[1:])))    # 시료 하나 (n,)도 (예전: (n, 1)로 봐서 오류)
        X = check_activity(X.astype(np.float32, copy=False), int(self.W.shape[1]))
        if self.binary:
            return (X > 0).astype(np.float32)
        return X / self.xp.maximum(X.max(axis=1, keepdims=True), 1e-8)

    def scores(self, X):
        xp, W = self.xp, self.W
        if self.mode == "assoc":
            W = W / xp.maximum(xp.linalg.norm(W, axis=1, keepdims=True), 1e-8)
        elif self.homeostasis:
            W = W / xp.maximum(W.mean(axis=1, keepdims=True), 1e-8)
        return self.activity(X) @ W.T

    def _masked(self, s, classes):
        if classes is None:
            return s
        return s + B.class_mask(classes, self.n_classes, self.xp, s.dtype)

    def predict(self, X, classes=None) -> np.ndarray:
        """예측 클래스 (numpy). 시료 하나 (n,)면 클래스 번호 하나"""
        out = B.numpy(self._masked(self.scores(X), classes).argmax(1))
        return out[0] if np.ndim(X.data if isinstance(X, Signal) else X) == 1 else out

    def dopamine(self, X, y, classes=None):
        """(B, C) 도파민 신호"""
        xp = self.xp
        y = B.to(B.check_labels(B.sample_labels(y), self.n_classes), self.device)
        onehot = xp.eye(self.n_classes, dtype=xp.float32)[y]
        if self.mode == "bidir":
            win = self._masked(self.scores(X), classes).argmax(1)
            wrong = (win != y).astype(xp.float32)[:, None]
            return wrong * (onehot - xp.eye(self.n_classes, dtype=xp.float32)[win])
        if self.mode in ("ltp", "assoc"):
            return onehot
        if self.mode == "ltd":
            return 1 - onehot
        s = self.scores(X)
        return ((s >= xp.take_along_axis(s, y[:, None], axis=1)) & (onehot == 0)).astype(xp.float32)

    def step(self, X, y, classes=None):
        """샘플 묶음 하나로 시냅스 갱신 (묶음 안 변화는 평균). classes: 경쟁할 출력 (bidir)"""
        xp = self.xp
        a = self.activity(X)
        if len(a) != len(B.sample_labels(y)):
            raise ValueError(f"X와 y의 개수가 다름: {len(a)} 대 {len(B.sample_labels(y))}")
        if self.mode == "assoc":                                          # 누적 평균: 순서가 바뀌어도 결과 같음
            da = self.dopamine(X, y, classes)
            self.n_seen += da.sum(0)
            self.W += (da.T @ a - da.sum(0)[:, None] * self.W) / xp.maximum(self.n_seen, 1)[:, None]
            return
        drive = self.dopamine(X, y, classes).T @ a / len(a)
        if self.mode == "bidir":
            self.W = xp.maximum(self.W + self.lr * drive, 0)
        elif self.mode == "ltp":
            self.W += self.lr * drive * (1 - self.W)
        else:
            self.W *= xp.maximum(1 - self.lr * drive, 0)

    def fit(self, X, y, epochs: int = 1, batch: int = 32, seed: int = 0, classes=None):
        rng = np.random.default_rng(seed)
        y = B.sample_labels(y)
        if hasattr(X, "to_numpy"):                                          # pandas (예전: 행 이름으로 골라 KeyError)
            X = X.to_numpy()
        if not hasattr(X, "shape") and not isinstance(X, Signal):           # 리스트 (예전: 번호 배열로 고르다 TypeError)
            X = np.asarray(X, np.float32)
        if len(X) != len(y):
            raise ValueError(f"X와 y의 개수가 다름: {len(X)} 대 {len(y)}")
        for _ in range(epochs):
            perm = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                j = perm[i:i + batch]
                self.step(_take(X, j), y[j], classes)
        return self

    def accuracy(self, X, y, classes=None) -> float:
        p, y = np.atleast_1d(self.predict(X, classes)), B.sample_labels(y)
        if len(p) != len(y):                                              # 예전: 라벨 1개면 모든 예측과 퍼져 비교 (조용히 틀림)
            raise ValueError(f"X와 y의 개수가 다름: {len(p)} 대 {len(y)}")
        return float((p == y).mean())


def _take(X, idx):
    if isinstance(X, Signal):
        return X.data[B.to(idx, X.device)]
    if hasattr(X, "detach"):
        import torch
        return X[torch.as_tensor(idx)]
    return X[B.to(idx, B.device_of(X))] if B.device_of(X) == "gpu" else X[idx]


class AssocReadout(DopamineReadout):
    """보상 연합 학습 리드아웃, 클래스마다 출력(원형) 여러 개 - 연속 학습용 (MushroomBodyOutput과 같은 계산)

    샘플 (a, y)가 오면 클래스 y의 출력 중 가장 잘 맞는 하나에만 보상 도파민 → 그 출력 = 받은 샘플들의 평균.
    비어 있는 출력이 있으면 그것부터 채움 (클래스당 온라인 k-평균). 점수 = 코사인, 클래스 점수 = 그 클래스 출력 중 최대.
    다른 클래스의 시냅스는 절대 바뀌지 않음 → 클래스를 차례로 배워도 앞의 것을 잊지 않음.
    """
    _ARRAYS = ("W", "count")

    def __init__(self, n_in: int, n_classes: int, per_class: int = 1, binary: bool = False,
                 device: str | None = None):
        _C.integer('n_in', n_in)
        _C.integer('n_classes', n_classes)
        _C.integer('per_class', per_class)
        self.n_classes, self.k, self.binary = n_classes, per_class, binary
        self.mode, self.homeostasis, self.lr = "assoc", False, 0.0
        self.device = B.check(device) if device is not None else B.default_device()
        xp = B.xp(self.device)
        self.W = xp.zeros((n_classes * per_class, n_in), dtype=xp.float32)
        self.count = xp.zeros(n_classes * per_class, dtype=xp.float32)

    def _config(self):
        return dict(n_in=int(self.W.shape[1]), n_classes=self.n_classes, per_class=self.k, binary=self.binary)

    def scores(self, X):
        """(B, C) 클래스 점수"""
        a = self.activity(X)
        return recall(a, self.W, self.count).reshape(len(a), self.n_classes, self.k).max(axis=2)

    def step(self, X, y, classes=None):
        """정답 클래스의 원형만 강화 (classes는 학습에 영향 없음 - 예측의 경쟁 범위에만 쓰임: predict(X, classes))"""
        reinforce_(self.W, self.count, self.activity(X), B.sample_labels(y), self.k)
