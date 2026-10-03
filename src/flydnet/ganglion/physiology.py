"""신경절의 작용: Signal을 받는 상태 없는 함수 (torch.nn.functional에 해당)

  transmit(x, values, wiring)   시냅스 전달: 커넥톰 배선(희소)으로 (B, n_pre) → (B, n_post)   ≈ F.linear
  fire(v, threshold, slope)     발화: 문턱 넘으면 1, 역전파는 대리 기울기
  inhibit(x, k=, frac=)         측억제: 가장 강한 k개만 남김 (APL)
  surprise(logits, y)           놀람 = −log p(정답) 평균                                    ≈ F.cross_entropy
  log_softmax(x)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import backend as B
from .signal import Signal, as_signal


# ─────────────── 시냅스 전달 ───────────────
class Wiring(NamedTuple):
    """희소 배선 (post × pre), CSR. 연결 e의 값 순서 = (post, pre) 정렬 순서"""
    indptr: object                 # (n_post + 1,)
    pre: object                    # (E,) 연결 e를 보내는 뉴런 (CSR 열 번호)
    post: object                   # (E,) 연결 e를 받는 뉴런
    n_post: int
    n_pre: int

    @property
    def shape(self):
        return self.n_post, self.n_pre

    @property
    def device(self) -> str:
        return B.device_of(self.pre)

    def to(self, device: str) -> "Wiring":
        return Wiring(B.to(self.indptr, device), B.to(self.pre, device), B.to(self.post, device),
                      self.n_post, self.n_pre)

    def matrix(self, values):
        """values 배열로 희소 행렬 (n_post, n_pre)"""
        sp = B.sparse(self.device)
        return sp.csr_matrix((values, self.pre, self.indptr), shape=self.shape)


def wiring(post, pre, n_post: int, n_pre: int, device: str = "cpu") -> tuple[Wiring, np.ndarray]:
    """연결 목록 → (Wiring, order). transmit의 values는 원래 순서의 값[order]로. 같은 연결이 두 번이면 오류"""
    post, pre = np.asarray(post, np.int64), np.asarray(pre, np.int64)
    if len(post) and (post.min() < 0 or post.max() >= n_post or pre.min() < 0 or pre.max() >= n_pre):
        raise ValueError("연결 번호가 범위를 벗어남")
    order = np.lexsort((pre, post))
    post, pre = post[order], pre[order]
    if len(post) > 1 and ((np.diff(post) == 0) & (np.diff(pre) == 0)).any():
        raise ValueError("같은 연결이 두 번 있음 — 시냅스 수를 먼저 합칠 것")
    indptr = np.zeros(n_post + 1, np.int64)
    indptr[1:] = np.cumsum(np.bincount(post, minlength=n_post))
    idx = np.int32 if max(n_post, n_pre, len(post)) < 2 ** 31 else np.int64
    w = Wiring(indptr.astype(idx), pre.astype(idx), post.astype(idx), n_post, n_pre)
    return (w.to(device) if device != "cpu" else w), order


def transmit(x, values: Signal, w: Wiring, edge_chunk: int = 1 << 26) -> Signal:
    """시냅스 전달: x (B, n_pre) → (B, n_post), out[:, post] = Σ_e values[e] · x[:, pre_e]
    역전파: d x = g @ W,  d values[e] = Σ_b g[b, post_e] · x[b, pre_e]  (연결 칸만 계산, 밀집 행렬 없음)"""
    x = as_signal(x)
    values = as_signal(values)
    if x.ndim != 2 or x.shape[1] != w.n_pre:
        raise ValueError(f"입력 모양 {x.shape} — (B, {w.n_pre}) 이어야 함")
    if x.device != w.device or values.device != w.device:
        raise RuntimeError(f"장치가 다름: 입력 {x.device}, 값 {values.device}, 배선 {w.device}")
    xp = x.xp
    W = w.matrix(values.data)
    xd = x.data
    out = (W @ xd.T).T                                           # (B, n_post)
    out = xp.ascontiguousarray(out)

    def back(g):
        dx = (W.T @ g.T).T if x.plastic else None
        dv = None
        if values.plastic:
            E = len(w.pre)
            step = max(1, edge_chunk // max(1, g.shape[0]))      # 메모리: 묶음 × step
            dv = xp.empty(E, dtype=values.data.dtype)
            for s in range(0, E, step):
                q, p = w.post[s:s + step], w.pre[s:s + step]
                dv[s:s + step] = (g[:, q] * xd[:, p]).sum(axis=0)
        return (None if dx is None else xp.ascontiguousarray(dx), dv)
    return Signal(out)._link((x, values), back)


# ─────────────── 발화 · 억제 ───────────────
def fire(v, threshold: float = 0.0, slope: float = 10.0) -> Signal:
    """발화: v > threshold면 1. 역전파는 g / (1 + slope·|v − threshold|)² (대리 기울기, SuperSpike)"""
    v = as_signal(v)
    d = v.data - threshold
    out = (d > 0).astype(v.data.dtype)
    return Signal(out)._link((v,), lambda g: (g / (1 + slope * abs(d)) ** 2,))


def inhibit(x, k: int | None = None, frac: float | None = None) -> Signal:
    """측억제 (APL처럼): 마지막 차원에서 가장 강한 k개(또는 비율 frac)만 남기고 0. 기울기는 남은 것에만.
    k번째 값과 같은 값이 여럿이면 모두 남음"""
    if (k is None) == (frac is None):
        raise ValueError("k와 frac 중 하나만")
    x = as_signal(x)
    n = x.shape[-1]
    k = int(round(frac * n)) if k is None else int(k)
    k = max(1, min(n, k))
    xp = x.xp
    kth = -xp.partition(-x.data, k - 1, axis=-1)[..., k - 1:k]   # k번째로 큰 값
    m = x.data >= kth
    return Signal(x.data * m)._link((x,), lambda g: (g * m,))


# ─────────────── 오차 ───────────────
def log_softmax(x, axis: int = -1) -> Signal:
    x = as_signal(x)
    xp = x.xp
    z = x.data - x.data.max(axis=axis, keepdims=True)
    lse = xp.log(xp.exp(z).sum(axis=axis, keepdims=True))
    out = z - lse
    sm = xp.exp(out)
    return Signal(out)._link((x,), lambda g: (g - sm * g.sum(axis=axis, keepdims=True),))


def surprise(logits, y) -> Signal:
    """놀람 = 정답 확률의 −log, 묶음 평균 (교차 엔트로피, F.cross_entropy). y: 정수 클래스 배열"""
    logits = as_signal(logits)
    xp = logits.xp
    y = B.to(B.labels(y), logits.device)
    lp = log_softmax(logits)
    picked = lp[xp.arange(len(y)), y]
    return -picked.mean()
