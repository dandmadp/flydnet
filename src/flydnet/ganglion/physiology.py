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


def transmit(x, values: Signal, w: Wiring, edge_chunk: int = 1 << 26, matrix=None) -> Signal:
    """시냅스 전달: x (B, n_pre) → (B, n_post), out[:, post] = Σ_e values[e] · x[:, pre_e]
    역전파: d x = g @ W,  d values[e] = Σ_b g[b, post_e] · x[b, pre_e]  (연결 칸만 계산, 밀집 행렬 없음)
    matrix: w.matrix(values.data)로 미리 만든 희소 행렬 (같은 values로 여러 번 전달할 때 다시 만들지 않게)"""
    x = as_signal(x)
    values = as_signal(values)
    if x.ndim != 2 or x.shape[1] != w.n_pre:
        raise ValueError(f"입력 모양 {x.shape} — (B, {w.n_pre}) 이어야 함")
    if x.device != w.device or values.device != w.device:
        raise RuntimeError(f"장치가 다름: 입력 {x.device}, 값 {values.device}, 배선 {w.device}")
    xp = x.xp
    W = w.matrix(values.data) if matrix is None else matrix
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


# ─────────────── 자리 바꾸기 ───────────────
def put(base, idx, values) -> Signal:
    """base 배열(상수)의 idx 자리에 values를 넣은 신호. 역행성 신호는 values에만 (idx 자리 것)"""
    values = as_signal(values)
    xp = values.xp
    out = xp.array(base, dtype=values.data.dtype, copy=True)
    out[idx] = values.data
    return Signal(out)._link((values,), lambda g: (g[idx],))


def put_columns(x, cols, values) -> Signal:
    """x (B, n)의 cols 열을 values (B, len(cols))로 바꾼 신호 (입력 뉴런 활동 고정 등)"""
    x, values = as_signal(x), as_signal(values)
    out = x.data.copy()
    out[:, cols] = values.data

    def back(g):
        gx = g.copy()
        gx[:, cols] = 0
        return gx, g[:, cols]
    return Signal(out)._link((x, values), back)


def add_columns(x, cols, values) -> Signal:
    """x (B, n)의 cols 열에 values (B, len(cols))를 더한 신호 (cols는 겹치지 않아야 함)"""
    x, values = as_signal(x), as_signal(values)
    out = x.data.copy()
    out[:, cols] += values.data
    return Signal(out)._link((x, values), lambda g: (g, g[:, cols]))


# ─────────────── 난수 (상태 없음) ───────────────
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def hash_uniform(xp, seed: int, step: int, shape) -> object:
    """(시드, 스텝, 칸 번호) → [0, 1) 균등 난수. 상태가 없어서 다시 계산해도 같고, CPU·GPU 결과도 같음 (splitmix64)"""
    n = int(np.prod(shape))
    z = xp.arange(n, dtype=xp.uint64)
    z += xp.uint64((int(seed) * 0x9E3779B97F4A7C15 + int(step) * 0xD1B54A32D192ED03) % (1 << 64))
    z ^= z >> xp.uint64(30); z *= _M1
    z ^= z >> xp.uint64(27); z *= _M2
    z ^= z >> xp.uint64(31)
    return ((z >> xp.uint64(40)).astype(xp.float32) * np.float32(1.0 / (1 << 24))).reshape(shape)


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


# ─────────────── 기억 (배열, 역전파 없음) ───────────────
def _arr(x):
    return x.data if isinstance(x, Signal) else x


def recall(a, prototypes, count=None):
    """기억 인출: 활동 a (B, n)와 원형 (P, n)의 코사인 유사도 (B, P) 배열. count가 0인(빈) 원형은 −inf"""
    a, W = _arr(a), _arr(prototypes)
    xp = B.xp(B.device_of(W))
    s = a @ (W / xp.maximum(xp.linalg.norm(W, axis=1, keepdims=True), 1e-8)).T
    return s if count is None else xp.where(_arr(count) == 0, -xp.inf, s)


def reinforce_(prototypes, count, a, y, per_class: int):
    """도파민 강화 (배열 제자리 갱신): 클래스 y의 원형 per_class개 중
    - 빈 원형이 있으면 그 클래스의 i번째 샘플이 i번째 빈 원형을 채움
    - 나머지 샘플은 가장 잘 맞는 원형 하나에만 보상 → 그 원형 = 받은 샘플들의 평균 (누적)
    다른 클래스의 원형은 바뀌지 않음 → 클래스를 차례로 배워도 잊지 않음"""
    xp = B.xp(B.device_of(prototypes))
    a = _arr(a)
    y = B.to(B.labels(y), B.device_of(prototypes))
    k = per_class
    C = len(count) // k
    onehot = xp.eye(C, dtype=xp.int64)[y]
    rank = (xp.cumsum(onehot, axis=0) * onehot).sum(1) - 1               # 클래스 안 순번
    empty = count.reshape(C, k) == 0
    seed = rank < empty.sum(1)[y]
    if bool(seed.any()):
        order = xp.argsort((~empty).astype(xp.int8), axis=1, kind="stable")   # 빈 원형 번호가 앞으로
        idx = y[seed] * k + order[y[seed], rank[seed]]
        prototypes[idx] = a[seed]
        count[idx] = 1
    rest = ~seed
    if not bool(rest.any()):
        return
    a, y = a[rest], y[rest]
    own = recall(a, prototypes, count).reshape(len(a), C, k)[xp.arange(len(a)), y]
    idx = y * k + own.argmax(1)
    n = xp.zeros_like(count)
    B.scatter_add(n, idx, xp.ones(len(a), dtype=count.dtype))
    tot = xp.zeros_like(prototypes)
    B.scatter_add(tot, idx, a.astype(prototypes.dtype, copy=False))
    count += n
    hit = n > 0
    prototypes[hit] += (tot[hit] - n[hit, None] * prototypes[hit]) / count[hit, None]


def kenyon_code(x, w_pn_kc, k: int, projection=None, center: bool = True, binary: bool = False):
    """KC 희소 부호화 (FlyHash 구조), 배열: [투영] → PN 활동 → [평균 빼기] → W_pn_kc → 상위 k만
    w_pn_kc: (n_kc, n_pn), projection: (n_pn, n_in) 또는 None"""
    W = _arr(w_pn_kc)
    xp = B.xp(B.device_of(W))
    a = B.to(_arr(x), B.device_of(W)).reshape(len(_arr(x)), -1).astype(W.dtype, copy=False)
    if projection is not None:
        a = a @ _arr(projection).T
    if center:
        a = a - a.mean(axis=1, keepdims=True)
    d = a @ W.T
    idx = xp.argpartition(-d, k - 1, axis=1)[:, :k]
    vals = xp.ones((len(d), k), dtype=d.dtype) if binary else xp.maximum(xp.take_along_axis(d, idx, axis=1), 0)
    out = xp.zeros_like(d)
    xp.put_along_axis(out, idx, vals, axis=1)
    return out
