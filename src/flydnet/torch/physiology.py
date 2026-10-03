"""생리학: 상태 없는 신경 작용 함수 (torch.nn.functional에 해당)

    import flydnet.torch.physiology as P
    y = P.transmit(x, values, wiring)        # 시냅스 전달        (F.linear처럼, 연결은 커넥톰에 있는 것만)
    s = P.fire(v)                            # 발화 (대리 기울기)
    h = P.inhibit(x, frac=0.05)              # 측억제: 가장 강한 5%만 남김
    r = P.transduce(x, max_rate=100)         # 감각 변환: 값 → 발화율
    c = P.kenyon_code(x, W_pn_kc, k=130)     # KC 희소 부호화
    s = P.recall(a, prototypes)              # 기억 인출: 원형과의 코사인 유사도
    P.reinforce_(prototypes, count, a, y, per_class=10)   # 도파민 강화 (제자리 갱신)

구조물(매개변수를 가진 모듈)은 flydnet.torch.anatomy. torch 없는 기준 엔진은 flydnet.ganglion.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from ..layers import SparsePropagate, SpikeFn


# ─────────────── 시냅스 전달 ───────────────
class Wiring(NamedTuple):
    """희소 배선 구조 (post × pre). values 순서는 wiring()이 돌려준 order를 따름"""
    crow: torch.Tensor
    col: torch.Tensor
    crow_t: torch.Tensor
    col_t: torch.Tensor
    perm_t: torch.Tensor

    @property
    def shape(self):
        return len(self.crow) - 1, len(self.crow_t) - 1

    def to(self, device):
        return Wiring(*(t.to(device) for t in self))


def wiring(post, pre, n_post: int, n_pre: int, device=None) -> tuple[Wiring, np.ndarray]:
    """연결 목록 (post[e], pre[e]) → (Wiring, order). transmit에 줄 values는 values[order] 순서로.
    같은 (post, pre)가 두 번 나오면 오류 (먼저 합칠 것)"""
    post, pre = np.asarray(post, np.int64), np.asarray(pre, np.int64)
    if len(post) and (post.min() < 0 or post.max() >= n_post or pre.min() < 0 or pre.max() >= n_pre):
        raise ValueError("연결 번호가 범위를 벗어남")
    order = np.lexsort((pre, post))                              # 행(post) 순, 그 안에서 열(pre) 순
    post, pre = post[order], pre[order]
    if len(post) > 1 and ((np.diff(post) == 0) & (np.diff(pre) == 0)).any():
        raise ValueError("같은 연결이 두 번 있음 — 시냅스 수를 먼저 합칠 것")
    crow = np.zeros(n_post + 1, np.int64); crow[1:] = np.cumsum(np.bincount(post, minlength=n_post))
    perm_t = np.lexsort((post, pre))                             # 전치: 열(pre) 순
    crow_t = np.zeros(n_pre + 1, np.int64); crow_t[1:] = np.cumsum(np.bincount(pre, minlength=n_pre))
    t = lambda a: torch.tensor(a, dtype=torch.long, device=device)
    return Wiring(t(crow), t(pre), t(crow_t), t(post[perm_t]), t(perm_t)), order


def transmit(x: torch.Tensor, values: torch.Tensor, w: Wiring) -> torch.Tensor:
    """시냅스 전달: (..., n_pre) → (..., n_post). out[post] = Σ values[e] · x[pre_e]
    values와 x 모두로 미분 가능, 역전파 메모리는 연결 수에 비례 (밀집 행렬을 만들지 않음)"""
    n_post, n_pre = w.shape
    if x.shape[-1] != n_pre:
        raise ValueError(f"입력 마지막 차원 {x.shape[-1]} ≠ 배선의 pre 수 {n_pre}")
    lead = x.shape[:-1]
    xt = x.reshape(-1, n_pre).T.contiguous()                     # (n_pre, B)
    out = SparsePropagate.apply(values, xt, *w)                  # (n_post, B)
    return out.T.reshape(*lead, n_post)


# ─────────────── 발화 · 억제 · 변환 ───────────────
def fire(v: torch.Tensor, threshold: float = 0.0, slope: float = 10.0) -> torch.Tensor:
    """발화: v > threshold면 1. 역전파는 빠른 시그모이드 기울기 (대리 기울기, SuperSpike)"""
    return SpikeFn.apply(v - threshold, slope)


def inhibit(x: torch.Tensor, k: int | None = None, frac: float | None = None, dim: int = -1) -> torch.Tensor:
    """측억제 (APL처럼): dim 방향으로 가장 강한 k개(또는 비율 frac)만 남기고 나머지는 0.
    남은 값은 그대로라 기울기도 남은 것에만 흐름"""
    if (k is None) == (frac is None):
        raise ValueError("k와 frac 중 하나만")
    n = x.shape[dim]
    k = int(round(frac * n)) if k is None else int(k)
    k = max(1, min(n, k))
    idx = x.topk(k, dim=dim).indices
    mask = torch.zeros_like(x, dtype=torch.bool).scatter_(dim, idx, True)
    return x * mask


def transduce(x: torch.Tensor, max_rate: float = 100.0) -> torch.Tensor:
    """감각 변환: (B, ...) → 발화율. 음수는 0, 샘플마다 최댓값 = max_rate (패턴이 같으면 세기가 달라도 같은 발화율)"""
    x = x.flatten(1).clamp_min(0)
    return x / x.amax(1, keepdim=True).clamp_min(1e-8) * max_rate


def kenyon_code(x: torch.Tensor, w_pn_kc: torch.Tensor, k: int, projection: torch.Tensor | None = None,
                center: bool = True, binary: bool = False) -> torch.Tensor:
    """KC 희소 부호화 (FlyHash 구조): [투영] → PN 활동 → [평균 빼기] → W_pn_kc → 상위 k만
    w_pn_kc: (n_kc, n_pn), projection: (n_pn, n_in) 또는 None (x가 이미 PN 활동)"""
    a = x.flatten(1).float()
    if projection is not None:
        a = a @ projection.T
    if center:
        a = a - a.mean(1, keepdim=True)
    d = a @ w_pn_kc.T
    top = d.topk(k, dim=1)
    vals = torch.ones_like(top.values) if binary else top.values.clamp_min(0)
    return torch.zeros_like(d).scatter_(1, top.indices, vals)


# ─────────────── 기억: 인출과 도파민 강화 ───────────────
def recall(a: torch.Tensor, prototypes: torch.Tensor, count: torch.Tensor | None = None) -> torch.Tensor:
    """기억 인출: 활동 a (B, n)와 원형 (P, n)의 코사인 유사도 (B, P). count가 0인(빈) 원형은 −inf"""
    s = a @ (prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-8)).T
    return s if count is None else s.masked_fill(count == 0, float("-inf"))


@torch.no_grad()
def reinforce_(prototypes: torch.Tensor, count: torch.Tensor, a: torch.Tensor, y: torch.Tensor, per_class: int):
    """도파민 강화 (제자리 갱신): 클래스 y의 원형 per_class개 중
    - 빈 원형이 있으면 그 클래스의 i번째 샘플이 i번째 빈 원형을 채움
    - 나머지 샘플은 가장 잘 맞는 원형 하나에만 보상 → 그 원형 = 받은 샘플들의 평균 (누적)
    다른 클래스의 원형은 절대 바뀌지 않음 → 클래스를 차례로 배워도 잊지 않음.
    prototypes: (C·per_class, n), count: (C·per_class,)"""
    k = per_class
    C = len(count) // k
    y = y.to(prototypes.device)
    onehot = torch.nn.functional.one_hot(y, C)
    rank = (onehot.cumsum(0) * onehot).sum(1) - 1                  # 클래스 안 순번
    empty = count.view(C, k) == 0
    seed = rank < empty.sum(1)[y]
    if seed.any():
        order = torch.argsort((~empty).float(), dim=1, stable=True)  # 빈 원형 번호가 앞으로
        idx = y[seed] * k + order[y[seed], rank[seed]]
        prototypes[idx] = a[seed]
        count[idx] = 1
    rest = ~seed
    if not rest.any():
        return
    a, y = a[rest], y[rest]
    own = recall(a, prototypes, count).view(len(a), C, k)[torch.arange(len(a), device=a.device), y]
    idx = y * k + own.argmax(1)
    n = torch.zeros_like(count).index_add_(0, idx, torch.ones(len(a), device=a.device, dtype=count.dtype))
    s = torch.zeros_like(prototypes).index_add_(0, idx, a)
    count += n
    hit = n > 0
    prototypes[hit] += (s[hit] - n[hit, None] * prototypes[hit]) / count[hit, None]
