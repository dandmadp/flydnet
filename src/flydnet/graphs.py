"""합성 그래프 → Circuit: 커넥톰과 같은 도구(ConnectomeLayer, compare, explain, genetics, ThreeFactor)를 어떤 구조에도

  fd.graphs.erdos_renyi(n, p)                       무작위 (연결 확률 p)
  fd.graphs.watts_strogatz(n, k, beta)              작은 세상망 (고리 이웃 k개, 다시 잇기 확률 beta)
  fd.graphs.barabasi_albert(n, m)                   척도 없는 망 (선호 연결, 새 노드마다 m개)
  fd.graphs.stochastic_block({"A": 100, ...}, p)    블록 구조 (그룹 쌍마다 연결 확률)
  fd.graphs.layered([20, 100, 10], p)               앞먹임 층 (in → h1 → ... → out)

모두 방향 있는 그래프. 공통 인자:
  weight:      연결 세기 (숫자). ConnectomeLayer는 세기에 w_syn(0.275 mV)을 곱하므로 스파이킹이면 5~30 정도
  inhibitory:  억제 노드 비율 (데일의 법칙: 그 노드가 주는 연결은 모두 음수)
  groups:      {이름: 노드 번호} - 입력·출력 그룹 지정 (나머지는 "rest"). stochastic_block·layered는 자동
  seed
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit


def _finish(n, pre, post, weight, inhibitory, groups, seed, name):
    pre, post = np.asarray(pre, np.int64), np.asarray(post, np.int64)
    keep = pre != post                                              # 자기 연결 없음
    pre, post = pre[keep], post[keep]
    key = np.unique(pre * n + post)                                 # 중복 연결 하나로
    pre, post = key // n, key % n
    w = np.full(len(pre), float(weight), np.float32)
    if inhibitory:
        inh = np.random.default_rng([seed, 1]).random(n) < inhibitory
        w[inh[pre]] *= -1
    return Circuit.from_edges(pre, post, w, groups=groups, name=name, n=n)


def erdos_renyi(n: int, p: float, weight: float = 10.0, inhibitory: float = 0.2, groups=None, seed: int = 0) -> Circuit:
    """무작위 방향 그래프: 모든 순서쌍이 확률 p로 연결 (평균 차수 p(n-1))"""
    rng = np.random.default_rng(seed)
    m = rng.binomial(n * (n - 1), p)
    flat = rng.choice(n * n, size=min(int(m * 1.05) + 10, n * n), replace=False)
    pre, post = flat // n, flat % n
    keep = pre != post
    pre, post = pre[keep][:m], post[keep][:m]
    return _finish(n, pre, post, weight, inhibitory, groups, seed, f"Erdos-Renyi (n {n}, p {p})")


def watts_strogatz(n: int, k: int, beta: float, weight: float = 10.0, inhibitory: float = 0.2, groups=None,
                   seed: int = 0) -> Circuit:
    """작은 세상망: 고리에서 각 노드가 앞뒤 k//2개 이웃으로 (양방향), 각 연결의 받는 쪽을 확률 beta로 무작위로 다시 잇기"""
    if k % 2 or k >= n:
        raise ValueError("k는 n보다 작은 짝수")
    rng = np.random.default_rng(seed)
    src = np.repeat(np.arange(n), k)
    off = np.tile(np.concatenate([np.arange(1, k // 2 + 1), -np.arange(1, k // 2 + 1)]), n)
    dst = (src + off) % n
    re = rng.random(len(dst)) < beta
    dst[re] = rng.integers(0, n, re.sum())
    return _finish(n, src, dst, weight, inhibitory, groups, seed, f"Watts-Strogatz (n {n}, k {k}, beta {beta})")


def barabasi_albert(n: int, m: int, weight: float = 10.0, inhibitory: float = 0.2, groups=None, seed: int = 0,
                    reciprocal: float = 0.0) -> Circuit:
    """척도 없는 망: 노드를 하나씩 더하며 이미 있는 노드에 연결 수에 비례한 확률로 m개 연결 (새 노드 → 기존 노드).
    reciprocal: 그 연결의 반대 방향도 만들 확률"""
    if not 1 <= m < n:
        raise ValueError("1 ≤ m < n")
    rng = np.random.default_rng(seed)
    pre, post = [], []
    targets = list(range(m))
    pool = []
    for v in range(m, n):
        for t in set(targets):
            pre.append(v); post.append(t)
            if reciprocal and rng.random() < reciprocal:
                pre.append(t); post.append(v)
        pool.extend(targets); pool.extend([v] * m)
        targets = [pool[i] for i in rng.integers(0, len(pool), m)]
    return _finish(n, pre, post, weight, inhibitory, groups, seed, f"Barabasi-Albert (n {n}, m {m})")


def stochastic_block(sizes: dict, p, weight: float = 10.0, inhibitory: float = 0.2, seed: int = 0) -> Circuit:
    """블록 구조: sizes = {그룹: 노드 수}, p = 숫자 행렬 (그룹 순서, p[a][b] = a → b 확률) 또는
    {("A", "B"): 확률} (없는 쌍은 0). 그룹 = 블록"""
    names = list(sizes)
    if isinstance(p, dict):
        P = np.array([[float(p.get((a, b), 0.0)) for b in names] for a in names])
    else:
        P = np.asarray(p, float)
        if P.shape != (len(names), len(names)):
            raise ValueError(f"p는 {len(names)}x{len(names)}")
    rng = np.random.default_rng(seed)
    start = np.cumsum([0] + [sizes[k] for k in names])
    n = int(start[-1])
    pre, post = [], []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if P[i, j] <= 0:
                continue
            A = rng.random((sizes[a], sizes[b])) < P[i, j]
            r, c = np.nonzero(A)
            pre.append(r + start[i]); post.append(c + start[j])
    pre = np.concatenate(pre) if pre else np.array([], np.int64)
    post = np.concatenate(post) if post else np.array([], np.int64)
    groups = {k: np.arange(start[i], start[i + 1]) for i, k in enumerate(names)}
    return _finish(n, pre, post, weight, inhibitory, groups, seed, f"블록 구조 ({', '.join(f'{k} {v}' for k, v in sizes.items())})")


def layered(sizes, p: float, weight: float = 10.0, inhibitory: float = 0.0, seed: int = 0) -> Circuit:
    """앞먹임 층: sizes = [입력, 숨은..., 출력], 인접한 층 사이만 확률 p로. 그룹 in, h1, h2, ..., out"""
    names = ["in"] + [f"h{i}" for i in range(1, len(sizes) - 1)] + ["out"]
    P = {(names[i], names[i + 1]): p for i in range(len(sizes) - 1)}
    c = stochastic_block(dict(zip(names, sizes)), P, weight=weight, inhibitory=inhibitory, seed=seed)
    c.name = f"앞먹임 층 {list(sizes)} (p {p})"
    return c
