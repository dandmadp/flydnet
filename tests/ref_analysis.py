"""분석 도구의 불변량: 무작위 대조군이 지켜야 할 것 (연결 수·차수·세기 분포·중복·자기 연결·위치 칸),
gradcheck가 정확한 기울기(연속값 뉴런)에서 '믿을 만함', explain 경로 합, ablate·subset·regroup

  python tests/ref_analysis.py
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

import flydnet as fd

bad = []


def ok(name, cond, detail=""):
    if not cond:
        bad.append(f"{name} {detail}")
        print("  ✗", name, detail)


def pair_key(c):
    g = c.group_of()
    return np.char.add(np.char.add(g[c.pre].astype(str), ">"), g[c.post].astype(str))


def deg(c, which, key):
    """(그룹 쌍, 뉴런)마다 연결 수"""
    nodes = c.pre if which == "out" else c.post
    return Counter(zip(key, nodes))


def no_dup_self(c):
    return len(set(zip(c.pre.tolist(), c.post.tolist()))) == c.n_edges and not np.any(c.pre == c.post)


circs = {
    "블록": fd.graphs.stochastic_block({"A": 40, "B": 30, "C": 15}, {("A", "B"): 0.2, ("B", "C"): 0.3, ("A", "A"): 0.1,
                                                                     ("C", "A"): 0.25}, inhibitory=0.3, seed=1),
    "척도 없음": fd.graphs.barabasi_albert(120, 3, groups={"in": range(10), "out": range(100, 120)}, seed=2),
}
try:
    from flydnet.data import missing
    if not missing("flywire"):
        circs["버섯체"] = fd.flywire()
except Exception:                                                        # noqa: BLE001
    pass

for name, c in circs.items():
    key = pair_key(c)
    wsorted = {k: np.sort(c.weight[key == k]) for k in set(key)}
    for seed in (0, 1):
        s = c.shuffled(seed=seed)
        ks = pair_key(s)
        ok(f"{name} shuffled: 쌍별 연결 수", Counter(ks) == Counter(key))
        ok(f"{name} shuffled: 뉴런별 나가는 수", deg(s, "out", ks) == deg(c, "out", key))
        ok(f"{name} shuffled: 뉴런별 받는 수", deg(s, "in", ks) == deg(c, "in", key))
        ok(f"{name} shuffled: 쌍별 세기 분포", all(np.array_equal(np.sort(s.weight[ks == k]), v) for k, v in wsorted.items()))
        ok(f"{name} shuffled: 중복·자기 연결 없음", no_dup_self(s))
        ok(f"{name} shuffled: 배선이 실제로 바뀜", np.mean(s.post != c.post) > 0.3)
        rnd = c.randomized(seed=seed)
        kr = pair_key(rnd)
        ok(f"{name} randomized: 쌍별 연결 수", Counter(kr) == Counter(key))
        ok(f"{name} randomized: 쌍별 세기 분포", all(np.array_equal(np.sort(rnd.weight[kr == k]), v) for k, v in wsorted.items()))
        ok(f"{name} randomized: 중복·자기 연결 없음", no_dup_self(rnd))
        sw = c.shuffled_weights(seed=seed)
        ok(f"{name} shuffled_weights: 배선 그대로", np.array_equal(sw.pre, c.pre) and np.array_equal(sw.post, c.post))
        ok(f"{name} shuffled_weights: 쌍별 세기 분포", all(np.array_equal(np.sort(sw.weight[key == k]), v) for k, v in wsorted.items()))
    # 데일의 법칙: 섞어도 보내는 뉴런의 부호는 그대로 (세기는 pre 쪽 연결에 붙어 있음)
    sgn = {}
    for p_, w_ in zip(c.pre, c.weight):
        sgn.setdefault(p_, set()).add(np.sign(w_))
    dale = all(len(v) == 1 for v in sgn.values())
    if dale:
        s = c.shuffled(seed=3)
        sg2 = {}
        for p_, w_ in zip(s.pre, s.weight):
            sg2.setdefault(p_, set()).add(np.sign(w_))
        ok(f"{name} shuffled: 데일의 법칙 유지", sg2 == sgn)

# local: 같은 위치 칸 안에서만 받는 뉴런이 바뀜
c = circs["블록"]
xy = np.random.default_rng(0).random((c.N, 2)) * 6
loc = c.shuffled(seed=0, local=(xy, 2.0))
cell = lambda i: tuple(np.floor(xy[i] / 2.0).astype(int))
ok("local: 받는 뉴런의 위치 칸 유지", all(cell(a) == cell(b) for a, b in zip(c.post, loc.post)))
ok("local: 중복·자기 연결 없음", no_dup_self(loc))

# pairs·exclude: 지정한 쌍만 바뀜
key = pair_key(c)
s = c.shuffled(seed=0, pairs=["A>B"])
ok("pairs: 다른 쌍은 그대로", np.array_equal(s.post[key != "A>B"], c.post[key != "A>B"]))
s = c.shuffled(seed=0, exclude=["A>B"])
ok("exclude: 그 쌍은 그대로", np.array_equal(s.post[key == "A>B"], c.post[key == "A>B"]))

# ablate·subset·regroup
line = fd.genetics.driver(c, group="B")
ab = fd.genetics.ablate(c, line)
ok("ablate: B가 주고받는 연결 없음", not np.isin(ab.pre, line.idx).any() and not np.isin(ab.post, line.idx).any())
ok("ablate: 나머지 연결 그대로", ab.n_edges == np.sum(~(np.isin(c.pre, line.idx) | np.isin(c.post, line.idx))))
sub = c.subset(["A", "C"])
ok("subset: 연결 수", sub.n_edges == np.sum(np.isin(c.pre, np.r_[c.groups["A"], c.groups["C"]]) &
                                         np.isin(c.post, np.r_[c.groups["A"], c.groups["C"]])))
ok("subset: 세기 합", np.isclose(sub.weight.sum(), c.weight[np.isin(c.pre, np.r_[c.groups["A"], c.groups["C"]]) &
                                                          np.isin(c.post, np.r_[c.groups["A"], c.groups["C"]])].sum()))
rg = c.regroup({"x": c.groups["A"][:5]})
ok("regroup: 연결 그대로, 나머지는 rest", np.array_equal(rg.pre, c.pre) and len(rg.groups["rest"]) == c.N - 5)

# gradcheck: 연속값 뉴런은 기울기가 정확 → cos ≈ 1, 크기 비율 ≈ 1, '믿을 만함'
g = fd.graphs.layered([8, 30, 5], 0.3, weight=0.5, seed=0)
L = fd.Connectome(g, "in", "out", t_ms=6, neuron="graded", trainable=True, device="cpu", share="pair")
X = np.random.default_rng(1).random((4, 8)).astype(np.float32)
W = np.random.default_rng(2).standard_normal((4, 5)).astype(np.float32)
gc = fd.gradcheck(lambda L_, s: (L_(X, seed=s) * fd.Signal(W)).sum(), L, eps=0.01)
ok("gradcheck: 정확한 기울기에서 cos > 0.999", gc.cos > 0.999, gc.cos)
ok("gradcheck: 크기 비율 ≈ 1", abs(gc.ratio - 1) < 0.01, gc.ratio)
ok("gradcheck: '믿을 만함'", "믿을 만함" in str(gc))

# explain: 경로별 예측의 합 = 연결마다 기울기 합 (pathways=True), 유형 합 = 뉴런 합
L2 = fd.Connectome(g, "in", "out", t_ms=6, neuron="graded", device="cpu")
e = fd.explain(lambda L_, s: (L_(X, seed=s) * fd.Signal(W)).sum(), L2, by="group", pathways=True)
ok("explain: 유형 합 = 뉴런 합", np.isclose(e.groups.pred_drop.sum(), e.neurons.sum(), rtol=1e-4))
# 연속값 뉴런은 선형 구간에서 끄기 = 1차 예측 (활동 r이 r_max·0에 걸리지 않으면 정확)
e2 = fd.explain(lambda L_, s: (L_(X, seed=s) * fd.Signal(W)).sum(), L2, by="group", verify=2)
v = e2.verified
ok("explain: 확인한 유형 수", len(v) >= 2)

print(f"분석 도구: 문제 {len(bad)}개")
sys.exit(1 if bad else 0)
