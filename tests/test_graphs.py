"""어떤 그래프든 회로로: from_edges / from_scipy / from_networkx, 합성 그래프, 예쁜꼬마선충, 다른 도구와 함께"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

import flydnet as fd


# ─────────────── 만들기 ───────────────
def test_from_edges_labels_groups_and_rest():
    c = fd.Circuit.from_edges(["a", "b", "c", "a"], ["b", "c", "a", "d"], [1, 2, -3, 4],
                              groups={"in": ["a"], "out": ["d"]})
    assert c.N == 4 and c.n_edges == 4
    assert list(c.meta.node) == ["a", "b", "c", "d"]                    # 이름 정렬 순서
    assert c.groups["in"].tolist() == [0] and c.groups["out"].tolist() == [3]
    assert sorted(c.groups["rest"].tolist()) == [1, 2]                    # 그룹 없는 노드 → rest
    np.testing.assert_array_equal(c.weight, [1, 2, -3, 4])
    c2 = fd.Circuit.from_edges([0, 1], [1, 2], n=5, groups=np.array(["x", "x", "y", "y", "y"], dtype=object))
    assert c2.N == 5 and set(c2.groups) == {"x", "y"}
    c3 = fd.Circuit.from_edges(["b", "a"], ["a", "b"], names=["b", "a", "z"])
    assert c3.N == 3 and c3.pre.tolist() == [0, 1]


def test_from_edges_errors():
    with pytest.raises(ValueError, match="NaN"):
        fd.Circuit.from_edges([0, 1], [1, 0], [1.0, np.nan])
    with pytest.raises(ValueError, match="여러 그룹"):
        fd.Circuit.from_edges([0, 1], [1, 0], groups={"a": [0, 1], "b": [1]})
    with pytest.raises(KeyError, match="names에 없는"):
        fd.Circuit.from_edges(["a"], ["q"], names=["a", "b"])
    with pytest.raises(ValueError, match="밖"):
        fd.Circuit.from_edges([0, 5], [1, 0], n=3)
    with pytest.raises(ValueError, match="rest"):
        fd.Circuit.from_edges([0], [1], n=3, groups={"rest": [0]})


def test_scipy_roundtrip_and_orientation():
    A = sps.random(30, 30, density=0.1, format="csr", random_state=0, dtype=np.float32)
    A.setdiag(0); A.eliminate_zeros()
    c = fd.Circuit.from_scipy(A)                                          # A[i, j] = i → j
    assert c.N == 30 and c.n_edges == A.nnz
    assert abs(c.to_scipy() - A).max() < 1e-6
    ct = fd.Circuit.from_scipy(A, orientation="post_pre")                 # A[j, i] = i → j
    assert abs(ct.to_scipy() - A.T).max() < 1e-6
    np.testing.assert_allclose(fd.Circuit.from_scipy(A.toarray()).to_scipy().toarray(), A.toarray())


def test_networkx_roundtrip():
    nx = pytest.importorskip("networkx")
    G = nx.DiGraph()
    G.add_node("s", group="in", size=1.5); G.add_node("m"); G.add_node("t", group="out")
    G.add_edge("s", "m", weight=3.0); G.add_edge("m", "t", weight=-2.0)
    c = fd.Circuit.from_networkx(G)
    assert c.N == 3 and set(c.groups) == {"in", "out", "rest"} and c.meta.loc[0, "size"] == 1.5
    H = c.to_networkx()
    assert H.number_of_edges() == 2 and H[0][1]["weight"] == 3.0 and H.nodes[2]["group"] == "out"
    U = nx.path_graph(4)                                                   # 방향 없음 → 양방향
    assert fd.Circuit.from_networkx(U).n_edges == 6


# ─────────────── 합성 그래프 ───────────────
def test_generators_structure():
    G = fd.graphs
    er = G.erdos_renyi(300, 0.05, inhibitory=0.2, seed=1)
    assert abs(er.n_edges / (300 * 299) - 0.05) < 0.005
    neg = er.weight < 0                                                    # 데일의 법칙: 노드마다 부호 하나
    assert not np.intersect1d(er.pre[neg], er.pre[~neg]).size and 0.1 < neg.mean() < 0.3
    ws = G.watts_strogatz(100, 6, 0.0, inhibitory=0)
    assert ws.n_edges == 600 and np.all(np.bincount(ws.pre, minlength=100) == 6)
    ba = G.barabasi_albert(300, 3)
    assert ba.n_edges == (300 - 3) * 3 and np.all(np.bincount(ba.pre, minlength=300)[3:] == 3)   # 새 노드마다 정확히 m개
    indeg = np.bincount(ba.post, minlength=300)
    assert indeg.max() > 5 * np.median(indeg[indeg > 0])                   # 허브
    sb = G.stochastic_block({"A": 100, "B": 100}, {("A", "A"): 0.2, ("A", "B"): 0.01}, inhibitory=0)
    g = sb.group_of()
    assert {(g[a], g[b]) for a, b in zip(sb.pre, sb.post)} == {("A", "A"), ("A", "B")}
    lay = G.layered([5, 20, 3], 0.5)
    assert set(lay.groups) == {"in", "h1", "out"}
    g = lay.group_of()
    assert {(g[a], g[b]) for a, b in zip(lay.pre, lay.post)} == {("in", "h1"), ("h1", "out")}
    assert G.erdos_renyi(50, 0.1, seed=3).n_edges == G.erdos_renyi(50, 0.1, seed=3).n_edges


def test_regroup():
    c = fd.graphs.erdos_renyi(60, 0.1).regroup({"in": range(10), "out": range(50, 60)})
    assert len(c.groups["in"]) == 10 and len(c.groups["rest"]) == 40 and c.N == 60


# ─────────────── 예쁜꼬마선충 ───────────────
def test_celegans():
    from flydnet.data import missing
    if missing("worm"):
        pytest.skip("worm 데이터 없음 (python -m flydnet download worm)")
    c = fd.Circuit.celegans()
    assert c.N == 448 and len(c.groups["neuron"]) == 300 and len(c.groups["body_muscle"]) == 95
    gaba = np.nonzero(c.meta.gaba.to_numpy())[0]
    assert len(gaba) == 26
    out = np.isin(c.pre, gaba)
    assert (c.weight[out] < 0).all() and (c.weight[~out] > 0).all()        # GABA 뉴런의 화학 시냅스만 억제
    both = fd.Circuit.celegans("both")
    assert both.n_edges == c.n_edges + fd.Circuit.celegans("electrical").n_edges
    ase = fd.genetics.driver(c, node=["ASEL", "ASER"])
    assert len(ase) == 2


# ─────────────── 다른 도구와 함께 ───────────────
def _model():
    c = fd.graphs.stochastic_block({"in": 20, "A": 60, "B": 60, "out": 10},
                                   {("in", "A"): 0.3, ("A", "A"): 0.05, ("A", "B"): 0.1, ("B", "B"): 0.05,
                                    ("B", "out"): 0.3, ("A", "out"): 0.05}, weight=25.0, inhibitory=0.15, seed=0)
    return c, fd.ConnectomeLayer(c, "in", "out", t_ms=60, trainable=True, device="cpu")


def test_tools_work_on_any_graph():
    c, L = _model()
    x = np.random.default_rng(0).uniform(50, 200, (4, 20)).astype(np.float32)
    with fd.quiescent():
        assert L(x, seed=0).numpy().mean() > 0                                # 출력이 발화함
    rep = fd.explain(lambda L_, s: L_(x, seed=s).sum(), L, verify=1)        # 기본 by = group
    assert set(rep.groups.name) == {"in", "A", "B", "out"} and rep.verified is not None
    assert set(fd.genetics.lines(c)) == {"in", "A", "B", "out"}             # 기본 by = group
    with fd.genetics.silence(L, fd.genetics.driver(c, group="B")), fd.quiescent():
        silenced = L(x, seed=0).numpy().mean()
    with fd.quiescent():
        assert silenced < L(x, seed=0).numpy().mean()
    tf = fd.ThreeFactor(L, feedback="random")
    o = tf(x, seed=1); (o * 0.01).sum().retrograde(); tf.assign(o)
    assert np.abs(L.log_scale.retro).sum() > 0
    run = lambda circuit, seed: float(fd.ConnectomeLayer(circuit, "in", "out", t_ms=60, device="cpu")(x, seed=seed).numpy().mean())
    assert run(c, 0) > 0                                                   # 실제 회로가 발화해야 비교가 의미 있음
    rep = fd.compare(run, c, controls=["shuffled", "randomized"], seeds=2, verbose=False)
    assert "randomized" in str(rep)
