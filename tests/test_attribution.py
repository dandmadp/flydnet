"""fd.explain: 탐침 기울기 = 가상 손상의 1차 예측, 세포 유형·경로 합, 실제로 꺼서 확인"""
import numpy as np
import pandas as pd
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from test_genetics import _chain


def _graded(c, **kw):
    return fd.ConnectomeLayer(c, "A", ("O",), t_ms=20, neuron="graded", device=kw.pop("device", "cpu"), **kw)


X = np.full((2, 6), 0.8, np.float32)


def score(layer, seed):
    return layer(X, seed=seed).sum()


def test_probe_gradient_matches_finite_difference():
    """뉴런 탐침 기울기 = 그 유형의 배율을 조금 낮췄을 때의 변화 (연속값 뉴런, 수치 미분)"""
    c = _chain()
    L = _graded(c)
    rep = fd.explain(score, L, by="cell_type")
    labels = c.meta.cell_type.to_numpy()
    eps = 1e-3
    with fd.quiescent():
        b0 = float(score(L, 0).data)
        for name in ("h1", "h2", "a"):
            m = np.ones((c.N, 1), np.float32); m[labels == name] = 1 - eps
            L._probe = {"neuron": fd.Signal(m)}
            fd_drop = (b0 - float(score(L, 0).data)) / eps
            L._probe = {}
            pred = float(rep.groups.pred_drop[rep.groups.name == name].iloc[0])
            assert pred == pytest.approx(fd_drop, rel=2e-2), name
    assert set(rep.groups.name) == {"a", "h1", "h2", "o"} and (rep.groups.pred_drop > 0).all()
    assert rep.groups.pred_drop.abs().is_monotonic_decreasing
    assert abs(rep.groups.share.abs().sum() - 1) < 1e-6
    assert not L._probe                                                  # 탐침은 끝나면 치움


def test_pathways_and_edge_gradient():
    c = _chain()
    L = _graded(c)
    rep = fd.explain(score, L, by="cell_type", pathways=True)
    p = rep.pathways.set_index("pathway")
    assert set(p.index) == {"a > h1", "a > h2", "h1 > o", "h2 > o"}
    assert p.edges.sum() == c.n_edges and (p.pred_drop > 0).all()
    eps = 1e-3
    with fd.quiescent():
        b0 = float(score(L, 0).data)
        pre_lab = c.meta.cell_type.to_numpy()[fd.ganglion.backend.numpy(L.wiring.pre)]
        post_lab = c.meta.cell_type.to_numpy()[fd.ganglion.backend.numpy(L.wiring.post)]
        m = np.ones(len(L.w_base), np.float32); m[(pre_lab == "h1") & (post_lab == "o")] = 1 - eps
        L._probe = {"edge": fd.Signal(m)}
        fd_drop = (b0 - float(score(L, 0).data)) / eps
        L._probe = {}
    assert p.pred_drop["h1 > o"] == pytest.approx(fd_drop, rel=2e-2)


def test_verify_compares_with_real_silencing():
    c = _chain()
    L = _graded(c)
    rep = fd.explain(score, L, by="cell_type", verify=2)
    v = rep.verified
    assert len(v) == 4 and set(v.columns) >= {"name", "pred_drop", "actual_drop"}
    assert (v.actual_drop > 0).all()                                     # 사슬의 어느 유형을 꺼도 출력이 줄어듦
    assert rep.agreement is not None and rep.agreement > 0.5
    text = str(rep)
    assert "확인: 실제로 끄기" in text and "순위 상관" in text


def test_explain_spiking_and_groups():
    """스파이킹 층 (대리 기울기)에서도 동작, by="group", 체크포인팅과 같은 결과"""
    c = _chain()
    x = np.full((2, 6), 150.0, np.float32)
    s = lambda L, seed: L(x, seed=seed).sum()
    reps = []
    for ce in (None, 9):
        L = fd.ConnectomeLayer(c, "A", ("O",), t_ms=60, device="cpu", checkpoint_every=ce)
        reps.append(fd.explain(s, L, by="group", seeds=2))
    a, b = (r.groups.set_index("name").pred_drop for r in reps)
    np.testing.assert_allclose(a.sort_index(), b.sort_index(), rtol=1e-4, atol=1e-6)
    assert set(a.index) == {"A", "H", "O"} and a["H"] > 0


def test_explain_errors():
    c = _chain()
    L = _graded(c)
    with pytest.raises(TypeError, match="값 하나"):
        fd.explain(lambda L_, s: L_(X, seed=s), L)
    with pytest.raises(KeyError, match="주석 열"):
        fd.explain(score, L, by="nerve")
    with pytest.raises(TypeError, match="ConnectomeLayer"):
        fd.explain(score, fd.Projection(3, 2))
    assert not L._probe


def test_explain_gpu_matches_cpu():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    c = _chain()
    r = [fd.explain(score, _graded(c, device=d), pathways=True).groups.set_index("name").pred_drop.sort_index()
         for d in ("cpu", "gpu")]
    np.testing.assert_allclose(r[0], r[1], rtol=1e-4)
