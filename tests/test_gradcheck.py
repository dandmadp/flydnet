"""대리 기울기 감쇠·구간 절단 역전파와 fd.gradcheck / fd.tune_surrogate"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from test_threefactor import _rec


def _strong():
    c = _rec(feedback_edges=True)
    return fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)


X = np.random.default_rng(1).uniform(50, 200, (8, 6)).astype(np.float32)
W = np.random.default_rng(2).standard_normal(5).astype(np.float32)


def score(L, s):
    return (L(X, seed=s) * fd.Signal(W)).sum()


def _layer(**kw):
    return fd.ConnectomeLayer(_strong(), "IN", "O", t_ms=80, trainable=True, device="cpu", input_mode="regular", **kw)


def test_damp_keeps_values_scales_gradients():
    c = _rec(feedback_edges=False)                                           # 출력에서 나가는 연결 없음
    c = fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)
    mk = lambda **kw: fd.ConnectomeLayer(c, "IN", "O", t_ms=80, trainable=True, device="cpu", input_mode="regular", **kw)
    a, b = mk(surrogate_damp=1.0), mk(surrogate_damp=0.25)
    with fd.quiescent():
        np.testing.assert_array_equal(a(X, seed=0).numpy(), b(X, seed=0).numpy())   # 값은 그대로
    ga, gb = [], []
    for L, g in ((a, ga), (b, gb)):
        score(L, 0).retrograde()
        g.append(L.log_scale.retro.copy())
    assert np.linalg.norm(gb[0]) < np.linalg.norm(ga[0])                     # 역전파만 줄어듦
    onto = np.isin(B.numpy(a.wiring.post), B.numpy(a.out_idx))
    np.testing.assert_allclose(gb[0][onto], 0.25 * ga[0][onto], rtol=1e-4, atol=1e-6 * np.abs(ga[0]).max())  # 출력 직전 = d배


def test_truncate_cuts_long_paths_only():
    a, b = _layer(), _layer(truncate=30)
    with fd.quiescent():
        np.testing.assert_array_equal(a(X, seed=0).numpy(), b(X, seed=0).numpy())
    score(a, 0).retrograde(); score(b, 0).retrograde()
    assert not np.allclose(a.log_scale.retro, b.log_scale.retro)
    c = _layer(truncate=30, checkpoint_every=7)                              # 체크포인팅과 같은 결과
    score(c, 0).retrograde()
    np.testing.assert_allclose(b.log_scale.retro, c.log_scale.retro, rtol=1e-4, atol=1e-6)


def test_damp_threefactor_consistent():
    """ThreeFactor도 같은 감쇠: 출력 직전 연결에서 역전파와 같은 값"""
    c = _rec(feedback_edges=False)
    c = fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=80, trainable=True, device="cpu", input_mode="regular", surrogate_damp=0.3)
    score(L, 0).retrograde()
    bptt = L.log_scale.retro.copy(); L.log_scale.retro = None
    tf = fd.ThreeFactor(L, feedback="none")
    o = tf(X, seed=0)
    (o * fd.Signal(W)).sum().retrograde(); tf.assign(o)
    onto = np.isin(B.numpy(L.wiring.post), B.numpy(L.out_idx))
    np.testing.assert_allclose(L.log_scale.retro[onto], bptt[onto], rtol=1e-3, atol=1e-6 * np.abs(bptt).max())


def test_gradcheck_and_tune():
    L = _layer(share="pair")
    r = fd.gradcheck(score, L)
    assert set(r.table.columns) >= {"pathway", "bptt", "finite_diff"} and len(r.table) > 1
    assert -1 <= r.cos <= 1 and r.ratio > 0 and "cos" in str(r)
    np.testing.assert_array_equal(L.log_scale.numpy(), np.zeros_like(L.log_scale.numpy()))   # 원래 값으로 되돌림
    before = L.surrogate_damp
    best = fd.tune_surrogate(score, L, candidates=(1.0, 0.1), verbose=False)
    if best is None:                                                         # 기준이 불안정하면 고르지 않고 그대로
        assert L.surrogate_damp == before
    else:
        assert best in (1.0, 0.1) and L.surrogate_damp == best and L.config["surrogate_damp"] == best
    P = _layer(share="pair")
    P.input_mode = "poisson"
    b2 = fd.tune_surrogate(score, P, candidates=(1.0, 0.1), verbose=False, seeds=12)
    assert b2 in (1.0, 0.1, None)
    with pytest.raises(ValueError, match="학습하는 연결"):
        fd.gradcheck(score, fd.ConnectomeLayer(_strong(), "IN", "O", t_ms=20, device="cpu"))


def test_damp_truncate_validation_and_custom_models():
    with pytest.raises(ValueError, match="surrogate_damp"):
        _layer(surrogate_damp=0)
    with pytest.raises(ValueError, match="truncate"):
        _layer(truncate=0)
    with pytest.raises(ValueError, match="시냅스 지연"):
        _layer(truncate=10)
    a = fd.ConnectomeLayer(_strong(), "IN", "O", t_ms=60, trainable=True, device="cpu", neuron=fd.neurons.LIF())
    b = fd.ConnectomeLayer(_strong(), "IN", "O", t_ms=60, trainable=True, device="cpu", neuron=fd.neurons.LIF(),
                           surrogate_damp=0.5, truncate=40)
    score(a, 0).retrograde(); score(b, 0).retrograde()
    assert np.linalg.norm(b.log_scale.retro) < np.linalg.norm(a.log_scale.retro)


def test_gradcheck_reference_reliability():
    """기준(차분) 신뢰도: 매끄러운 연속값 회로는 eps·eps/2 차분이 일치"""
    c = _strong()
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=40, trainable=True, device="cpu", neuron="graded", share="pair")
    Xg = np.random.default_rng(1).uniform(0.2, 1, (8, 6)).astype(np.float32)
    r = fd.gradcheck(lambda L_, s: (L_(Xg, seed=s) * fd.Signal(W)).sum(), L)
    assert r.reliable_reference and r.fd_consistency > 0.99
    assert r.cos > 0.99 and abs(r.ratio - 1) < 0.05                          # 연속값 뉴런은 역전파가 정확
    assert "finite_diff_half" in r.table


def test_gradcheck_seeds_average():
    L = _layer(share="pair")
    L.input_mode = "poisson"
    r = fd.gradcheck(score, L, seeds=3)
    assert r.seeds == 3 and np.isfinite(r.cos)


def test_membrane_noise():
    a, b, c = _layer(), _layer(noise=1.0), _layer(noise=1.0, checkpoint_every=7)
    with fd.quiescent():
        base = a(X, seed=0).numpy()
        n1, n1b, n2 = b(X, seed=0).numpy(), b(X, seed=0).numpy(), b(X, seed=1).numpy()
    np.testing.assert_array_equal(n1, n1b)                                   # 시드로 정해짐
    assert not np.array_equal(n1, base) and not np.array_equal(n1, n2)
    score(b, 0).retrograde(); score(c, 0).retrograde()
    np.testing.assert_allclose(b.log_scale.retro, c.log_scale.retro, rtol=1e-4, atol=1e-6)   # 체크포인팅과 같음
    with pytest.raises(ValueError, match="noise"):
        _layer(noise=-1)


def test_gradcheck_flags_silent_output():
    c = _strong()
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=40, trainable=True, device="cpu", input_mode="regular",
                           gains={"H>O": 0.0001, "IN>H": 0.0001})                # 출력이 발화하지 않게
    r = fd.gradcheck(lambda L_, s: L_(X, seed=s).sum(), L)
    assert r.flat and not r.reliable_reference and "출력이 변하지 않음" in str(r)
