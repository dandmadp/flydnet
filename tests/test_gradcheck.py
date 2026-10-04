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
    best = fd.tune_surrogate(score, L, candidates=(1.0, 0.1), verbose=False)
    assert best in (1.0, 0.1) and L.surrogate_damp == best and L.config["surrogate_damp"] == best
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
