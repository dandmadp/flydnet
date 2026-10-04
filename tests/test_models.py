"""fd.ConnectomeModel·fd.MushroomBody: 한 줄 모델 (입력 Hz 변환·자동 보정·분류 층)"""
import warnings

import numpy as np
import pytest

import flydnet as fd
from flydnet.data import missing


def _graph_task(n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 8)).astype(np.float32)
    return X, (X[:, 0] > X[:, 1]).astype(int)


def _model(**kw):
    c = fd.graphs.layered([20, 100, 10], 0.15, seed=0)
    return fd.ConnectomeModel(c, "in", "out", n_in=8, n_classes=2, device="cpu", **kw)


def test_fit_calibrates_first_and_learns():
    X, y = _graph_task()
    m = _model(trainable=False)
    assert not m.is_calibrated
    h = m.fit(X[:120], y[:120], val=(X[120:], y[120:]), epochs=8, verbose=False)
    assert m.is_calibrated and len(h["val_acc"]) == 8
    with fd.quiescent():
        r = m.layer(m.encoder(X[:16]).data, seed=0).numpy()
    assert abs(r.mean() - m.target_hz) <= 0.3 * m.target_hz               # 출력이 목표 근처로 보정됨
    assert m.score(X[120:], y[120:]) == pytest.approx(h["val_acc"][-1])
    assert m.predict(X[:5]).shape == (5,)


def test_forward_before_calibration_warns_and_calibrates():
    X, _ = _graph_task(32)
    m = _model()
    with pytest.warns(UserWarning, match="보정 전"):
        m(X[:16], seed=0)
    assert m.is_calibrated


def test_save_load_keeps_calibration_and_predictions(tmp_path):
    X, y = _graph_task()
    m = _model()
    m.fit(X, y, epochs=2, verbose=False)
    p = m.save(tmp_path / "m")
    m2 = _model()
    m2.load(p)
    assert m2.is_calibrated and m2.layer.gains == m.layer.gains
    np.testing.assert_array_equal(m2.predict(X), m.predict(X))


def test_synapses_not_registered_twice():
    m = _model()
    names = [n for n, _ in m.named_synapses()]
    assert len(names) == len(set(names)) and len(m.synapses()) == len({id(s) for s in m.synapses()})
    assert m.encoder is not None and m.layer.n_out == 10


def test_trainable_kinds():
    c = fd.graphs.layered([20, 100, 10], 0.15, seed=0)
    assert fd.ConnectomeModel._path_pairs(c, ["in"], ["out"]) == ["h1>out", "in>h1"]
    assert fd.ConnectomeModel._input_pairs(c, ["in"]) == ["in>h1"]
    assert not _model(trainable=False).layer.trainable
    assert _model().layer.trainable


def test_rate_encoder_keeps_information_with_few_features():
    """특징이 k보다 적으면 모든 입력 뉴런이 같은 평균을 받아 시료와 상관없이 전부 max_rate이던 것"""
    e = fd.RateEncoder(8, 20, device="cpu")
    r = e(np.random.default_rng(0).random((3, 8)).astype(np.float32)).numpy()
    assert all(len(np.unique(row.round(3))) > 5 for row in r)
    assert not np.allclose(r[0], r[1])


@pytest.mark.parametrize("kw,match", [(dict(n_classes=1), "n_classes"), (dict(encoder="foo"), "encoder"),
                                      (dict(encoder=None), "입력 특징 수"), (dict(target_hz=-1), "target_hz")])
def test_bad_arguments(kw, match):
    c = fd.graphs.layered([20, 100, 10], 0.15, seed=0)
    with pytest.raises((ValueError, TypeError), match=match):
        fd.ConnectomeModel(c, "in", "out", n_in=8, **dict(dict(n_classes=2), **kw), device="cpu")


@pytest.mark.skipif(missing("flywire") != [] or missing("door") != [], reason="데이터 없음")
def test_mushroom_body_three_lines():
    Xtr, ytr, Xte, yte = fd.door_task(n_odors=6, samples=12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                     # 한 줄 모델은 경고 없이
        m = fd.MushroomBody(Xtr.shape[1], 6, device="cpu", trainable=False)
        h = m.fit(Xtr, ytr, val=(Xte, yte), epochs=6, verbose=False)
    assert type(m.encoder).__name__ == "GlomerularEncoder" and m.target_hz == 5.0
    assert h["val_acc"][-1] > 0.6


def test_single_sample_before_calibration():
    """보정 전에 시료 하나(1차원)를 넣으면 특징을 시료로 세던 것 (특징 > 256이면 특징 일부만 골라 보정)"""
    c = fd.graphs.layered([300, 100, 10], 0.05, seed=0)
    x = np.random.default_rng(0).random(300).astype(np.float32)
    m = fd.ConnectomeModel(c, "in", "out", n_in=300, n_classes=2, device="cpu")
    with pytest.warns(UserWarning, match="보정 전"):
        out = m(x, seed=0)
    assert out.shape == (2,) and m.is_calibrated
