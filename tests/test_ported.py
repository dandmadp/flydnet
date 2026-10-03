"""0.1 기능의 자체 엔진판 (torch 없음) — torch판과 같은 결과인지

난수가 없는 계산은 값이 같아야 함. 난수를 쓰는 부분(무작위 투영, 데이터 생성)은 생성기가 달라 모양·성질만 확인.
"""
import numpy as np
import pytest

import flydnet as fd

torch = pytest.importorskip("torch")
needs_data = pytest.mark.skipif(bool(fd.data.missing("flywire")), reason="FlyWire 데이터 없음")


@pytest.fixture(scope="module")
def mb():
    return fd.Circuit.from_flywire()


def test_rate_encoder_identity_matches_torch():
    x = np.random.default_rng(0).normal(size=(5, 12)).astype(np.float32)
    a = fd.RateEncoder(12, 12, projection=None, device="cpu")(x).numpy()
    b = fd.torch.RateEncoder(12, 12, projection=None)(torch.tensor(x)).numpy()
    np.testing.assert_allclose(a, b, rtol=1e-6)
    r = fd.RateEncoder(50, 20, k=5, device="cpu")
    out = r(np.random.rand(3, 50)).numpy()
    assert out.shape == (3, 20) and np.allclose(out.max(1), 100) and (out >= 0).all()
    assert ((r.P > 0).sum(1) == 5).all()


def test_rate_encoder_is_differentiable():
    import flydnet.ganglion as G
    x = G.Signal(np.random.rand(4, 10), plastic=True)
    fd.RateEncoder(10, 6, k=3, device="cpu")(x).sum().retrograde()
    assert x.retro is not None and np.isfinite(x.retro).all()


@needs_data
def test_glomerular_encoder_and_door_match_torch(mb):
    a, b = fd.GlomerularEncoder(mb, device="cpu"), fd.torch.GlomerularEncoder(mb)
    assert a.glomeruli == b.glomeruli
    x = np.random.default_rng(1).random((3, a.n_glomeruli)).astype(np.float32)
    np.testing.assert_allclose(a(x).numpy(), b(torch.tensor(x)).numpy(), rtol=1e-5)
    if not fd.data.missing("door"):
        da, dt = fd.door_odors(a.glomeruli), fd.torch.door_odors(a.glomeruli)
        assert np.array_equal(da["X"], dt["X"].numpy()) and np.array_equal(da["measured"], dt["measured"].numpy())
        assert list(da["names"]) == list(dt["names"])


@needs_data
def test_kc_expansion_matches_torch_on_pn_input(mb):
    a = fd.KCExpansion(mb, k_frac=0.05, device="cpu")
    b = fd.torch.KCExpansion(mb, k_frac=0.05, device="cpu")
    np.testing.assert_array_equal(a.W, b.W.numpy())
    x = np.random.default_rng(2).random((6, a.n_pn)).astype(np.float32)
    ca, cb = a(x).numpy(), b(torch.tensor(x)).numpy()
    assert ((ca > 0) == (cb > 0)).mean() > 0.999 and np.allclose(ca, cb, atol=1e-3)
    g = fd.KCExpansion(mb, n_in=50, projection="gaussian", k_frac=0.1, binary=True, device="cpu")
    out = g(np.random.rand(4, 50)).numpy()
    assert out.shape == (4, 2597) and (out.sum(1) == g.k).all()


def test_datasets_shapes():
    Xtr, ytr, Xte, yte = fd.synthetic_odors(4, 30, 10, 5, protos_per_class=2, seed=0)
    assert Xtr.shape == (40, 30) and Xte.shape == (20, 30) and Xtr.dtype == np.float32
    assert np.array_equal(fd.synthetic_odors(4, 30, 10, 5, seed=3)[0], fd.synthetic_odors(4, 30, 10, 5, seed=3)[0])
    x, y, p = fd.biconditional_mixtures(np.random.rand(8, 30), [(0, 1, 2, 3), (4, 5, 6, 7)], n=3, seed=0)
    assert x.shape == (24, 30) and set(y) == {0, 1} and set(p) == {0, 1} and (x < 1).all()


def test_train_linear_and_extract():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    r = fd.train_linear(X[:200], y[:200], X[200:], y[200:], epochs=200, device="cpu")   # 200샘플·배치 512 → 에폭당 1스텝
    assert r["test_acc"] > 0.9
    r2 = fd.train_linear(X[:200], y[:200], X[200:], y[200:], epochs=200, device="cpu")
    assert r["test_acc"] == r2["test_acc"]                                    # 같은 seed → 같은 결과
    enc = fd.RateEncoder(5, 5, projection=None, device="cpu")
    feats = fd.extract(lambda z: z * 2, enc, X, batch=64)
    assert feats.shape == (300, 5) and isinstance(feats, np.ndarray)


def test_readout_save_load(tmp_path):
    X = np.random.rand(50, 8).astype(np.float32); y = np.arange(50) % 3
    for r in (fd.AssocReadout(8, 3, per_class=2, device="cpu"), fd.DopamineReadout(8, 3, device="cpu")):
        r.fit(X, y)
        r.save(tmp_path / "r.npz")
        back = type(r).load(tmp_path / "r.npz", device="cpu")
        assert np.array_equal(back.predict(X), r.predict(X))


@needs_data
def test_visual_tools_match_torch():
    import warnings
    vc = fd.visual_circuit()
    a = fd.column_map(vc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = fd.torch.column_map(vc)
    np.testing.assert_allclose(a, b, atol=1e-5, equal_nan=True)
    xy = a[vc.groups["R1-6"]]
    np.testing.assert_allclose(fd.drifting_grating(xy, [0, 45], 60, 12, onset_ms=10),
                               fd.torch.drifting_grating(xy, [0, 45], 60, 12, onset_ms=10).numpy(), atol=1e-5)
