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


@pytest.mark.parametrize("script", sorted(p.name for p in __import__("pathlib").Path(__file__).resolve()
                                          .parents[1].joinpath("examples").glob("*.py")))
def test_example_help_runs(script):
    """예제의 --help가 죽지 않는지 (도움말 문장의 %는 %%로 써야 함 — 실제로 잡힌 버그)"""
    import subprocess, sys, os
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples" / script
    r = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True, encoding="utf-8",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=300)
    assert r.returncode == 0, r.stderr[-500:]


def test_doctor_extra_hint_matches_pyproject():
    """doctor가 권하는 GPU 옵션이 pyproject에 실제로 있고, 새 드라이버에는 지원하는 것 중 가장 새것"""
    import re
    from pathlib import Path
    from flydnet.__main__ import GPU_EXTRAS, _extra_hint
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert sorted(int(v) for v in re.findall(r"^gpu-cuda(\d+)\s*=", text, re.M)) == list(GPU_EXTRAS)
    assert "gpu-cuda12" in _extra_hint(12)
    assert "gpu-cuda13" in _extra_hint(13)
    assert f"gpu-cuda{GPU_EXTRAS[-1]}" in _extra_hint(99)          # 미래 드라이버 → 지원하는 최신 (하위 호환)
    assert "업데이트" in _extra_hint(11)


def test_kernel_compile_failure_falls_back():
    """전용 커널 컴파일이 실패해도 CuPy 기본 연산으로 같은 결과"""
    import numpy as np
    import pytest
    import scipy.sparse as sps
    from flydnet.ganglion import backend as B, kernels as K
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    M = sps.random(60, 50, density=0.2, format="csr", dtype=np.float32, random_state=1)
    x = np.random.default_rng(1).standard_normal((50, 20)).astype(np.float32)
    Mg = B.sparse("gpu").csr_matrix(M); Mg.indices = Mg.indices.astype(np.int32)
    xg = B.to(x, "gpu")
    fast = B.numpy(K.spmm(Mg, xg))
    saved = list(K._MOD)
    try:
        K._MOD[:] = [None]                                           # 컴파일 실패한 상태
        slow = B.numpy(K.spmm(Mg, xg))
        e = K.edge_dot(B.to(np.ones((60, 20), np.float32), "gpu"), xg,
                       B.to(M.indptr.astype(np.int32), "gpu"), B.to(np.repeat(np.arange(60), np.diff(M.indptr)).astype(np.int32), "gpu"),
                       B.to(M.indices.astype(np.int32), "gpu"))
    finally:
        K._MOD[:] = saved
    np.testing.assert_allclose(slow, fast, atol=1e-4)
    np.testing.assert_allclose(B.numpy(e), x[M.indices].sum(1), atol=1e-4)
