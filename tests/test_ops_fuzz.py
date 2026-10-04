"""연산·희소 커널 퍼징을 고정 seed로 (회귀) + 수치 안정성·정확도"""
import warnings

import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B

import fuzz_ops
import fuzz_sparse


def test_ops_fuzz_cpu():
    fails = fuzz_ops.run(8, seed=11, verbose=False)
    assert not fails, {k: v[:2] for k, v in fails.items()}


def test_ops_fuzz_gpu():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    fails = fuzz_ops.run(4, seed=12, device="gpu", verbose=False)
    assert not fails, {k: v[:2] for k, v in fails.items()}


def test_sparse_fuzz():
    fails = fuzz_sparse.run(20, seed=13, verbose=False)
    assert not fails, {k: v[:2] for k, v in fails.items()}


# ─────────────── 수치 안정성·정확도 ───────────────
def test_sigmoid_no_overflow():
    x = fd.Signal(np.array([-1e4, -100., 0., 100., 1e4], np.float32), plastic=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = x.sigmoid()
        s.sum().retrograde()
    np.testing.assert_allclose(s.numpy(), [0, 0, 0.5, 1, 1], atol=1e-6)
    assert np.isfinite(x.retro).all()


def test_std_of_constant_has_zero_gradient():
    c = fd.Signal(np.full((2, 5), 3.0, np.float32), plastic=True)
    c.std().retrograde()
    assert np.isfinite(c.retro).all() and (c.retro == 0).all()


def test_float32_reductions_accumulate_in_float64():
    y = np.array([[1e7 + 1, 1e7 + 2, 1e7 + 3]], np.float32)
    assert abs(float(fd.Signal(y).var().data) - 2 / 3) < 1e-6           # numpy float32는 1.667
    assert fd.Signal(y).sum().data.dtype == np.float32
    z = np.random.default_rng(0).random(10 ** 6).astype(np.float32) + 1000
    exact = z.astype(np.float64).sum()
    assert abs(float(fd.Signal(z).sum().data) - exact) / exact < 1e-7


def test_where_numpy_condition_on_gpu():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    a = fd.Signal(np.ones(4, np.float32), device="gpu")
    out = fd.ganglion.where(np.array([True, False, True, False]), a, 0.0)
    assert out.device == "gpu" and out.numpy().tolist() == [1, 0, 1, 0]


def test_learned_scale_exponent_clamped():
    """학습 배율의 지수를 ±20으로 제한: 학습률이 아주 커도 무한대로 넘치지 않고 부호 유지"""
    rng = np.random.default_rng(0)
    pre, post = rng.integers(0, 10, 60), rng.integers(10, 30, 60)
    key = np.unique(pre * 30 + post)
    c = fd.Circuit.from_edges(key // 30, key % 30, rng.choice([-3.0, 2.0], len(key)), n=30,
                              groups={"A": range(10), "B": range(10, 30)})
    n = fd.Neuropil(c, "A", "B", train="edge", device="cpu")
    s0 = np.sign(n.dense())
    x = fd.Signal(rng.random((3, 10)).astype(np.float32))
    t = fd.Signal(rng.standard_normal((3, 20)).astype(np.float32))
    rule = fd.Plasticity(n.synapses(), rate=50.0)
    for _ in range(20):
        rule.clear(); (n(x) * t).sum().retrograde(); rule.step()
    assert np.isfinite(n(x).numpy()).all()
    nz = s0 != 0
    assert (np.sign(n.dense())[nz] == s0[nz]).all()
