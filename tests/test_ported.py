"""0.1 기능의 자체 엔진판 (torch 없음): 인코더·KC 확장·데이터셋·리드아웃·시각계 도구가 수식대로인지
0.1.17까지는 torch판 복사본과 값을 비교했음 - 0.1.18에서 torch판을 빼며 수식으로 직접 계산한 기대값과 비교로 바꿈
"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B

needs_data = pytest.mark.skipif(bool(fd.data.missing("flywire")), reason="FlyWire 데이터 없음")


@pytest.fixture(scope="module")
def mb():
    return fd.Circuit.from_flywire()


@pytest.mark.filterwarnings("ignore:RateEncoder 입력의")             # 음수를 0으로 자르는 공식을 일부러 시험
def test_rate_encoder_identity_is_formula():
    """특징 하나 = 뉴런 하나: 음수는 0, 시료마다 최댓값 = max_rate"""
    x = np.random.default_rng(0).normal(size=(5, 12)).astype(np.float32)
    a = fd.RateEncoder(12, 12, projection=None, device="cpu")(x).numpy()
    r = np.maximum(x, 0)
    np.testing.assert_allclose(a, r / np.maximum(r.max(1, keepdims=True), 1e-8) * 100, rtol=1e-6)
    enc = fd.RateEncoder(50, 20, k=5, device="cpu")
    out = enc(np.random.rand(3, 50)).numpy()
    assert out.shape == (3, 20) and np.allclose(out.max(1), 100) and (out >= 0).all()
    assert ((B.numpy(enc.P) > 0).sum(1) == 5).all()


def test_rate_encoder_is_differentiable():
    import flydnet.ganglion as G
    x = G.Signal(np.random.rand(4, 10), plastic=True)
    fd.RateEncoder(10, 6, k=3, device="cpu")(x).sum().retrograde()
    assert x.retro is not None and np.isfinite(x.retro).all()


@needs_data
def test_glomerular_encoder_and_door_are_formula(mb):
    """같은 사구체의 단일 사구체형 PN = 같은 발화율 (사구체 반응을 PN으로 복사한 뒤 최댓값 정규화)"""
    a = fd.GlomerularEncoder(mb, device="cpu")
    pm = mb.meta.iloc[mb.groups["PN"]]
    uni = pm.cell_sub_class.astype(str).eq("uniglomerular").values
    glom = pm.cell_type.astype(str).str.split("_").str[0].values
    assert a.glomeruli == sorted(set(glom[uni]))
    x = np.random.default_rng(1).random((3, a.n_glomeruli)).astype(np.float32)
    pn = np.zeros((3, len(pm)), np.float32)
    for i in np.nonzero(uni)[0]:
        pn[:, i] = x[:, a.glomeruli.index(glom[i])]
    np.testing.assert_allclose(a(x).numpy(), pn / pn.max(1, keepdims=True) * 100, rtol=1e-5)
    if not fd.data.missing("door"):
        d = fd.door_odors(a.glomeruli)
        assert d["X"].shape == (len(d["names"]), a.n_glomeruli) and (d["X"] >= 0).all()


@needs_data
def test_kc_expansion_is_formula_on_pn_input(mb):
    """W = 실제 PN→KC 흥분성 시냅스 수, 코드 = PN 활동 평균 빼기 → W → 상위 k만 (음수는 0)"""
    a = fd.KCExpansion(mb, k_frac=0.05, device="cpu")
    P, K = mb.groups["PN"], mb.groups["KC"]
    m = np.isin(mb.pre, P) & np.isin(mb.post, K) & (mb.weight > 0)
    W = np.zeros((len(K), len(P)), np.float32)
    np.add.at(W, (np.searchsorted(K, mb.post[m]), np.searchsorted(P, mb.pre[m])), mb.weight[m])
    np.testing.assert_array_equal(B.numpy(a.W), W)
    x = np.random.default_rng(2).random((6, a.n_pn)).astype(np.float32)
    d = (x - x.mean(1, keepdims=True)) @ W.T
    ref = np.zeros_like(d)
    for i in range(len(d)):
        top = np.argsort(-d[i])[:a.k]
        ref[i, top] = np.maximum(d[i, top], 0)
    np.testing.assert_allclose(a(x).numpy(), ref, rtol=1e-4, atol=1e-3)
    g = fd.KCExpansion(mb, n_in=50, projection="gaussian", k_frac=0.1, binary=True, device="cpu")
    out = g(np.random.rand(4, 50)).numpy()
    assert out.shape == (4, 2597) and (out.sum(1) == g.k).all()


def test_datasets_shapes():
    Xtr, ytr, Xte, yte = fd.synthetic_odors(4, 30, 10, 5, protos_per_class=2, seed=0)
    assert Xtr.shape == (40, 30) and Xte.shape == (20, 30) and Xtr.dtype == np.float32
    assert np.array_equal(fd.synthetic_odors(4, 30, 10, 5, seed=3)[0], fd.synthetic_odors(4, 30, 10, 5, seed=3)[0])
    x, y, p = fd.biconditional_mixtures(np.random.rand(8, 30), [(0, 1, 2, 3), (4, 5, 6, 7)], n=3, seed=0)
    assert x.shape == (24, 30) and set(y) == {0, 1} and set(p) == {0, 1} and (x < 1).all()


@pytest.mark.filterwarnings("ignore:RateEncoder 입력의")             # 표준 정규 특징 - extract 경로만 확인
def test_train_linear_and_extract():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    r = fd.train_linear(X[:200], y[:200], X[200:], y[200:], epochs=200, device="cpu")   # 200샘플·배치 512 → 에폭당 1스텝
    assert r["test_acc"] > 0.9
    r2 = fd.train_linear(X[:200], y[:200], X[200:], y[200:], epochs=200, device="cpu")
    assert r["test_acc"] == r2["test_acc"]                                    # 같은 seed → 같은 결과
    enc = fd.RateEncoder(5, 5, projection=None, device="cpu")
    feats = fd.extract(lambda z: z * 2, X, enc, batch=64)
    assert feats.shape == (300, 5) and isinstance(feats, np.ndarray)


def test_readout_save_load(tmp_path):
    X = np.random.rand(50, 8).astype(np.float32); y = np.arange(50) % 3
    for r in (fd.AssocReadout(8, 3, per_class=2, device="cpu"), fd.DopamineReadout(8, 3, device="cpu")):
        r.fit(X, y)
        r.save(tmp_path / "r.npz")
        back = type(r).load(tmp_path / "r.npz", device="cpu")
        assert np.array_equal(back.predict(X), r.predict(X))


@needs_data
def test_visual_tools():
    """시야 지도: 기둥 세포만 좌표, 결정론적 / 격자: 밝기 = contrast·sin(k·xy - 2π f t + 위상), 시작 전 0"""
    vc = fd.visual_circuit()
    a = fd.column_map(vc)
    np.testing.assert_array_equal(a, fd.column_map(vc))
    assert np.isfinite(a[vc.groups["Mi1"]]).all() and np.isnan(a[vc.groups["HSE"]]).all()
    xy = a[vc.groups["R1-6"]]
    lum = fd.drifting_grating(xy, [0, 45], 60, 12, onset_ms=10, wavelength=8.0, temporal_hz=5.0)
    t = (np.arange(12) + 0.5) * 60 / 12
    th = np.radians([0, 45])
    k = np.stack([np.cos(th), np.sin(th)], 1) * 2 * np.pi / 8.0
    ref = np.sin((k @ np.nan_to_num(xy).T)[:, None, :] - 2 * np.pi * 5.0 * (np.maximum(t - 10, 0) / 1000)[None, :, None])
    ref = ref * (t >= 10)[None, :, None]
    np.testing.assert_allclose(lum, ref, atol=1e-4)


@pytest.mark.parametrize("script", sorted(f"{d}/{p.name}" for d in ("examples", "lab") for p in __import__("pathlib").Path(__file__)
                                          .resolve().parents[1].joinpath(d).glob("*.py") if not p.name.startswith("_")))
def test_example_help_runs(script):
    """예제(examples/, lab/)의 --help가 죽지 않는지 (도움말 문장의 %는 %%로 써야 함 — 실제로 잡힌 버그)"""
    import subprocess, sys, os
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / script
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
