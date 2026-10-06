"""엔진 정리 (scipy·pyarrow 선택화, 자체 CPU 희소 연산) 회귀 시험"""
import sys

import numpy as np
import pytest

import flydnet as fd
from flydnet import data as D


# ─────────────── 2단계: 연결 npz ───────────────
def _flywire_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDNET_FLYWIRE", str(tmp_path))
    return tmp_path


def test_npz_replaces_parquet_in_missing(tmp_path, monkeypatch):
    d = _flywire_dir(tmp_path, monkeypatch)
    assert D.CONNECTIVITY in D.missing("flywire")
    np.savez_compressed(d / D.CONNECTIVITY_NPZ, pre=np.zeros(3, np.int32), post=np.zeros(3, np.int32),
                        weight=np.ones(3, np.int32))
    assert D.CONNECTIVITY not in D.missing("flywire")                   # npz만 복사해 온 컴퓨터도 됨
    st = D.verify("flywire")
    assert st[D.CONNECTIVITY] == "not_needed"
    assert st[D.CONNECTIVITY_NPZ] == "size"                             # 연결 수가 다름


def test_npz_verify_states(tmp_path, monkeypatch):
    d = _flywire_dir(tmp_path, monkeypatch)
    assert D.verify("flywire")[D.CONNECTIVITY_NPZ] == "missing"
    (d / D.CONNECTIVITY_NPZ).write_bytes(b"not a zip")
    assert D.verify("flywire")[D.CONNECTIVITY_NPZ] == "sha256"          # 깨진 파일
    np.savez(d / D.CONNECTIVITY_NPZ, pre=np.zeros(3, np.int32), post=np.zeros(2, np.int32), weight=np.ones(3, np.int32))
    assert D.verify("flywire")[D.CONNECTIVITY_NPZ] == "sha256"          # 길이가 다른 배열


def test_bad_npz_without_parquet_says_how_to_fix(tmp_path, monkeypatch):
    d = _flywire_dir(tmp_path, monkeypatch)
    np.savez(d / D.CONNECTIVITY_NPZ, pre=np.zeros(3, np.int32), post=np.zeros(3, np.int32), weight=np.ones(3, np.int32))
    with pytest.raises(ValueError, match="download flywire"):
        D.read_connectivity()


def test_parquet_without_engine_says_how_to_fix(tmp_path, monkeypatch):
    import pandas as pd
    d = _flywire_dir(tmp_path, monkeypatch)
    (d / D.CONNECTIVITY).write_bytes(b"x")

    def no_engine(*a, **k):
        raise ImportError("Unable to find a usable engine")
    monkeypatch.setattr(pd, "read_parquet", no_engine)
    with pytest.raises(ImportError, match="pip install pyarrow"):
        D.read_connectivity()


def test_custom_npz_connectivity(tmp_path, monkeypatch):
    d = _flywire_dir(tmp_path, monkeypatch)
    np.savez(d / "mine.npz", pre=np.array([0, 1], np.int32), post=np.array([1, 2], np.int32),
             weight=np.array([3, -2], np.int32))
    pre, post, w = D.read_connectivity(name="mine.npz")
    assert pre.dtype == post.dtype == w.dtype == np.int64
    assert w.tolist() == [3, -2]


@pytest.mark.skipif(bool(fd.data.missing("flywire")), reason="FlyWire 데이터 없음")
def test_real_npz_matches_parquet():
    """바꿔 둔 npz와 parquet가 같은 배열 (자료형 포함) - pyarrow가 있을 때만 비교할 수 있음"""
    pytest.importorskip("pyarrow")
    d = D.data_dir("flywire")
    if not (d / D.CONNECTIVITY).exists():
        pytest.skip("parquet 없음 (npz만)")
    a = D._read_parquet(d / D.CONNECTIVITY)
    b = D.read_connectivity()
    assert D.verify("flywire")[D.CONNECTIVITY_NPZ] == "ok"
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and np.array_equal(x, y)


# ─────────────── 1단계: 지연 import ───────────────
def test_import_does_not_load_scipy_or_cupy():
    import subprocess
    code = ("import sys, flydnet; bad = [m for m in ('scipy', 'cupy', 'networkx', 'torch') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"import flydnet이 불러온 선택 모듈: {out.stdout.strip()}"


def test_to_networkx_without_networkx_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "networkx", None)                  # import networkx → ImportError
    c = fd.graphs.layered([3, 4], 0.5, seed=0)
    with pytest.raises(ImportError, match=r"flydnet\[graph\]"):
        c.to_networkx()


# ─────────────── 3단계: 주변부 scipy 교체 ───────────────
def test_t975_matches_t_distribution():
    """scipy.stats.t.ppf(0.975, df) 값 (소수 여섯째 자리)"""
    from flydnet.controls import t975
    for df, want in [(1, 12.706205), (2, 4.302653), (9, 2.262157), (30, 2.042272), (31, 2.039513),
                     (60, 2.000298), (120, 1.979930), (10 ** 6, 1.959966)]:
        assert abs(t975(df) - want) < 3e-6, (df, t975(df), want)


def test_ci95_without_scipy(monkeypatch):
    from flydnet.controls import _ci95
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    lo, hi = _ci95(np.array([0.5, 0.7, 0.6, 0.65]))
    m, h = 0.6125, 3.182446 * np.std([0.5, 0.7, 0.6, 0.65], ddof=1) / 2
    assert abs(lo - (m - h)) < 1e-9 and abs(hi - (m + h)) < 1e-9


def test_reach_hops_numpy_csr():
    """홉 수 BFS는 numpy CSR (예전 scipy). 발화율을 재는 순전파의 scipy 의존은 4단계에서 없앰"""
    c = fd.graphs.layered([5, 8, 4], 0.5, seed=0)
    layer = fd.Connectome(c, "in", "out", t_ms=20, device="cpu")
    t = layer.reach(np.full((1, 5), 80, np.float32)).table
    assert dict(zip(t.group, t.hops)) == {"in": 0.0, "h1": 1.0, "out": 2.0}


# ─────────────── 4단계: 자체 CPU 희소 행렬 ───────────────
from flydnet.ganglion import csr as S  # noqa: E402

PATHS = [p for p in ("c", "scipy", "numpy") if (p != "c" or S.c_available()) and (p != "scipy" or S._scipy_ok())]


def _random_csr(rng, n_rows, n_cols, density, dtype=np.float32, empty_rows=()):
    D = np.where(rng.random((n_rows, n_cols)) < density, rng.standard_normal((n_rows, n_cols)), 0).astype(dtype)
    D[list(empty_rows)] = 0
    r, c = np.nonzero(D)
    indptr = np.zeros(n_rows + 1, np.int64)
    indptr[1:] = np.cumsum(np.bincount(r, minlength=n_rows))
    return S.CSR(D[r, c], c, indptr, D.shape), D


@pytest.fixture(params=PATHS)
def path(request, monkeypatch):
    monkeypatch.setenv("FLYDNET_SPARSE", request.param)
    return request.param


def test_csr_matches_dense(path):
    rng = np.random.default_rng(0)
    A, D = _random_csr(rng, 70, 50, 0.2, empty_rows=(0, 5, 69))       # 빈 행 (처음·중간·끝)
    for nb in (1, 3, 32):
        x = rng.standard_normal((50, nb)).astype(np.float32)
        got = A @ x
        assert got.dtype == np.float32 and got.shape == (70, nb)
        np.testing.assert_allclose(got, D.astype(np.float64) @ x, rtol=1e-5, atol=1e-5)
        assert not got[[0, 5, 69]].any()
    v = rng.standard_normal(50).astype(np.float32)                      # 1차원
    np.testing.assert_allclose(A @ v, D.astype(np.float64) @ v, rtol=1e-5, atol=1e-5)


def test_csr_float64_and_mixed(path):
    rng = np.random.default_rng(1)
    A64, D = _random_csr(rng, 40, 30, 0.3, dtype=np.float64)
    x32 = rng.standard_normal((30, 4)).astype(np.float32)
    out = A64 @ x32                                                     # 공통 자료형 float64
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, D @ x32.astype(np.float64), rtol=1e-12, atol=1e-12)
    A32, _ = _random_csr(rng, 40, 30, 0.3)
    assert (A32 @ x32.astype(np.float64)).dtype == np.float64
    assert (A32 @ (x32 > 0)).dtype == np.float32                        # 불리언·정수 입력 → float32


def test_csr_non_contiguous_input(path):
    rng = np.random.default_rng(2)
    A, D = _random_csr(rng, 30, 40, 0.25)
    big = rng.standard_normal((80, 12)).astype(np.float32)
    for x in (big[::2, :5], np.asfortranarray(big[:40, :7]), big[:40, ::3]):
        np.testing.assert_allclose(A @ x, D.astype(np.float64) @ x, rtol=1e-5, atol=1e-5)


def test_csr_zero_edges_and_empty_shapes(path):
    A = S.CSR(np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(6, np.int64), (5, 4))
    assert (A @ np.ones((4, 3), np.float32)).tolist() == [[0.0] * 3] * 5
    assert A.T.shape == (4, 5) and (A.T @ np.ones((5, 2), np.float32)).shape == (4, 2)
    E = S.CSR(np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(1, np.int64), (0, 4))
    assert (E @ np.ones((4, 3), np.float32)).shape == (0, 3)
    B1, _ = _random_csr(np.random.default_rng(3), 6, 5, 0.5)
    assert (B1 @ np.ones((5, 0), np.float32)).shape == (6, 0)
    with pytest.raises(ValueError, match="모양"):
        B1 @ np.ones((4, 2), np.float32)


def test_csr_transpose_and_coo():
    rng = np.random.default_rng(4)
    A, D = _random_csr(rng, 25, 35, 0.2, empty_rows=(3,))
    T = A.T
    np.testing.assert_array_equal(T.toarray(), D.T)
    np.testing.assert_array_equal(T.T.toarray(), D)
    assert all((np.diff(T.indices[T.indptr[i]:T.indptr[i + 1]]) > 0).all() for i in range(T.shape[0]))   # 행 안에서 정렬
    C = S.CSR.from_coo(np.array([1.0, 2.0, 3.0, 4.0], np.float32), [1, 0, 1, 1], [2, 1, 2, 0], (3, 3))
    assert C.toarray().tolist() == [[0, 2, 0], [4, 0, 4], [0, 0, 0]]  # (1, 2) 두 번 → 1 + 3
    lib = S._LIB[:]
    try:                                                                # C 전치와 numpy 전치가 같은 구조
        S._TRANS.clear()
        t_c = S.transpose_structure(A.indptr, A.indices, A.shape[1])
        S._TRANS.clear()
        S._LIB[:] = [None, "시험"]
        t_np = S.transpose_structure(A.indptr, A.indices, A.shape[1])
    finally:
        S._LIB[:] = lib
        S._TRANS.clear()
    for a, b in zip(t_c, t_np):
        np.testing.assert_array_equal(a, b)


def test_paths_give_same_bits(monkeypatch):
    """C·scipy·numpy 경로가 같은 비트 (행마다 연결 순서대로 더함)"""
    rng = np.random.default_rng(5)
    A, _ = _random_csr(rng, 300, 300, 0.1)
    x = rng.standard_normal((300, 9)).astype(np.float32)
    outs = []
    for p in PATHS:
        monkeypatch.setenv("FLYDNET_SPARSE", p)
        outs.append(A @ x)
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


@pytest.mark.skipif(not S.c_available(), reason="C 커널 빌드 없음")
def test_c_threads_do_not_change_bits(monkeypatch):
    rng = np.random.default_rng(6)
    A, _ = _random_csr(rng, 2000, 2000, 0.05)                           # 연결 x 배치 > PARALLEL_MIN → 나눠 계산
    x = rng.standard_normal((2000, 16)).astype(np.float32)
    monkeypatch.setenv("FLYDNET_SPARSE", "c")
    outs = []
    for t in ("1", "3", "8"):
        monkeypatch.setenv("FLYDNET_THREADS", t)
        outs.append(A @ x)
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


def test_propagate_forward_backward(path):
    """순전파 M @ x, 역전파 dx = MT @ g, 연결 기울기 edge_dot - 밀집 계산과 비교"""
    from flydnet.ganglion import kernels as K
    from flydnet.ganglion.physiology import wiring
    rng = np.random.default_rng(7)
    n = 60
    key = rng.choice(n * n, 500, replace=False)
    post, pre = key // n, key % n
    w, order = wiring(post, pre, n, n)
    vals = rng.standard_normal(len(post)).astype(np.float32)[order]
    D = np.zeros((n, n))
    D[w.post, w.pre] = vals
    M, MT = K.matrices(w, vals)
    assert isinstance(M, S.CSR) and isinstance(MT, S.CSR)
    x = fd.Signal(rng.standard_normal((n, 5)).astype(np.float32), plastic=True)
    v = fd.Signal(vals.copy(), plastic=True)
    out = K.propagate(x, v, M, MT, w)
    np.testing.assert_allclose(out.data, D @ x.data, rtol=1e-5, atol=1e-5)
    g = rng.standard_normal((n, 5)).astype(np.float32)
    (out * fd.Signal(g)).sum().retrograde()
    np.testing.assert_allclose(x.retro, D.T @ g, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v.retro, (g[w.post] * x.data[w.pre]).sum(1), rtol=1e-5, atol=1e-5)


def test_forward_without_scipy(monkeypatch):
    """scipy 없이 CPU 순전파·역전파·reach (C 커널, 없으면 numpy 경로)"""
    import warnings
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    monkeypatch.delenv("FLYDNET_SPARSE", raising=False)
    c = fd.graphs.layered([5, 8, 4], 0.5, seed=0)
    layer = fd.Connectome(c, "in", "out", t_ms=40, device="cpu", trainable=True, gains={"in>h1": 6.0, "h1>out": 6.0})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CPU 희소 연산이 느린 경로")
        out = layer(np.full((1, 5), 120, np.float32), seed=0)
        assert float(out.data.sum()) > 0                                # 출력까지 신호가 감 (역전파도 0이 아님)
        out.sum().retrograde()
        t = layer.reach(np.full((1, 5), 80, np.float32)).table
    assert np.abs(layer.log_scale.retro).sum() > 0
    assert dict(zip(t.group, t.hops)) == {"in": 0.0, "h1": 1.0, "out": 2.0}


def test_numpy_fallback_warns_once(monkeypatch):
    import warnings
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    monkeypatch.delenv("FLYDNET_SPARSE", raising=False)
    monkeypatch.setattr(S, "_LIB", [None, "시험: 빌드 없음"])
    monkeypatch.setattr(S, "_WARNED", [])
    with pytest.warns(UserWarning, match="느린 경로"):
        assert S.backend() == "numpy"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert S.backend() == "numpy"                                   # 두 번째는 조용히


def test_sparse_env_validation(monkeypatch):
    monkeypatch.setenv("FLYDNET_SPARSE", "mkl")
    with pytest.raises(ValueError, match="FLYDNET_SPARSE"):
        S.backend()
    monkeypatch.setenv("FLYDNET_SPARSE", "c")
    monkeypatch.setattr(S, "_LIB", [None, "시험: 빌드 없음"])
    with pytest.raises(RuntimeError, match="C 커널"):
        S.backend()
    monkeypatch.setenv("FLYDNET_THREADS", "0")
    with pytest.raises(ValueError, match="FLYDNET_THREADS"):
        S.threads()


# ─────────────── 6단계: 버그 ───────────────
def _graded_layer(**kw):
    c = fd.graphs.layered([30, 60, 10], 0.2, seed=0)
    return fd.Connectome(c, "in", "out", t_ms=100, dt=1.0, neuron="graded", device="cpu",
                         bias={"h1": 0.5, "out": 0.5}, **kw)


def test_graded_calibrate_accepts_activity_input():
    """연속값 뉴런: 입력이 Hz가 아닌 활동 (0~2) - 예전에는 "입력 뉴런이 거의 발화하지 않음"으로 멈췄음"""
    import warnings
    layer = _graded_layer()
    act = (np.random.default_rng(0).random((4, 20, 30)) * 2).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # 순전파·보정 모두 Hz 경고 없이
        layer(act, seed=0)
        t = layer.calibrate(act)
    out = t[t.group == "out"].iloc[0]
    assert out.role == "목표" and out.target_hz == 5.0                  # 기본 목표 = r_max(10)의 절반
    assert abs(out.after_hz - 5.0) <= 0.5


def test_graded_calibrate_rejects_unreachable_target():
    layer = _graded_layer()
    act = np.ones((2, 30), np.float32)
    with pytest.raises(ValueError, match="r_max"):
        layer.calibrate(act, target=20)
    with pytest.raises(ValueError, match="relay_hz"):
        layer.calibrate(act, target=3, relay_hz=12)
    assert layer._input_spikes(np.ones((2, layer.circuit.N), np.float32)) is None


def test_lif_calibrate_still_rejects_quiet_inputs():
    """스파이킹 뉴런은 그대로: 0~1 값을 Hz로 넣으면 바로 오류"""
    c = fd.graphs.layered([30, 60, 10], 0.2, seed=0)
    layer = fd.Connectome(c, "in", "out", t_ms=50, device="cpu")
    with pytest.raises(ValueError, match="거의 발화하지 않음"):
        layer.calibrate(np.full((2, 30), 0.5, np.float32))


def test_extract_new_and_old_order():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    enc = lambda x: x * 2                                               # noqa: E731
    layer = lambda x: x + 1                                             # noqa: E731
    want = X * 2 + 1
    np.testing.assert_array_equal(fd.extract(layer, X, enc), want)
    np.testing.assert_array_equal(fd.extract(layer, X, encoder=enc, batch=3), want)
    np.testing.assert_array_equal(fd.extract(layer, X), X + 1)
    with pytest.warns(DeprecationWarning, match="예전 순서"):
        np.testing.assert_array_equal(fd.extract(layer, enc, X), want)  # 예전: (layer, encoder, X)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(fd.extract(layer, None, X, batch=2), X + 1)
    t = fd.Projection(3, 2, device="cpu")                                # 구조물(Tissue)도 예전 순서의 encoder로
    with pytest.warns(DeprecationWarning):
        old = fd.extract(layer, t, X)
    np.testing.assert_array_equal(old, fd.extract(layer, X, t))
    with pytest.raises(TypeError, match="데이터"):
        fd.extract(layer, None)
