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
