"""자체 CPU 희소 행렬 (CSR) - scipy 없이

  A = CSR(data, indices, indptr, shape)     indptr int64, indices int32, data float32 (또는 float64)
  A @ x                                     x (n_cols,) 또는 (n_cols, nb) 밀집 → (n_rows[, nb])
  A.T                                       전치 (구조는 배선마다 한 번 만들어 둠, 값만 다시 모음)

곱셈 경로 (위에서부터 쓸 수 있는 것):
  c      직접 작성한 C 커널 (_csr.c, ctypes). 파이썬 스레드가 행을 나눠 부름 (ctypes는 호출 동안 GIL을 놓음)
  scipy  scipy.sparse (설치되어 있으면)
  numpy  numpy reduceat (느림 - 처음 한 번 경고)
환경변수 FLYDNET_SPARSE=c|scipy|numpy로 고름, FLYDNET_THREADS=n으로 C 경로의 스레드 수.
세 경로 모두 행마다 연결 순서대로 하나씩 더함 → 같은 입력이면 같은 비트 (tests/test_golden.py로 확인).
GPU는 이 모듈을 쓰지 않음 (cupyx 희소 행렬 + kernels.py의 전용 CUDA 커널).
"""
from __future__ import annotations

import ctypes
import os
import threading
import warnings
from pathlib import Path

import numpy as np

_ABI = 1                                     # _csr.c의 flydnet_csr_abi()와 같아야 함
_LIB = []                                    # [ctypes 라이브러리 또는 None, 못 쓰는 이유]
_WARNED = []
_LOCK = threading.Lock()


# ─────────────── C 라이브러리 ───────────────
def _candidates():
    """이 폴더의 _csr 빌드 결과: 파이썬 확장 모듈(휠·setup.py 빌드)이 먼저, 그다음 scripts/build_csr.py의 공유 라이브러리"""
    import importlib.machinery
    here = Path(__file__).resolve().parent
    out = [here / f"_csr{s}" for s in importlib.machinery.EXTENSION_SUFFIXES]
    out += [here / n for n in ("_csr.dll", "_csr.so", "_csr.dylib")]
    return [p for p in out if p.exists()]


def _load():
    if _LIB:
        return _LIB[0]
    with _LOCK:
        if _LIB:
            return _LIB[0]
        lib, why = None, "C 커널 빌드 결과(_csr)가 없음"
        for p in _candidates():
            try:
                cand = ctypes.CDLL(str(p))
                abi = cand.flydnet_csr_abi
                abi.restype = ctypes.c_int64
                if abi() != _ABI:
                    why = f"{p.name}의 규격 번호가 다름 ({abi()} 대 {_ABI}) - 다시 빌드할 것"
                    continue
            except (OSError, AttributeError) as e:
                why = f"{p.name}을 열 수 없음 ({type(e).__name__}: {e})"
                continue
            P, I64 = ctypes.c_void_p, ctypes.c_int64
            for name in ("csr_spmm_f32", "csr_spmm_f64"):
                f = getattr(cand, name)
                f.argtypes = [P, P, P, P, P, I64, I64, I64]
                f.restype = None
            cand.csr_transpose.argtypes = [P, P, I64, I64, P, P, P, P]
            cand.csr_transpose.restype = None
            lib, why = cand, None
            break
        _LIB[:] = [lib, why]
    return lib


def c_available() -> bool:
    """직접 작성한 C 커널을 쓸 수 있는지"""
    return _load() is not None


def _scipy_ok() -> bool:
    try:
        import scipy.sparse  # noqa: F401
        return True
    except ImportError:
        return False


def backend() -> str:
    """지금 쓰는 곱셈 경로: "c" / "scipy" / "numpy" (FLYDNET_SPARSE로 고정 가능)"""
    forced = os.environ.get("FLYDNET_SPARSE", "").strip().lower()
    if forced:
        if forced not in ("c", "scipy", "numpy"):
            raise ValueError(f"FLYDNET_SPARSE는 c, scipy, numpy 중 하나: {forced!r}")
        if forced == "c" and not c_available():
            raise RuntimeError(f"FLYDNET_SPARSE=c인데 C 커널을 쓸 수 없음 ({_LIB[1]})")
        if forced == "scipy" and not _scipy_ok():
            raise RuntimeError("FLYDNET_SPARSE=scipy인데 scipy가 없음 - pip install scipy")
        return forced
    if c_available():
        return "c"
    if _scipy_ok():
        return "scipy"
    if not _WARNED:
        _WARNED.append(True)
        warnings.warn(f"CPU 희소 연산이 느린 경로(numpy)로 돎 - {_LIB[1]}. 플랫폼용 휠로 다시 설치하거나 "
                      "(pip install --force-reinstall flydnet), 소스에서: python scripts/build_csr.py, 또는 pip install scipy",
                      stacklevel=3)
    return "numpy"


# ─────────────── 스레드 ───────────────
_POOL = []


def threads() -> int:
    """C 경로의 스레드 수 (FLYDNET_THREADS, 기본 min(CPU 수, 8))"""
    v = os.environ.get("FLYDNET_THREADS", "").strip()
    if v:
        n = int(v)
        if n < 1:
            raise ValueError(f"FLYDNET_THREADS는 1 이상: {v}")
        return n
    return max(1, min(os.cpu_count() or 1, 8))


def _pool(n):
    if not _POOL or _POOL[0][0] < n:
        from concurrent.futures import ThreadPoolExecutor
        with _LOCK:
            if not _POOL or _POOL[0][0] < n:
                old = _POOL[0][1] if _POOL else None
                _POOL[:] = [(n, ThreadPoolExecutor(n, thread_name_prefix="flydnet-csr"))]
                if old is not None:
                    old.shutdown(wait=False)
    return _POOL[0][1]


if hasattr(os, "register_at_fork"):                     # fork한 자식에는 스레드가 없음 → 새로 만들게
    os.register_at_fork(after_in_child=_POOL.clear)

PARALLEL_MIN = 1 << 18                                   # 연결 수 x 배치가 이보다 작으면 스레드 하나 (나누는 비용이 더 큼)


def _spmm_c(indptr, indices, data, x, out):
    lib = _load()
    f = lib.csr_spmm_f32 if data.dtype == np.float32 else lib.csr_spmm_f64
    n_rows, nb = out.shape
    args = (indptr.ctypes.data, indices.ctypes.data, data.ctypes.data, x.ctypes.data, out.ctypes.data)
    nnz = int(indptr[-1])
    n = threads()
    if n == 1 or nnz * nb < PARALLEL_MIN or n_rows < 2 * n:
        f(*args, 0, n_rows, nb)
        return out
    # 연결 수가 고르게 행을 나눔 (행 하나는 한 스레드가 처음부터 끝까지 → 덧셈 순서가 스레드 수와 상관없이 같음)
    cuts = np.searchsorted(indptr, np.linspace(0, nnz, n + 1)[1:-1]).tolist()
    bounds = sorted(set([0] + cuts + [n_rows]))
    jobs = [_pool(n).submit(f, *args, r0, r1, nb) for r0, r1 in zip(bounds[:-1], bounds[1:]) if r1 > r0]
    for j in jobs:
        j.result()
    return out


def _row_order(indptr):
    """연결이 많은 행부터의 순서와 그 행들의 연결 수·시작 위치 (구조마다 한 번)"""
    def make():
        n = np.diff(indptr)
        order = np.argsort(-n, kind="stable")
        return order, n[order], indptr[:-1][order]
    return _cached(_ROWS, (indptr,), make)


def _spmm_numpy(indptr, indices, data, x, out):
    """모든 행의 k번째 연결을 한꺼번에 더함 (k = 0, 1, ...) → 행마다 연결 순서대로 0에서 하나씩 더함 = C 커널과 같은 비트.
    (np.add.reduceat은 구간 합을 쌍으로 나눠 더해 순서가 다름 - 마지막 자리가 달라짐)"""
    out[...] = 0
    order, n_sorted, starts = _row_order(indptr)
    m = len(n_sorted)
    for k in range(int(n_sorted[0]) if m else 0):
        while m and n_sorted[m - 1] <= k:                              # k번째 연결이 있는 행 = 앞의 m개
            m -= 1
        e = starts[:m] + k
        out[order[:m]] += data[e, None] * x[indices[e]]
    return out


def _spmm_scipy(indptr, indices, data, x, shape):
    import scipy.sparse as sps
    return np.asarray(sps.csr_matrix((data, indices, indptr), shape=shape) @ x)


# ─────────────── 구조 맞추기·전치 (배선마다 한 번) ───────────────
_CANON = {}
_ROWS = {}
_TRANS = {}


def _cached(cache, key_arrays, make):
    """배열 객체가 같으면 다시 만들지 않음 (객체를 붙잡아 같은 주소의 다른 배열과 섞이지 않게)"""
    key = tuple(id(a) for a in key_arrays)
    hit = cache.get(key)
    if hit is not None and all(h is a for h, a in zip(hit[0], key_arrays)):
        return hit[1]
    val = make()
    if len(cache) > 64:
        cache.clear()
    cache[key] = (key_arrays, val)
    return val


def _canonical(indptr, indices):
    """indptr → int64, indices → int32 (열 수가 2^31 미만이면). 이미 그 자료형이면 그대로"""
    def make():
        ip = np.ascontiguousarray(indptr, dtype=np.int64)
        ix = np.asarray(indices)
        if ix.dtype != np.int32 and (len(ix) == 0 or int(ix.max()) < 2 ** 31):
            ix = ix.astype(np.int32)
        return ip, np.ascontiguousarray(ix)
    if isinstance(indptr, np.ndarray) and indptr.dtype == np.int64 and isinstance(indices, np.ndarray) \
            and indices.dtype == np.int32 and indptr.flags.c_contiguous and indices.flags.c_contiguous:
        return indptr, indices
    return _cached(_CANON, (indptr, indices), make)


def transpose_structure(indptr, indices, n_cols: int):
    """전치 구조 (t_indptr int64, t_indices int32, perm int64): 전치의 연결 k = 원래 연결 perm[k].
    열마다 원래 행 순서 (안정 정렬) → 전치의 행 안에서 열 번호가 오름차순"""
    def make():
        n_rows = len(indptr) - 1
        nnz = len(indices)
        lib = _load() if indices.dtype == np.int32 else None
        if lib is not None and nnz:
            t_indptr = np.empty(n_cols + 1, np.int64)
            t_indices = np.empty(nnz, np.int32)
            perm = np.empty(nnz, np.int64)
            count = np.zeros(n_cols, np.int64)
            lib.csr_transpose(indptr.ctypes.data, indices.ctypes.data, n_rows, n_cols, t_indptr.ctypes.data,
                              t_indices.ctypes.data, perm.ctypes.data, count.ctypes.data)
            return t_indptr, t_indices, perm
        perm = np.argsort(indices, kind="stable")
        t_indptr = np.zeros(n_cols + 1, np.int64)
        t_indptr[1:] = np.cumsum(np.bincount(indices, minlength=n_cols))
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))
        t_indices = rows[perm].astype(np.int32 if n_rows < 2 ** 31 else np.int64)
        return t_indptr, t_indices, perm.astype(np.int64)
    return _cached(_TRANS, (indptr, indices), make)


# ─────────────── 행렬 ───────────────
class CSR:
    """CPU 희소 행렬 (CSR). scipy.sparse.csr_matrix 중 엔진이 쓰는 부분만: @, .T, .tocsr(), .toarray(), 속성 data·indices·indptr·shape"""

    __array_priority__ = 20                              # numpy 배열 @ CSR이 numpy 쪽에서 처리되지 않게

    def __init__(self, data, indices, indptr, shape):
        self.indptr, self.indices = _canonical(indptr, indices)
        data = np.asarray(data)
        if data.dtype.kind not in "f":
            data = data.astype(np.float32)
        self.data = np.ascontiguousarray(data)
        self.shape = (int(shape[0]), int(shape[1]))
        if len(self.indptr) != self.shape[0] + 1 or len(self.data) != len(self.indices) or \
                (len(self.indptr) and int(self.indptr[-1]) != len(self.indices)):
            raise ValueError(f"CSR 모양이 맞지 않음: 행 {self.shape[0]}, indptr {len(self.indptr)}, "
                             f"indices {len(self.indices)}, data {len(self.data)}")

    @classmethod
    def from_coo(cls, data, rows, cols, shape):
        """(값, (행, 열)) → CSR. 같은 (행, 열)은 더함 (scipy csr_matrix((data, (i, j)))와 같게). 행 안에서 열 오름차순"""
        rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        data = np.asarray(data)
        order = np.lexsort((cols, rows))
        r, c, d = rows[order], cols[order], data[order]
        if len(r):
            new = np.r_[True, (np.diff(r) != 0) | (np.diff(c) != 0)]
            starts = np.flatnonzero(new)
            d = np.add.reduceat(d, starts) if len(starts) < len(r) else d
            r, c = r[starts], c[starts]
        indptr = np.zeros(int(shape[0]) + 1, np.int64)
        indptr[1:] = np.cumsum(np.bincount(r, minlength=int(shape[0])))
        return cls(d, c, indptr, shape)

    @property
    def nnz(self) -> int:
        return len(self.data)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "CSR":
        t_indptr, t_indices, perm = transpose_structure(self.indptr, self.indices, self.shape[1])
        return CSR(self.data[perm], t_indices, t_indptr, (self.shape[1], self.shape[0]))

    def transpose(self) -> "CSR":
        return self.T

    def tocsr(self) -> "CSR":
        return self

    def sort_indices(self):
        """구조가 늘 정렬되어 있음 (scipy와 같은 이름의 호환용)"""

    def toarray(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.data.dtype)
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        np.add.at(out, (rows, self.indices), self.data)
        return out

    def to_cupy(self):
        """같은 행렬의 cupyx 희소 행렬 (GPU). 색인은 int32 (전용 CUDA 커널이 요구)"""
        import cupy as cp
        import cupyx.scipy.sparse as cps
        idx = np.int32 if self.nnz < 2 ** 31 else np.int64
        return cps.csr_matrix((cp.asarray(self.data), cp.asarray(self.indices.astype(idx, copy=False)),
                               cp.asarray(self.indptr.astype(idx))), shape=self.shape)

    def __matmul__(self, x):
        return spmm(self, x)

    def dot(self, x):
        return spmm(self, x)

    def __repr__(self):
        return f"<CSR {self.shape[0]}x{self.shape[1]}, 연결 {self.nnz}, {self.data.dtype}>"


def spmm(A: CSR, x) -> np.ndarray:
    """A (n_rows, n_cols) @ x (n_cols[, nb]) → 밀집 (n_rows[, nb]). 결과 자료형은 float32·float64 중
    (A.data와 x의 공통 자료형). x는 행 우선이 아니어도 됨 (복사)"""
    x = np.asarray(x)
    one = x.ndim == 1
    if one:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != A.shape[1]:
        raise ValueError(f"모양이 맞지 않음: 행렬 {A.shape} @ {x.shape if not one else x.shape[:1]}")
    dt = np.result_type(A.data.dtype, x.dtype)
    dt = np.float32 if dt == np.float32 else np.float64
    data = A.data.astype(dt, copy=False)
    x = np.ascontiguousarray(x, dtype=dt)
    n_rows, nb = A.shape[0], x.shape[1]
    be = backend()
    if be == "scipy":
        out = _spmm_scipy(A.indptr, A.indices, data, x, A.shape).astype(dt, copy=False)
    else:
        out = np.empty((n_rows, nb), dtype=dt)
        if n_rows and nb:
            if be == "c" and A.indices.dtype == np.int32:
                _spmm_c(A.indptr, A.indices, data, x, out)
            else:
                _spmm_numpy(A.indptr, A.indices, data, x, out)
        else:
            out[...] = 0
    return out[:, 0] if one else out
