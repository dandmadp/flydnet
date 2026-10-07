"""CPU 희소 행렬 곱 경로 비교 (속도·결과): 직접 작성한 C 커널 / scipy / numpy

  python scripts/bench_sparse.py                # 버섯체 (배치 8), 시각계 전체 (배치 32), 전치 구조
  FLYDNET_THREADS=2 python scripts/bench_sparse.py

연결 값은 실제 회로의 시냅스 수 x 0.275 (float32), 입력은 고정 seed 무작위. 결과는 C 경로 기준 최대 상대 차이."""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import flydnet as fd  # noqa: E402
from flydnet.ganglion import csr as S  # noqa: E402
from flydnet.ganglion.physiology import wiring  # noqa: E402


def best(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


def case(name, c, nb, reps):
    w, order = wiring(c.post, c.pre, c.N, c.N)
    vals = (c.weight[order] * 0.275).astype(np.float32)
    A = S.CSR(vals, w.pre, w.indptr, w.shape)
    x = np.random.default_rng(0).random((c.N, nb), dtype=np.float32)
    print(f"\n{name}: N {c.N:,}, 연결 {A.nnz:,}, 배치 {nb}, 스레드 {S.threads()}")
    ref = None
    for be in ("c", "scipy", "numpy"):
        if be == "c" and not S.c_available():
            print("  c      (빌드 없음)")
            continue
        if be == "scipy" and not S._scipy_ok():
            print("  scipy  (설치 안 됨)")
            continue
        os.environ["FLYDNET_SPARSE"] = be
        out = A @ x
        ref = out if ref is None else ref
        diff = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-30))
        same = "같은 비트" if np.array_equal(out, ref) else f"상대 차이 {diff:.2g}"
        ms = best(lambda: A @ x, reps if be != "numpy" else max(1, reps // 10))
        print(f"  {be:<6} {ms:8.2f} ms   {same}")
    os.environ.pop("FLYDNET_SPARSE", None)
    # 전치 구조: C (계수 정렬) 대 numpy (안정 argsort)
    for label, use_c in (("C", True), ("numpy", False)):
        lib = S._LIB[:]
        if not use_c:
            S._LIB[:] = [None, "비교"]

        def tr():
            S._TRANS.clear()
            return S.transpose_structure(A.indptr, A.indices, A.shape[1])
        if use_c and not S.c_available():
            continue
        ms = best(tr, 5)
        S._LIB[:] = lib
        print(f"  전치 구조 {label:<6} {ms:8.2f} ms")
    t1 = S.transpose_structure(A.indptr, A.indices, A.shape[1])
    S._TRANS.clear()
    lib = S._LIB[:]
    S._LIB[:] = [None, "비교"]
    t2 = S.transpose_structure(A.indptr, A.indices, A.shape[1])
    S._LIB[:] = lib
    S._TRANS.clear()
    print("  전치 구조 C == numpy:", all(np.array_equal(a, b) for a, b in zip(t1, t2)))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    case("버섯체 fd.flywire()", fd.flywire(), 8, 200)
    case("시각계 fd.visual_circuit()", fd.visual_circuit(), 32, 20)
