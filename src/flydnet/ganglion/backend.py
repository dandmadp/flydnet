"""계산 장치: CPU = NumPy/SciPy, GPU = CuPy. 같은 코드가 양쪽에서 돌도록 배열 모듈(xp)을 고름"""
from __future__ import annotations

import contextlib

import numpy as np

try:
    import cupy as _cp
    import cupyx as _cpx
    import cupyx.scipy.sparse as _cps
    _GPU_ERROR = None
    try:
        _cp.cuda.runtime.getDeviceCount()
    except Exception as e:                                   # CuPy는 있지만 GPU·드라이버가 없음
        _cp = None
        _GPU_ERROR = e
except ImportError as e:
    _cp = None
    _GPU_ERROR = e

import scipy.sparse as _sps


_HEALTH = []


def gpu_available() -> bool:
    """GPU를 실제로 쓸 수 있는지 (처음 한 번 작은 계산으로 확인 - CuPy는 있어도 CUDA 런타임·NVRTC가 없으면 계산에서 실패)"""
    if _cp is None:
        return False
    if not _HEALTH:
        try:
            ok = float((_cp.arange(4, dtype=_cp.float32) * 2).sum()) == 12.0
            _HEALTH.append(ok)
        except Exception as e:
            import warnings
            warnings.warn(f"CuPy는 있지만 GPU 계산이 안 되어 CPU를 씀 ({type(e).__name__}: {str(e)[:200]}). "
                          "CUDA 런타임이 없다면: pip install \"cupy-cuda12x[ctk]\" (또는 cuda13x)")
            _HEALTH.append(False)
    return _HEALTH[0]


def default_device() -> str:
    """기본 장치: 환경변수 FLYDNET_DEVICE가 cpu·gpu면 그것, 아니면 GPU를 쓸 수 있으면 gpu, 없으면 cpu"""
    import os
    forced = os.environ.get("FLYDNET_DEVICE", "").strip().lower()
    if forced in ("cpu", "gpu"):
        return check(forced)
    return "gpu" if gpu_available() else "cpu"


def check(device: str) -> str:
    if device not in ("cpu", "gpu"):
        raise ValueError(f"장치는 'cpu' 또는 'gpu': {device}")
    if device == "gpu" and not gpu_available():
        why = _GPU_ERROR if _cp is None else "GPU 계산 시험 실패 (위 경고 참고)"
        raise RuntimeError(f"GPU를 쓸 수 없음 - pip install \"flydnet[gpu-cuda12]\" 또는 [gpu-cuda13] ({why})")
    return device


def xp(device: str):
    """장치의 배열 모듈 (numpy 또는 cupy)"""
    return _cp if check(device) == "gpu" else np


def sparse(device: str):
    """장치의 희소 행렬 모듈 (scipy.sparse 또는 cupyx.scipy.sparse)"""
    return _cps if check(device) == "gpu" else _sps


def device_of(a) -> str:
    return "gpu" if (_cp is not None and isinstance(a, _cp.ndarray)) else "cpu"


def to(a, device: str):
    """배열을 장치로 (이미 거기 있으면 그대로)"""
    check(device)
    if device_of(a) == device:
        return a
    return _cp.asarray(a) if device == "gpu" else _cp.asnumpy(a)


def numpy(a) -> np.ndarray:
    return _cp.asnumpy(a) if device_of(a) == "gpu" else np.asarray(a)


def scatter_add(target, idx, values):
    """target[idx] += values (같은 idx가 여러 번이면 모두 더함)"""
    if device_of(target) == "gpu":
        _cp.add.at(target, idx, values)
    else:
        np.add.at(target, idx, values)
    return target


def labels(y) -> np.ndarray:
    """정수 라벨을 numpy int64로 (Signal, numpy, cupy, torch, 리스트 모두)"""
    if hasattr(y, "plastic") and hasattr(y, "data"):           # Signal
        y = y.data
    if device_of(y) == "gpu":
        y = numpy(y)
    elif hasattr(y, "detach"):                                  # torch 텐서
        y = y.detach().cpu().numpy()
    a = np.asarray(y)
    if a.dtype.kind in "fc":                                    # 1.7을 1로 조용히 자르지 않게
        if a.size and not np.all(np.isfinite(a)) or a.size and not np.all(a == np.round(a)):
            bad = a[~(np.isfinite(a) & (a == np.round(a)))][:5].tolist() if a.size else []
            raise ValueError(f"라벨은 정수(클래스 번호): 정수가 아닌 값 {bad} - 확률·원-핫이면 argmax로 번호를 만들 것")
    elif a.dtype.kind not in "biu":
        raise ValueError(f"라벨은 정수(클래스 번호), {a.dtype} 자료형은 안 됨 - 문자열이면: "
                         "names, y = np.unique(y, return_inverse=True)")
    return a.astype(np.int64)


def limit_gpu_memory(fraction: float = 0.75):
    """GPU 메모리 상한 (전체의 비율). 넘치면 바로 오류 → Windows 가상 메모리(C 드라이브)로 흘러가지 않음"""
    if not gpu_available():
        return
    _cp.get_default_memory_pool().set_limit(fraction=fraction)


def gpu_memory_peak_reset():
    if gpu_available():
        _cp.get_default_memory_pool().free_all_blocks()


def gpu_memory_used() -> int:
    """CuPy 메모리 풀이 지금 쓰고 있는 바이트"""
    return int(_cp.get_default_memory_pool().total_bytes()) if gpu_available() else 0


class GPUMemoryError(MemoryError):
    """GPU 메모리 부족 (CuPy OutOfMemoryError에 해결 방법을 붙인 것)"""


@contextlib.contextmanager
def oom_hint(what: str):
    """GPU 메모리가 모자라면 해결 방법을 붙여 GPUMemoryError로 (원래 오류는 __cause__)"""
    try:
        yield
    except GPUMemoryError:
        raise
    except Exception as e:
        if _cp is None or not isinstance(e, _cp.cuda.memory.OutOfMemoryError):
            raise
        pool = _cp.get_default_memory_pool()
        limit = pool.get_limit()
        free, total = _cp.cuda.runtime.memGetInfo()
        raise GPUMemoryError(
            f"{what} 중 GPU 메모리 부족 (flydnet 사용 {pool.used_bytes() / 2**30:.2f} GB"
            f"{f', 상한 {limit / 2**30:.2f} GB' if limit else ''}, GPU 전체 {total / 2**30:.1f} GB 중 남음 {free / 2**30:.2f} GB). "
            "해결: 배치 줄이기 / ConnectomeLayer(checkpoint_every=10 같은 값)로 중간 상태를 다시 계산 / t_ms 줄이기 / "
            "다른 프로그램(torch 등)이 GPU 메모리를 쥐고 있지 않은지 확인 / FLYDNET_DEVICE=cpu") from e


def sample_labels(y, what: str = "라벨") -> np.ndarray:
    """시료마다 클래스 번호 하나인 라벨 (n,) int64. (n, 1) 열 벡터(scikit-learn 습관)는 폄, 그 밖의 2차원 이상은 오류.
    예전: (n, 1)을 그대로 두어 예측 (n,)과 비교할 때 (n, n)으로 퍼져 정확도가 조용히 틀렸음 (fd.evaluate 23.0 등)"""
    y = labels(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"{what}은 시료마다 클래스 번호 하나 (1차원): 모양 {y.shape} - 원-핫이면 argmax로 번호를 만들 것")
    return y


def check_labels(y, n_classes: int, what: str = "라벨") -> np.ndarray:
    """정수 라벨이 0 ~ n_classes−1 안인지 (음수는 파이썬 음수 인덱스로 조용히 엉뚱한 칸을 고르므로 오류)"""
    y = labels(y)
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        bad = sorted({int(v) for v in y[(y < 0) | (y >= n_classes)]})[:5]
        raise ValueError(f"{what}은 0 ~ {n_classes - 1} 이어야 함: 범위 밖 값 {bad} (클래스 수 {n_classes})")
    return y
