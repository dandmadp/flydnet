"""계산 장치: CPU = NumPy/SciPy, GPU = CuPy. 같은 코드가 양쪽에서 돌도록 배열 모듈(xp)을 고름"""
from __future__ import annotations

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


def gpu_available() -> bool:
    return _cp is not None


def default_device() -> str:
    return "gpu" if gpu_available() else "cpu"


def check(device: str) -> str:
    if device not in ("cpu", "gpu"):
        raise ValueError(f"장치는 'cpu' 또는 'gpu': {device}")
    if device == "gpu" and not gpu_available():
        raise RuntimeError(f"GPU를 쓸 수 없음 (CuPy 설치: pip install flydnet[gpu]) — {_GPU_ERROR}")
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
    return np.asarray(y, dtype=np.int64)


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
