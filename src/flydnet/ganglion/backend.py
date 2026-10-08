"""계산 장치: CPU = NumPy (희소 행렬은 자체 CSR - csr.py), GPU = CuPy. 같은 코드가 양쪽에서 돌도록 배열 모듈(xp)을 고름"""
from __future__ import annotations

import contextlib

import sys

import numpy as np

# CuPy·scipy는 처음 쓸 때 불러옴 (import flydnet이 scipy·cupy 없이도 되고, CuPy의 느린 초기화를 GPU를 쓸 때만)
_GPU = []                                                    # [cupy 모듈 또는 None, 실패 이유]


def _gpu_module():
    """CuPy (GPU가 있을 때) 또는 None. 처음 한 번만 불러 봄"""
    if not _GPU:
        try:
            import cupy as cp
            try:
                cp.cuda.runtime.getDeviceCount()
                _GPU[:] = [cp, None]
            except Exception as e:                           # CuPy는 있지만 GPU·드라이버가 없음
                _GPU[:] = [None, e]
        except ImportError as e:
            _GPU[:] = [None, e]
    return _GPU[0]


def __getattr__(name):                                       # 예전 이름 B._cp (모듈 속성) - 지연 로딩 뒤에도 그대로
    if name == "_cp":
        return _gpu_module()
    if name == "_GPU_ERROR":
        _gpu_module()
        return _GPU[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_HEALTH = []


def gpu_available() -> bool:
    """GPU를 실제로 쓸 수 있는지 (처음 한 번 작은 계산으로 확인 - CuPy는 있어도 CUDA 런타임·NVRTC가 없으면 계산에서 실패)"""
    _cp = _gpu_module()
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
        why = _GPU[1] if _gpu_module() is None else "GPU 계산 시험 실패 (위 경고 참고)"
        raise RuntimeError(f"GPU를 쓸 수 없음 - pip install \"flydnet[gpu-cuda12]\" 또는 [gpu-cuda13] ({why})")
    return device


def xp(device: str):
    """장치의 배열 모듈 (numpy 또는 cupy)"""
    return _gpu_module() if check(device) == "gpu" else np


def sparse(device: str):
    """장치의 희소 행렬 모듈 (scipy.sparse 또는 cupyx.scipy.sparse) - 처음 쓸 때 불러옴. 엔진의 CPU 계산은 쓰지 않음
    (자체 CSR, csr.py) - 예전 코드 호환용"""
    if check(device) == "gpu":
        import cupyx.scipy.sparse as cps
        return cps
    try:
        import scipy.sparse as sps
    except ImportError as e:
        raise ImportError("scipy가 필요한 기능 - pip install scipy") from e
    return sps


def device_of(a) -> str:
    cp = sys.modules.get("cupy")                            # CuPy를 아무도 불러오지 않았으면 CuPy 배열도 있을 수 없음
    return "gpu" if (cp is not None and isinstance(a, cp.ndarray)) else "cpu"


def to(a, device: str):
    """배열을 장치로 (이미 거기 있으면 그대로)"""
    check(device)
    if device_of(a) == device:
        return a
    _cp = _gpu_module()
    return _cp.asarray(a) if device == "gpu" else _cp.asnumpy(a)


def numpy(a) -> np.ndarray:
    return sys.modules["cupy"].asnumpy(a) if device_of(a) == "gpu" else np.asarray(a)


def scatter_add(target, idx, values):
    """target[idx] += values (같은 idx가 여러 번이면 모두 더함). GPU도 결정론적: 같은 입력이면 늘 같은 비트.
    예전: cupy.add.at (원자적 덧셈, 같은 칸에 더하는 순서가 실행마다 달라 마지막 자리가 달라짐) - share="pair"·세포 유형별
    매개변수처럼 여러 연결이 값 하나를 함께 쓰면 학습이 실행마다 갈렸음"""
    if device_of(target) == "gpu":
        _scatter_add_gpu(target, idx, values)
    else:
        np.add.at(target, idx, values)
    return target


_PLANS = {}
_SEGSUM = []


def _scatter_plan(shape, idx):
    """None (칸이 겹치지 않음 - 그냥 더하면 됨) 또는 (쓸 칸 번호, 정렬 순서 int32, 구간 경계, target[idx]의 모양).
    같은 idx 배열(객체)이면 다시 만들지 않음 (학습 버퍼 train_which·pair_of·group_idx는 순전파마다 같은 객체).
    정렬 키 = 칸 번호 x n + 원래 자리 → 겹치는 칸 안의 덧셈 순서가 원래 자리 순서로 정해짐 (정렬 알고리즘과 상관없이)"""
    cp = sys.modules["cupy"]
    key = (id(idx), tuple(shape))
    fp = _index_fingerprint(idx) if isinstance(idx, cp.ndarray) else None
    hit = _PLANS.get(key)
    if hit is not None and hit[0] is idx and hit[1] == fp:          # 같은 객체 + 같은 내용 (제자리에서 바꾼 배열은 다시 만듦)
        return hit[2]
    sel = cp.arange(int(np.prod(shape)), dtype=cp.int64).reshape(shape)[idx]
    pos = sel.reshape(-1)
    n = len(pos)
    plan = None
    if n > 1:
        order = cp.argsort(pos * n + cp.arange(n, dtype=cp.int64))
        sp = pos[order]
        new = cp.concatenate([cp.ones(1, bool), sp[1:] != sp[:-1]])
        if not bool(new.all()):                                       # 같은 칸이 여러 번
            starts = cp.nonzero(new)[0]
            plan = (sp[starts], order.astype(cp.int32 if n < 2 ** 31 else cp.int64),
                    cp.concatenate([starts, cp.asarray([n])]).astype(cp.int64), sel.shape)
    if isinstance(idx, cp.ndarray):                                   # 배열 하나일 때만 (튜플·슬라이스는 매번 새로)
        if len(_PLANS) > 64:
            _PLANS.clear()
        _PLANS[key] = (idx, fp, plan)                                 # 객체를 붙잡아 같은 주소의 다른 배열과 섞이지 않게
    return plan


def _index_fingerprint(idx):
    """인덱스 배열 내용의 지문 (모양, 자료형, 합, 자리 가중합) - 같은 객체를 제자리에서 바꾸면 달라짐. 예전엔 객체만 보고
    예전 계획을 써서 겹침이 생긴 인덱스의 기울기가 조용히 틀렸음 (idx[:] = 0 → [1, 0, 0, 0], 맞는 값 [4, 0, 0, 0])"""
    cp = sys.modules["cupy"]
    f = idx.reshape(-1).astype(cp.int64)
    w = cp.arange(len(f), dtype=cp.int64) % 65521 + 1
    return (idx.shape, idx.dtype.str, int(f.sum()), int((f * w).sum()))


def _scatter_add_gpu(target, idx, values):
    cp = sys.modules["cupy"]
    if isinstance(idx, (list, np.ndarray)) or (isinstance(idx, tuple) and any(isinstance(p, (list, np.ndarray)) for p in idx)):
        idx = tuple(cp.asarray(p) if isinstance(p, (list, np.ndarray)) else p for p in idx) if isinstance(idx, tuple) \
            else cp.asarray(idx)
    plan = _scatter_plan(target.shape, idx)
    if plan is None:                                                  # 칸이 겹치지 않음: 그냥 더하기 (순서 문제 없음)
        target[idx] += values
        return
    pos, order, bounds, sel_shape = plan
    vals = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(values, dtype=target.dtype), sel_shape).reshape(-1))
    if not _SEGSUM:                                                   # 칸마다 스레드 하나가 정렬 순서대로 하나씩 더함
        _SEGSUM.append({})
    kern = _SEGSUM[0].get(order.dtype.name)
    if kern is None:
        it = "int32" if order.dtype == cp.int32 else "int64"
        kern = _SEGSUM[0][order.dtype.name] = cp.ElementwiseKernel(
            f"raw T vals, raw {it} order, raw int64 bounds", "T out",
            "T s = 0; for (long long e = bounds[i]; e < bounds[i + 1]; ++e) s += vals[order[e]]; out = s;",
            f"flydnet_segment_sum_{it}")
    sums = cp.empty(len(pos), dtype=target.dtype)
    kern(vals, order, bounds, sums)
    flat = target.reshape(-1)                                         # 연속 배열이면 같은 메모리
    flat[pos] += sums
    if not target.flags.c_contiguous:                                 # reshape이 사본이었으면 되돌려 씀
        target[...] = flat.reshape(target.shape)


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
    _gpu_module().get_default_memory_pool().set_limit(fraction=fraction)


def gpu_memory_peak_reset():
    if gpu_available():
        _gpu_module().get_default_memory_pool().free_all_blocks()


def gpu_memory_used() -> int:
    """CuPy 메모리 풀이 지금 쓰고 있는 바이트"""
    return int(_gpu_module().get_default_memory_pool().total_bytes()) if gpu_available() else 0


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
        _cp = sys.modules.get("cupy")
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


def class_mask(classes, n_classes: int, xp, dtype):
    """predict(classes=...)의 마스크: 고를 클래스는 0, 나머지는 -inf. 비었거나 범위 밖(음수 포함)이면 오류 - 예전: 빈 목록은
    모든 점수가 -inf라 0번 클래스로, 음수는 파이썬 음수 인덱스로 뒤쪽 클래스를 조용히 골랐음"""
    c = np.asarray(list(classes))
    if c.size == 0:
        raise ValueError("classes가 비어 있음 - 고를 클래스를 하나 이상")
    c = check_labels(c, n_classes, "classes")
    mask = xp.full(n_classes, -xp.inf, dtype=dtype)
    mask[to(c, "gpu" if xp is not np else "cpu")] = 0
    return mask


def check_labels(y, n_classes: int, what: str = "라벨") -> np.ndarray:
    """정수 라벨이 0 ~ n_classes−1 안인지 (음수는 파이썬 음수 인덱스로 조용히 엉뚱한 칸을 고르므로 오류)"""
    y = labels(y)
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        bad = sorted({int(v) for v in y[(y < 0) | (y >= n_classes)]})[:5]
        raise ValueError(f"{what}은 0 ~ {n_classes - 1} 이어야 함: 범위 밖 값 {bad} (클래스 수 {n_classes})")
    return y
