"""시간 시뮬레이션용 합친 연산 (역전파를 직접 유도해 필요한 값만 저장 → 메모리·시간 절약)

신호 배치는 (뉴런 N, 배치 B) - 희소 행렬 곱에 바로 쓰고 전치가 없게.

  propagate(x, values, M, MT, wiring)        out = M @ x,  d values[e] = Σ_b g[post_e, b]·x[pre_e, b]
  lif_step(V, G, I, p_in, spikes, ...)       LIF 한 스텝 (누설 적분 → 입력 주입 → 발화 → 리셋)
  graded_step(V, I, x_in, b, a, ...)         연속값 뉴런 한 스텝
"""
from __future__ import annotations

import numpy as np

from . import backend as B
from .signal import Signal, multi_output

_CUDA_SRC = r"""
extern "C" {
// a group of gw threads per output row (gw = power of two <= 32), threads over batch columns;
// x and out are (n, nb) row-major so neighbouring threads read neighbouring floats
__global__ void spmm_rm(const int* indptr, const int* indices, const float* data, const float* x,
                        float* out, const int n_rows, const int nb, const int gw) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long row = tid / gw;
    const int lane = (int)(tid % gw);
    if (row >= n_rows || lane >= nb) return;
    const int start = indptr[row], end = indptr[row + 1];
    for (int b = lane; b < nb; b += gw) {
        float s = 0.f;
        for (int k = start; k < end; ++k) s += data[k] * x[(long long)indices[k] * nb + b];
        out[row * nb + b] = s;
    }
}
// one thread per edge (small batches)
__global__ void edge_dot_thread(const int* post, const int* pre, const float* g, const float* x,
                                float* out, const long long n_edges, const int nb) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    const float* gr = g + (long long)post[e] * nb;
    const float* xr = x + (long long)pre[e] * nb;
    float s = 0.f;
    for (int b = 0; b < nb; ++b) s += gr[b] * xr[b];
    out[e] = s;
}
// for each edge k (row, indices[k]): sum_b g[row, b] * x[indices[k], b], reduced inside the warp
__global__ void edge_dot_rm(const int* indptr, const int* indices, const float* g, const float* x,
                            float* out, const int n_rows, const int nb) {
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    const int start = indptr[warp], end = indptr[warp + 1];
    const float* gr = g + warp * nb;
    for (int k = start; k < end; ++k) {
        const float* xr = x + (long long)indices[k] * nb;
        float s = 0.f;
        for (int b = lane; b < nb; b += 32) s += gr[b] * xr[b];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) out[k] = s;
    }
}
}
"""
# 커널 설명: spmm_rm = 받는 뉴런(행) 하나를 스레드 gw개가 (배치가 작으면 워프 하나가 행 여러 개), 행 우선이라 연속 읽기
#            edge_dot_rm = 연결마다 Σ_b g[post, b]·x[pre, b], 워프 안에서 합산 / edge_dot_thread = 배치가 작을 때 연결마다 스레드 하나
# 소스는 ASCII만 (CuPy가 시스템 인코딩으로 파일을 써서, 한국어 윈도우(cp949)에서 다른 문자가 있으면 컴파일 실패)
assert _CUDA_SRC.isascii()
_MOD = []


def _cuda():
    """전용 CUDA 커널 모듈. 컴파일이 안 되면 (새 GPU 구조를 모르는 오래된 NVRTC, CuPy 변경 등) 경고 후 None → CuPy 기본 연산"""
    if not _MOD:
        import cupy as cp
        try:
            mod = cp.RawModule(code=_CUDA_SRC)
            for name in ("spmm_rm", "edge_dot_rm", "edge_dot_thread"):   # 여기서 컴파일 (지연 컴파일이라 나중에 실패하지 않게)
                mod.get_function(name)
        except Exception as e:
            import warnings
            warnings.warn(f"flydnet 전용 GPU 커널 컴파일 실패 - CuPy 기본 연산으로 계속함 (결과 같음, 더 느림). "
                          f"({type(e).__name__}: {str(e)[:200]}) python -m flydnet doctor 참고")
            mod = None
        _MOD.append(mod)
    return _MOD[0]


def _gpu_ok(*arrays) -> bool:
    import cupy as cp
    return all(a.dtype == cp.float32 and a.flags.c_contiguous for a in arrays)


def spmm(M, x):
    """희소 CSR (n_rows, n) @ 밀집 (n, nb). GPU·float32면 행 우선 전용 커널 (CuPy 기본보다 몇 배 빠름)"""
    if (B.device_of(x) != "gpu" or x.ndim != 2 or not _gpu_ok(x, M.data) or M.indices.dtype.itemsize != 4
            or _cuda() is None):
        return M @ x
    import cupy as cp
    n_rows, nb = M.shape[0], x.shape[1]
    out = cp.empty((n_rows, nb), dtype=cp.float32)
    gw = min(32, 1 << max(0, (nb - 1).bit_length()))                 # 배치가 작으면 워프 하나가 행 여러 개
    threads = 256
    blocks = (n_rows * gw + threads - 1) // threads
    _cuda().get_function("spmm_rm")((blocks,), (threads,), (M.indptr, M.indices, M.data, x, out,
                                                              np.int32(n_rows), np.int32(nb), np.int32(gw)))
    return out


def edge_dot(g, x, indptr, post, pre, chunk: int = 1 << 26):
    """연결마다 Σ_b g[post_e, b] · x[pre_e, b] (연결은 post 순 CSR 순서). g, x: (N, B) 행 우선"""
    if B.device_of(g) == "gpu":
        import cupy as cp
        g = g if g.flags.c_contiguous else cp.ascontiguousarray(g)
        x = x if x.flags.c_contiguous else cp.ascontiguousarray(x)
        if _gpu_ok(g, x) and pre.dtype.itemsize == 4 and _cuda() is not None:
            n_rows, nb = len(indptr) - 1, g.shape[1]
            out = cp.empty(len(pre), dtype=cp.float32)
            threads = 256
            if nb >= 16:                                             # 워프가 배치 방향으로 나눠 읽고 안에서 합산
                blocks = (n_rows * 32 + threads - 1) // threads
                _cuda().get_function("edge_dot_rm")((blocks,), (threads,), (indptr, pre, g, x, out,
                                                                              np.int32(n_rows), np.int32(nb)))
            else:                                                    # 배치가 작으면 스레드 하나가 연결 하나
                blocks = (len(pre) + threads - 1) // threads
                _cuda().get_function("edge_dot_thread")((blocks,), (threads,), (post, pre, g, x, out,
                                                                                  np.int64(len(pre)), np.int32(nb)))
            return out
        return (g[post] * x[pre]).sum(axis=1)
    E, nb = len(post), g.shape[1]
    out = np.empty(E, dtype=g.dtype)
    step = max(1, chunk // max(1, nb))
    for s in range(0, E, step):
        out[s:s + step] = np.einsum("ij,ij->i", g[post[s:s + step]], x[pre[s:s + step]])
    return out


def matrices(wiring, values_data):
    """(M, MT): 순전파용 CSR과 역전파용 전치 CSR (순전파당 한 번)"""
    M = wiring.matrix(values_data)
    MT = M.T.tocsr()
    if B.device_of(values_data) == "gpu":                           # 커널은 int32 색인 (CuPy 기본)
        MT.sort_indices()
    return M, MT


def propagate(x: Signal, values: Signal, M, MT, wiring) -> Signal:
    """시냅스 전달 (N_pre, B) → (N_post, B)"""
    xd = x.data

    def back(gs):
        g = gs[0]
        if g is None:
            return None, None
        dx = spmm(MT, g) if x.plastic else None
        dv = edge_dot(g, xd, wiring.indptr, wiring.post, wiring.pre) if values.plastic else None
        return dx, dv
    return multi_output((x, values), [spmm(M, xd)], back)[0]


def _reduce_to(g, like):
    """배치 차원을 합쳐 like 모양으로 (v_eq·a가 (N, 1)일 때)"""
    if like is None:
        return None
    return g.sum(axis=1, keepdims=True).reshape(like.shape)


def lif_step(V: Signal, G: Signal, I: Signal, p_in: Signal, spikes, act, in_idx, v_eq, a, gd: float,
             poi_w: float, v_th: float, v_rst: float, scale: float, slope: float):
    """LIF 한 스텝. 반환 (V, G, spk) - torch판 ConnectomeLayer의 한 스텝과 같은 계산
      G1 = G + I;  V1 = act ? V + (v_eq - V + G1)·a : V;  G2 = act ? G1·gd : G1
      V2 = V1 + 입력 스파이크·poi_w (입력 뉴런 행);  spk = (V2 - v_th)/scale > 0
      V3 = spk ? v_rst : V2;  G3 = spk ? 0 : G2
    역전파: 리셋은 기울기 끊음, 발화는 대리 기울기 1/(1 + slope·|u|)², 입력 스파이크는 확률 p_in으로 (straight-through).
    v_eq, a: 숫자 또는 (N, 1) Signal (세포 유형별 매개변수)"""
    xp = B.xp(B.device_of(V.data))
    ve = v_eq.data if isinstance(v_eq, Signal) else v_eq
    aa = a.data if isinstance(a, Signal) else a
    Vd = V.data
    G1 = G.data + I.data
    V1 = xp.where(act, Vd + (ve - Vd + G1) * aa, Vd)
    G2 = xp.where(act, G1 * gd, G1)
    V2 = V1
    V2[in_idx] += spikes * poi_w                                     # V1은 새 배열이라 제자리 수정 가능
    u = (V2 - v_th) * (1.0 / scale)
    fired = u > 0
    spk = fired.astype(Vd.dtype)
    V3 = xp.where(fired, Vd.dtype.type(v_rst), V2)
    G3 = xp.where(fired, Vd.dtype.type(0), G2)
    parents = [V, G, I, p_in] + [s for s in (v_eq, a) if isinstance(s, Signal)]

    # 역전파용으로 남기는 것: Vd, G1, u (실수 3개) + act, fired (참거짓). 구동 항·대리 기울기는 그때 다시 계산
    def back(gs):
        gV3, gG3, gspk = gs
        z = lambda: xp.zeros_like(Vd)
        gV2 = xp.where(fired, 0, gV3) if gV3 is not None else z()
        if gspk is not None:
            gV2 = gV2 + gspk * ((1.0 / scale) / (1 + slope * xp.abs(u)) ** 2)
        gG2 = xp.where(fired, 0, gG3) if gG3 is not None else z()
        g_pin = gV2[in_idx] * poi_w if p_in.plastic else None
        gG1 = xp.where(act, gG2 * gd + gV2 * aa, gG2)
        gV = xp.where(act, gV2 * (1 - aa), gV2)
        out = [gV, gG1, gG1, g_pin]
        if isinstance(v_eq, Signal):
            out.append(_reduce_to(xp.where(act, gV2 * aa, 0), v_eq))
        if isinstance(a, Signal):
            out.append(_reduce_to(xp.where(act, gV2 * (ve - Vd + G1), 0), a))
        return tuple(out)
    return multi_output(parents, [V3, G3, spk], back)


def lif_step_brian(V: Signal, G: Signal, I: Signal, p_in: Signal, spikes, act, in_idx, v_eq, e_v, e_g, gd: float,
                   poi_w: float, v_th: float, v_rst: float, scale: float, slope: float):
    """LIF 한 스텝, Shiu et al. 2024 Brian2 모델과 같은 순서·적분 (ConnectomeLayer timing="brian")
      V1 = act ? v_eq + e_v·(V - v_eq) + e_g·G : V;  G1 = act ? G·gd : G      (정확한 선형 적분 = Brian 'linear')
      spk = (V1 - v_th)/scale > 0
      G2 = act ? G1 + I : G1          불응기 중에 도착한 시냅스 입력은 버림 (Brian2와 같음)
      V2 = V1 + 입력 스파이크·poi_w    발화 판정 뒤에 더함 → 다음 스텝에 발화, 발화한 스텝에 온 것은 리셋으로 사라짐
      V3 = spk ? v_rst : V2;  G3 = spk ? 0 : G2
    v_eq, e_v, e_g: 숫자 또는 (N, 1) Signal (세포 유형별 매개변수 - 역전파 됨)"""
    xp = B.xp(B.device_of(V.data))
    val = lambda s: s.data if isinstance(s, Signal) else s
    ve, ev, eg = val(v_eq), val(e_v), val(e_g)
    Vd, Gd = V.data, G.data
    V1 = xp.where(act, ve + ev * (Vd - ve) + eg * Gd, Vd)
    G1 = xp.where(act, Gd * gd, Gd)
    u = (V1 - v_th) * (1.0 / scale)
    fired = u > 0
    spk = fired.astype(Vd.dtype)
    G2 = xp.where(act, G1 + I.data, G1)
    V1[in_idx] += spikes * poi_w                                     # V1은 새 배열 (u는 이미 계산)
    V3 = xp.where(fired, Vd.dtype.type(v_rst), V1)
    G3 = xp.where(fired, Vd.dtype.type(0), G2)
    coef = [s for s in (v_eq, e_v, e_g) if isinstance(s, Signal)]
    parents = [V, G, I, p_in] + coef

    # 남기는 것: Vd, Gd, u (실수 3개) + act, fired
    def back(gs):
        gV3, gG3, gspk = gs
        z = lambda: xp.zeros_like(Vd)
        gV2 = xp.where(fired, 0, gV3) if gV3 is not None else z()
        gG2 = xp.where(fired, 0, gG3) if gG3 is not None else z()
        g_pin = gV2[in_idx] * poi_w if p_in.plastic else None
        gI = xp.where(act, gG2, 0)
        gV1 = gV2 if gspk is None else gV2 + gspk * ((1.0 / scale) / (1 + slope * xp.abs(u)) ** 2)
        gV1a = xp.where(act, gV1, 0)                                 # 적분이 일어난 칸만
        gV = xp.where(act, gV1 * ev, gV1)
        gG = xp.where(act, gV1 * eg + gG2 * gd, gG2)
        out = [gV, gG, gI, g_pin]
        if isinstance(v_eq, Signal):
            out.append(_reduce_to(gV1a * (1 - ev), v_eq))
        if isinstance(e_v, Signal):
            out.append(_reduce_to(gV1a * (Vd - ve), e_v))
        if isinstance(e_g, Signal):
            out.append(_reduce_to(gV1a * Gd, e_g))
        return tuple(out)
    return multi_output(parents, [V3, G3, spk], back)


def graded_step(V: Signal, I: Signal, x_in: Signal, in_idx, b, a, r_max: float):
    """연속값 뉴런 한 스텝. 반환 (V', r):  V' = V + (b - V + I)·a,  r = clip(V', 0, r_max), 입력 뉴런 행은 r = x_in"""
    xp = B.xp(B.device_of(V.data))
    bb = b.data if isinstance(b, Signal) else b
    aa = a.data if isinstance(a, Signal) else a
    Vd = V.data
    drive = bb - Vd + I.data
    V1 = Vd + drive * aa
    r = xp.clip(V1, 0, r_max)
    r[in_idx] = x_in.data
    pass_ = (V1 >= 0) & (V1 <= r_max)                                # torch.clamp처럼 경계에서도 기울기 통과
    pass_[in_idx] = False
    parents = [V, I, x_in] + [s for s in (b, a) if isinstance(s, Signal)]

    def back(gs):
        gV1n, gr = gs
        gV1 = gV1n if gV1n is not None else xp.zeros_like(Vd)
        if gr is not None:
            gV1 = gV1 + gr * pass_
        g_x = gr[in_idx] if (gr is not None and x_in.plastic) else None
        out = [gV1 * (1 - aa), gV1 * aa, g_x]
        if isinstance(b, Signal):
            out.append(_reduce_to(gV1 * aa, b))
        if isinstance(a, Signal):
            out.append(_reduce_to(gV1 * drive, a))
        return tuple(out)
    return multi_output(parents, [V1, r], back)
