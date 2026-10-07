"""테스트용 참조 구현 (torch 기본 연산 + torch autograd): 자체 엔진의 legacy LIF·graded 뉴런을 독립으로 다시 계산
0.1.17까지 비교 기준이던 torch판 ConnectomeLayer를 0.1.18에서 빼면서, 같은 수식을 최소한으로 옮겨 둔 것.
연결 배열(배선 순서·학습 위치)만 엔진 층에서 가져오고 계산은 전부 torch로 - 값과 기울기를 엔진과 비교

  out, params = torch_legacy_lif(layer, x)     # 정규 입력(input_mode="regular")만, 결정론적
  out, params = torch_graded(layer, x)
  params: {"log_scale": ..., "bias": ..., "log_t_mbr": ...} torch Parameter (기울기 비교용), x는 requires_grad
"""
from __future__ import annotations

import numpy as np
import torch

from flydnet.ganglion import backend as B


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + ctx.slope * x.abs()) ** 2, None


def _np(a):
    return B.numpy(a)


def _clamp_incl(V, lo, hi):
    """clamp, 경계값에서도 기울기 통과 (엔진 규약 - torch 2.x clamp는 경계에서 0). 막전위가 정확히 0에 머무는
    뉴런(입력 없음, bias 0)에서 차이가 남"""
    inside = (V >= lo) & (V <= hi)
    return torch.where(inside, V, V.clamp(lo, hi).detach())


def _params(layer):
    P = {}
    if layer.trainable:
        P["log_scale"] = torch.nn.Parameter(torch.tensor(_np(layer.log_scale.data), dtype=torch.float64))
    if layer.neuron_params:
        b = layer.bias.data if hasattr(layer.bias, "data") and not isinstance(layer.bias, np.ndarray) else layer.bias
        t = layer.log_t_mbr.data if hasattr(layer.log_t_mbr, "data") and not isinstance(layer.log_t_mbr, np.ndarray) \
            else layer.log_t_mbr
        P["bias"] = torch.nn.Parameter(torch.tensor(_np(b), dtype=torch.float64))
        P["log_t_mbr"] = torch.nn.Parameter(torch.tensor(_np(t), dtype=torch.float64))
    return P


def _dense_W(layer, P):
    """연결 값 (배선 순서) → 밀집 (N, N) [post, pre]. 값 = 원래 세기 x 배율 x w_syn, 학습 위치는 x exp(log_scale)"""
    N = layer.circuit.N
    base = torch.tensor(_np(layer.w_base), dtype=torch.float64)
    if "log_scale" in P:
        pos = torch.tensor(_np(layer.train_pos), dtype=torch.long)
        which = torch.tensor(_np(layer.train_which), dtype=torch.long)
        scale = torch.ones_like(base).index_put((pos,), P["log_scale"].clamp(-20, 20).exp()[which])
        base = base * scale
    post = torch.tensor(_np(layer.wiring.post), dtype=torch.long)
    pre = torch.tensor(_np(layer.wiring.pre), dtype=torch.long)
    return torch.zeros(N, N, dtype=torch.float64).index_put((post, pre), base, accumulate=True)


def _neuron_terms(layer, P, dt):
    p = layer.p
    if "bias" in P:
        gi = torch.tensor(_np(layer.group_idx), dtype=torch.long)
        v_eq = (p["v_0"] if layer.neuron == "lif" else 0.0) + P["bias"][gi][:, None]
        a = (dt / P["log_t_mbr"].exp()[gi][:, None]).clamp(max=1.0)
        return v_eq, a
    return (p["v_0"] if layer.neuron == "lif" else 0.0), dt / p["t_mbr"]


def _frames(x, steps):
    """(B, n_in) 또는 (B, T, n_in) → 스텝 s의 (n_in, B)"""
    if x.dim() == 2:
        return lambda s: x.T
    T = x.shape[1]
    return lambda s: x[:, s * T // steps].T


def torch_legacy_lif(layer, x):
    """timing="legacy" LIF (오일러 적분, 불응기 중 입력 쌓아 둠, 입력 스파이크는 발화 판정 전), 정규 입력.
    surrogate_damp = 1 (legacy 기본). 반환 (출력 발화율 (B, n_out), 매개변수 dict)"""
    assert layer.timing == "legacy" and layer.input_mode == "regular" and layer.neuron == "lif"
    P = _params(layer)
    p, N = layer.p, layer.circuit.N
    dt = p["dt"]; steps = int(round(layer.t_ms / dt))
    dly = max(int(round(p["t_dly"] / dt)), 1); R = dly + 1
    rfc = int(round(p["t_rfc"] / dt))
    gd = float(np.exp(-dt / p["tau"]))
    poi_w = p["w_syn"] * p["f_poi"]
    scale = p["v_th"] - p["v_rst"]
    W = _dense_W(layer, P)
    v_eq, a = _neuron_terms(layer, P, dt)
    in_idx = torch.tensor(_np(layer.in_idx), dtype=torch.long)
    Bn = x.shape[0]
    frame = _frames(x * (dt / 1000.0), steps)
    s_cnt = int(round(layer.count_from_ms / dt))
    rfc_vec = torch.full((N, 1), float(rfc), dtype=torch.float64); rfc_vec[in_idx] = 0
    if layer.v_init == "random":
        V = (p["v_rst"] + torch.tensor(_np(layer.v_frac), dtype=torch.float64)[:, None] * scale).expand(-1, Bn).clone()
    else:
        V = torch.full((N, Bn), p["v_0"], dtype=torch.float64)
    G = torch.zeros(N, Bn, dtype=torch.float64)
    refr = torch.zeros(N, Bn, dtype=torch.float64)
    counts = torch.zeros(N, Bn, dtype=torch.float64)
    phase = torch.tensor(_np(layer.phase0), dtype=torch.float64)[:, None].expand(-1, Bn).clone()
    buf = [torch.zeros(N, Bn, dtype=torch.float64) for _ in range(R)]
    for s in range(steps):
        ps = frame(s)
        G = G + buf[s % R]
        act = refr <= 0
        V = torch.where(act, V + (v_eq - V + G) * a, V)
        G = torch.where(act, G * gd, G)
        phase = phase + ps.detach()
        inp = (phase >= 1).to(torch.float64)
        phase = phase - inp
        inp = inp + (ps - ps.detach())                                     # 값은 스파이크, 기울기는 확률로
        V = V.index_add(0, in_idx, inp * poi_w)
        spk = _Spike.apply((V - p["v_th"]) / scale, layer.slope)
        fired = spk.detach() > 0
        if s >= s_cnt:
            counts = counts + spk
        V = torch.where(fired, torch.full_like(V, p["v_rst"]), V)
        G = torch.where(fired, torch.zeros_like(G), G)
        refr = torch.where(fired, rfc_vec.expand(-1, Bn), refr - 1)
        buf[(s + dly) % R] = W @ spk
    span = layer._span(steps, s_cnt, dt)
    rate = counts.T * (1000.0 / span)
    return rate[:, torch.tensor(_np(layer.out_idx), dtype=torch.long)], P


def torch_graded(layer, x):
    """연속값 뉴런: I = W r, V += (b - V + I) a, r = clamp(V, 0, r_max), 입력 뉴런 r = 입력값. 반환 (평균 활동, 매개변수)"""
    assert layer.neuron == "graded"
    P = _params(layer)
    p, N = layer.p, layer.circuit.N
    dt = p["dt"]; steps = int(round(layer.t_ms / dt))
    r_max = p.get("r_max", 10.0)
    W = _dense_W(layer, P)
    b, a = _neuron_terms(layer, P, dt)
    in_idx = torch.tensor(_np(layer.in_idx), dtype=torch.long)
    Bn = x.shape[0]
    frame = _frames(x, steps)
    s_cnt = int(round(layer.count_from_ms / dt))
    V = torch.zeros(N, Bn, dtype=torch.float64)
    r = torch.zeros(N, Bn, dtype=torch.float64)
    acc = torch.zeros(N, Bn, dtype=torch.float64)
    for s in range(steps):
        I = W @ r
        V = V + (b - V + I) * a
        r = _clamp_incl(V, 0.0, r_max).index_copy(0, in_idx, frame(s))
        if s >= s_cnt:
            acc = acc + r
    mean = acc.T / (steps - s_cnt)
    return mean[:, torch.tensor(_np(layer.out_idx), dtype=torch.long)], P
