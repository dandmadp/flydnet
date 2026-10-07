"""정답 고정(골든) 시험의 계산: 회로 세 개(버섯체, 시각 운동 경로, 합성)에서 엔진의 결과를 같은 방식으로 다시 계산

  make_golden.py가 이 결과를 tests/golden/*.npz로 저장하고, tests/test_golden.py가 지금 엔진의 결과와 비교.
  모두 CPU, input_mode="regular", 고정 seed. 이 파일의 계산을 바꾸면 골든도 다시 만들어야 함 (값이 달라짐)

  회로마다
    lif_out, lif_spikes          스파이킹 순전파 출력 (B, n_out) · 모든 뉴런의 스텝별 스파이크 (B, steps, N) 0/1 → 정확히 같아야 함
    graded_out                   연속값 뉴런 순전파 출력
    lif_grad_edge, lif_grad_x    trainable=True(share="edge")에서 손실 = Σ out·R 의 연결마다 기울기 · 입력 기울기
    graded_grad_edge, graded_grad_x   연속값 뉴런에서 같은 것
    neuropil_out, neuropil_grad  Neuropil(pre → post) 출력 · 연결마다 기울기
    kc_out                       KCExpansion(pre → post) 출력
"""
from __future__ import annotations

import numpy as np

import flydnet as fd
from flydnet.ganglion import backend as B
from flydnet.ganglion.signal import Signal

T_MS, DT, BATCH = 50.0, 0.5, 2


def circuits():
    """이름 → (회로 만드는 함수, 입력, 출력, Neuropil·KCExpansion의 pre·post, LIF bias, 데이터 필요)"""
    def mb():
        return fd.flywire()

    def visual():
        return fd.visual_circuit().subset(fd.MOTION_PATHWAY + ["LPLC2", "LC4"])

    def synthetic():
        return fd.graphs.layered([20, 60, 10], 0.2, seed=0)
    return {
        "mushroom_body": dict(make=mb, inputs="PN", outputs="MBON", pre="PN", post="KC",
                              gains={"PN>KC": 4.0, "KC>MBON": 4.0}, bias=None, graded_bias=None, data=True),
        # 광수용체(R1-6)는 L 세포를 억제 → 입력만으로는 하류가 조용함. 모든 하류 세포에 9 mV를 더해 자발 발화
        "visual": dict(make=visual, inputs="R1-6", outputs=["LPLC2", "LC4"], pre="Mi1", post="T4a",
                       gains={}, bias=9.0, graded_bias=0.5, data=True),
        "synthetic": dict(make=synthetic, inputs="in", outputs="out", pre="in", post="h1",
                          gains={"in>h1": 2.0, "h1>out": 2.0}, bias=None, graded_bias=None, data=False),
    }


def _bias(c, spec, value):
    if value is None:
        return None
    ins = [spec["inputs"]] if isinstance(spec["inputs"], str) else list(spec["inputs"])
    return {g: value for g in c.groups if g not in ins}


def _layer(c, spec, neuron, trainable=False):
    kw = dict(t_ms=T_MS, dt=DT, device="cpu", input_mode="regular", gains=spec["gains"], neuron=neuron,
              trainable=trainable, share="edge")
    b = _bias(c, spec, spec["bias"] if neuron == "lif" else spec["graded_bias"])
    if b is not None:
        kw["bias"] = b
    return fd.Connectome(c, spec["inputs"], spec["outputs"], **kw)


def _np(a):
    return np.asarray(B.numpy(a.data if isinstance(a, Signal) else a))


def _grad(c, spec, neuron, X):
    """손실 = Σ out · R (R 고정 무작위) → 연결마다 log_scale 기울기, 입력 기울기"""
    layer = _layer(c, spec, neuron, trainable=True)
    x = Signal(X.copy(), plastic=True)
    out = layer(x, seed=1)
    R = np.random.default_rng(7).normal(size=out.data.shape).astype(np.float32)
    (out * Signal(R)).sum().retrograde()
    return _np(layer.log_scale.retro), _np(x.retro)


def compute(name: str) -> dict:
    spec = circuits()[name]
    c = spec["make"]()
    rng = np.random.default_rng(0)
    n_in = sum(len(c.groups[g]) for g in ([spec["inputs"]] if isinstance(spec["inputs"], str) else spec["inputs"]))
    X_hz = rng.uniform(0, 150, (BATCH, n_in)).astype(np.float32)          # 스파이킹: 발화율 Hz
    X_gr = (rng.random((BATCH, n_in)) * 2).astype(np.float32)             # 연속값: 활동
    res = {"X_hz": X_hz, "X_graded": X_gr}

    with fd.quiescent():
        lif = _layer(c, spec, "lif")
        out, rec = lif(X_hz, seed=1, record=np.arange(c.N))
        res["lif_out"] = _np(out)
        res["lif_spikes"] = (np.asarray(rec) > 0).astype(np.uint8)
        res["graded_out"] = _np(_layer(c, spec, "graded")(X_gr, seed=1))

    res["lif_grad_edge"], res["lif_grad_x"] = _grad(c, spec, "lif", X_hz)
    res["graded_grad_edge"], res["graded_grad_x"] = _grad(c, spec, "graded", X_gr)

    # Neuropil / KCExpansion: pre 그룹 활동 (0 ~ 1)
    n_pre = len(c.groups[spec["pre"]])
    Xp = rng.random((4, n_pre)).astype(np.float32)
    npl = fd.Neuropil(c, spec["pre"], spec["post"], train="edge", device="cpu")
    xs = Signal(Xp.copy(), plastic=True)
    o = npl(xs)
    res["neuropil_out"] = _np(o)
    Rn = np.random.default_rng(8).normal(size=o.data.shape).astype(np.float32)
    (o * Signal(Rn)).sum().retrograde()
    res["neuropil_grad"] = np.concatenate([_np(s.retro).ravel() for _, s in npl.named_synapses()])
    res["neuropil_grad_x"] = _np(xs.retro)
    with fd.quiescent():
        res["kc_out"] = _np(fd.KCExpansion(c, pre=spec["pre"], post=spec["post"], device="cpu")(Xp))
    return res
