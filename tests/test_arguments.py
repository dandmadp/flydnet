"""인자 검증: 잘못된 값(음수·0·NaN·무한대·정수 자리에 실수·문자열)이 조용히 통과하지 않는지 - 공개 함수 58개 인자"""
import warnings

import numpy as np
import pytest

import flydnet as fd
from flydnet.data import missing
from test_threefactor import _rec

warnings.simplefilter("ignore")
c = _rec(feedback_edges=True)
c = fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)
X = np.random.default_rng(0).uniform(50, 200, (4, 6)).astype(np.float32)
P = lambda **k: fd.Projection(6, 3, device="cpu", **k)
L = lambda **k: fd.Connectome(c, "IN", "O", **dict(dict(t_ms=20, device="cpu"), **k))
syn = lambda: fd.Projection(3, 2, device="cpu").synapses()
NEEDS_DATA = ("Glomeruli", "KCExpansion", "door_task")

BAD = {"pos": [0, -1, float("nan"), float("inf")], "nonneg": [-1, float("nan"), float("inf")],
       "int_pos": [0, -1, 2.5, "3"], "int_nonneg": [-1, 2.5], "unit": [-0.1, 1.5, float("nan")],
       "unit_open": [-0.1, 1.0, 1.5, float("nan")], "finite": [float("nan"), float("inf")]}
TABLE = [
    ("Projection", lambda v: fd.Projection(v, 3, device="cpu"), "int_pos"),
    ("Projection n_out", lambda v: fd.Projection(6, v, device="cpu"), "int_pos"),
    ("Connectome t_ms", lambda v: L(t_ms=v), "pos"),
    ("Connectome dt", lambda v: L(dt=v), "pos"),
    ("Connectome slope", lambda v: L(slope=v), "pos"),
    ("Connectome count_from_ms", lambda v: L(count_from_ms=v), "nonneg"),
    ("Connectome checkpoint_every", lambda v: L(checkpoint_every=v), "int_pos"),
    ("Connectome noise", lambda v: L(noise=v), "nonneg"),
    ("Connectome surrogate_damp", lambda v: L(surrogate_damp=v), "pos"),
    ("Connectome truncate", lambda v: L(truncate=v), "int_pos"),
    ("Connectome params w_syn", lambda v: L(params={"w_syn": v}), "finite"),
    ("Connectome params t_mbr", lambda v: L(params={"t_mbr": v}), "pos"),
    ("Connectome params tau", lambda v: L(params={"tau": v}), "pos"),
    ("Connectome params t_rfc", lambda v: L(params={"t_rfc": v}), "nonneg"),
    ("Connectome params t_dly", lambda v: L(params={"t_dly": v}), "nonneg"),
    ("Connectome params f_poi", lambda v: L(params={"f_poi": v}), "nonneg"),
    ("Connectome forward batch", lambda v: L(inputs=None)(None, batch=v), "int_pos"),
    ("Plasticity rate", lambda v: fd.Plasticity(syn(), rate=v), "nonneg"),
    ("Plasticity momentum", lambda v: fd.Plasticity(syn(), momentum=v), "unit_open"),
    ("Plasticity decay", lambda v: fd.Plasticity(syn(), decay=v), "nonneg"),
    ("Adaptive betas[0]", lambda v: fd.Adaptive(syn(), betas=(v, 0.999)), "unit_open"),
    ("Adaptive betas[1]", lambda v: fd.Adaptive(syn(), betas=(0.9, v)), "unit_open"),
    ("Adaptive eps", lambda v: fd.Adaptive(syn(), eps=v), "pos"),
    ("Adaptive decay", lambda v: fd.Adaptive(syn(), decay=v), "nonneg"),
    ("RateEncoder max_rate", lambda v: fd.RateEncoder(6, 4, max_rate=v, device="cpu"), "nonneg"),
    ("RateEncoder k", lambda v: fd.RateEncoder(6, 4, k=v, device="cpu"), "int_pos"),
    ("Glomeruli max_rate", lambda v: fd.Glomeruli(fd.flywire(), max_rate=v, device="cpu"), "nonneg"),
    ("KCExpansion k_frac", lambda v: fd.KCExpansion(fd.flywire(), k_frac=v, device="cpu"), "unit"),
    ("DopamineReadout lr", lambda v: fd.DopamineReadout(6, 2, lr=v, device="cpu"), "nonneg"),
    ("DopamineReadout n_classes", lambda v: fd.DopamineReadout(6, v, device="cpu"), "int_pos"),
    ("AssocReadout per_class", lambda v: fd.AssocReadout(6, 2, per_class=v, device="cpu"), "int_pos"),
    ("MBON per_class", lambda v: fd.MBON(6, 2, per_class=v, device="cpu"), "int_pos"),
    ("AxonHillock slope", lambda v: fd.AxonHillock(slope=v), "pos"),
    ("Homeostasis eps", lambda v: fd.Homeostasis(eps=v), "nonneg"),
    ("train rate", lambda v: fd.train(fd.Pathway(P()), X, [0, 1, 0, 1], rate=v, epochs=1, verbose=False), "nonneg"),
    ("train clip", lambda v: fd.train(fd.Pathway(P()), X, [0, 1, 0, 1], clip=v, epochs=1, verbose=False), "pos"),
    ("train_linear epochs", lambda v: fd.train_linear(X, [0, 1, 0, 1], X, [0, 1, 0, 1], epochs=v), "int_nonneg"),
    ("train_linear lr", lambda v: fd.train_linear(X, [0, 1, 0, 1], X, [0, 1, 0, 1], lr=v, epochs=1), "nonneg"),
    ("extract batch", lambda v: fd.extract(L(), None, X, batch=v), "int_pos"),
    ("explain verify", lambda v: fd.explain(lambda l, s: l(X, seed=s).sum(), L(), verify=v), "int_nonneg"),
    ("compare ceiling", lambda v: fd.compare(lambda cc, s: 0.5 + s * 0.01, c, seeds=3, ceiling=v, verbose=False), "unit"),
    ("screen hz", lambda v: fd.genetics.screen(lambda l, s: 1.0, L(inputs=None), fd.genetics.lines(c), effector="activate", hz=v, seeds=2, verbose=False), "pos"),
    ("activate level", lambda v: fd.genetics.activate(L(neuron="graded"), fd.genetics.driver(c, group="H"), level=v), "finite"),
    ("mosaic p", lambda v: fd.genetics.mosaic(L(), p=v), "unit"),
    ("STDP a_plus", lambda v: fd.STDP(L(trainable=True), a_plus=v), "nonneg"),
    ("STDP tau_plus", lambda v: fd.STDP(L(trainable=True), tau_plus=v), "pos"),
    ("gradcheck eps", lambda v: fd.gradcheck(lambda l, s: l(X, seed=s).sum(), L(trainable=True, share="pair"), eps=v), "pos"),
    ("calibrate iters", lambda v: L(outputs=("H", "O")).calibrate(X, {"H": 10}, iters=v), "int_nonneg"),
    ("calibrate step", lambda v: L(outputs=("H", "O")).calibrate(X, {"H": 10}, step=v), "pos"),
    ("graphs.erdos_renyi p", lambda v: fd.graphs.erdos_renyi(20, v), "unit"),
    ("graphs.erdos_renyi n", lambda v: fd.graphs.erdos_renyi(v, 0.1), "int_pos"),
    ("graphs inhibitory", lambda v: fd.graphs.erdos_renyi(20, 0.1, inhibitory=v), "unit"),
    ("graphs.watts_strogatz beta", lambda v: fd.graphs.watts_strogatz(20, 4, v), "unit"),
    ("graphs weight", lambda v: fd.graphs.layered([3, 4, 2], 0.5, weight=v), "finite"),
    ("Izhikevich tau_syn", lambda v: L(neuron=fd.neurons.Izhikevich(tau_syn=v))(X, seed=0), "pos"),
    ("door_task noise", lambda v: fd.door_task(n_odors=3, noise=v), "nonneg"),
    ("door_task background", lambda v: fd.door_task(n_odors=3, background=v), "nonneg"),
    ("synthetic_odors noise", lambda v: fd.synthetic_odors(3, 5, 4, 4, noise=v), "nonneg"),
]


CASES = [(name, f, v) for name, f, rule in TABLE for v in BAD[rule]]


@pytest.mark.parametrize("name,f,v", CASES, ids=[f"{n}={v!r}" for n, _, v in CASES])
def test_bad_argument_rejected(name, f, v):
    if any(k in name for k in NEEDS_DATA) and (missing("flywire") or missing("door")):
        pytest.skip("데이터 없음")
    with pytest.raises((ValueError, TypeError, KeyError)):
        f(v)


def test_lif_params_typo_and_threshold():
    with pytest.raises(KeyError, match="혹시 't_mbr'"):
        L(params={"tmbr": 10})
    with pytest.raises(ValueError, match="v_th"):
        L(params={"v_th": -60})
    L(params={"t_mbr": 15.0, "w_syn": 0.3})                                   # 정상 값은 그대로


def test_valid_numpy_scalars_accepted():
    """numpy 정수·실수도 정상 인자로"""
    fd.Projection(np.int64(6), np.int32(3), device="cpu")
    fd.Adaptive(syn(), rate=np.float32(1e-3), betas=(np.float64(0.9), 0.999))
    L(checkpoint_every=np.int64(5), noise=np.float32(0.5))


def test_mean_dtype_honored():
    s = fd.Signal(np.ones((2, 3), np.float32))
    assert s.mean(dtype=np.float64).data.dtype == np.float64 and s.mean().data.dtype == np.float32
