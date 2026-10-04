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


# ─────────────── 2차: 1차 표에 없던 공개 함수 ───────────────
G = fd.genetics
BAD2 = {"pos": [0, -1, float("nan"), float("inf")], "int_pos": [0, -1, 2.5, "3"], "group": ["없는그룹", 3]}
mb = None


def _mb():
    global mb
    if mb is None:
        mb = fd.flywire()
    return mb


TABLE2 = [
    ("Neuropil train", lambda v: fd.Neuropil(c, "IN", "H", train=v, device="cpu"), ["foo", 3]),
    ("Neuropil init", lambda v: fd.Neuropil(c, "IN", "H", init=v, device="cpu"), ["foo", 3, None]),
    ("Neuropil pre 그룹", lambda v: fd.Neuropil(c, v, "H", device="cpu"), "group"),
    ("Projection seed", lambda v: fd.Projection(3, 2, seed=v, device="cpu"), [1.5, "a"]),
    ("Circuit.shuffled seed", lambda v: c.shuffled(seed=v), [1.5, "a"]),
    ("Circuit.shuffled pairs 없는 종류", lambda v: c.shuffled(pairs=[v]), ["Q>Z"]),
    ("Circuit.shuffled local 반경", lambda v: c.shuffled(local=(np.random.rand(c.N, 2), v)), "pos"),
    ("Circuit.randomized seed", lambda v: c.randomized(seed=v), [1.5, "a"]),
    ("Circuit.subset 없는 그룹", lambda v: c.subset([v]), ["없는그룹"]),
    ("Circuit.with_sign sign", lambda v: c.with_sign(["IN"], v), [0, 2, -3, 0.5]),
    ("Circuit.with_sign 없는 그룹", lambda v: c.with_sign([v], 1), ["없는그룹"]),
    ("Circuit.regroup 범위 밖", lambda v: c.regroup({"x": [v]}), [999, -1]),
    ("from_edges weight 길이", lambda v: fd.Circuit.from_edges([0, 1], [1, 2], [1.0] * v), [1, 3]),
    ("driver 없는 그룹", lambda v: G.driver(c, group=v), ["없는그룹"]),
    ("lines min_size", lambda v: G.lines(c, min_size=v), "int_pos"),
    ("activate hz", lambda v: G.activate(L(inputs=None), G.driver(c, group="H"), hz=v), "pos"),
    ("mosaic by", lambda v: G.mosaic(L(), by=v), ["없는열", 3]),
    ("screen effector", lambda v: G.screen(lambda l, s: 1.0, L(), G.lines(c), effector=v, seeds=2, verbose=False), ["foo"]),
    ("screen seeds", lambda v: G.screen(lambda l, s: 1.0, L(), G.lines(c), seeds=v, verbose=False), [0, -1, 1.5]),
    ("ThreeFactor feedback", lambda v: fd.ThreeFactor(L(trainable=True), feedback=v), ["foo", 3]),
    ("ThreeFactor seed", lambda v: fd.ThreeFactor(L(trainable=True), seed=v), [1.5, "a"]),
    ("tune candidates 범위 밖", lambda v: fd.tune(lambda l, s: l(X, seed=s).sum(), L(trainable=True, share="pair"),
                                               candidates=(v,), verbose=False), [0, -1, 2.0, "x"]),
    ("gradcheck seeds", lambda v: fd.gradcheck(lambda l, s: l(X, seed=s).sum(), L(trainable=True, share="pair"), seeds=v), [0, -1]),
    ("compare seeds", lambda v: fd.compare(lambda cc, s: 0.5 + 0.01 * s, c, seeds=v, verbose=False), [1.5, "a", -3]),
    ("compare chance", lambda v: fd.compare(lambda cc, s: 0.5 + 0.01 * s, c, seeds=3, chance=v, verbose=False), [-0.1, 1.5, float("nan")]),
    ("compare 대조군 이름", lambda v: fd.compare(lambda cc, s: 0.5, c, controls=[v], seeds=3, verbose=False), ["shuffle", 3]),
    ("Local radius", lambda v: fd.controls.Local(np.random.rand(c.N, 2), v), "pos"),
    ("Local xy 모양", lambda v: fd.compare(lambda cc, s: 0.5, c, controls=[fd.controls.Local(np.random.rand(v, 2), 1.0)], seeds=3, verbose=False), [3]),
    ("door_odors min_measured", lambda v: fd.door_odors(fd.Glomeruli(_mb(), device="cpu").glomeruli, min_measured=v), [-1, 2.5]),
    ("biconditional_mixtures n", lambda v: fd.biconditional_mixtures(np.random.rand(10, 5), [(0, 1, 2, 3)], v), [0, -1, 2.5]),
    ("drifting_grating frames", lambda v: fd.drifting_grating(np.random.rand(10, 2), [0], t_ms=50, frames=v), "int_pos"),
    ("drifting_grating t_ms", lambda v: fd.drifting_grating(np.random.rand(10, 2), [0], t_ms=v, frames=5), "pos"),
    ("drifting_grating wavelength", lambda v: fd.drifting_grating(np.random.rand(10, 2), [0], t_ms=50, frames=5, wavelength=v), "pos"),
    ("column_map smooth", lambda v: fd.column_map(fd.visual_circuit(), smooth=v), [-1, 2.5]),
    ("Glomeruli 없는 그룹", lambda v: fd.Glomeruli(_mb(), group=v, device="cpu"), ["없는그룹"]),
    ("KCExpansion 없는 그룹", lambda v: fd.KCExpansion(_mb(), pre=v, device="cpu"), ["없는그룹"]),
    ("RateEncoder seed", lambda v: fd.RateEncoder(6, 4, seed=v, device="cpu"), [1.5, "a"]),
    ("evaluate batch", lambda v: fd.evaluate(fd.Pathway(fd.Projection(6, 2, device="cpu")), X, [0, 1, 0, 1], batch=v), "int_pos"),
    ("train schedule", lambda v: fd.train(fd.Pathway(fd.Projection(6, 2, device="cpu")), X, [0, 1, 0, 1], schedule=v, epochs=1, verbose=False), ["linear", 3]),
    ("train val 모양", lambda v: fd.train(fd.Pathway(fd.Projection(6, 2, device="cpu")), X, [0, 1, 0, 1], val=v, epochs=1, verbose=False), [(X,), (X, [0, 1])]),
    ("Connectome forward seed 음수", lambda v: L()(X, seed=v), [-1]),
    ("Connectome record 실수", lambda v: L()(X, record=v), [[0.5], ["a"]]),
    ("Connectome input_mode", lambda v: L(input_mode=v), ["foo", 3]),
    ("Connectome trainable 문자열 하나", lambda v: L(trainable=v), ["IN>H"]),
    ("Connectome timing", lambda v: L(timing=v), ["foo", 3]),
    ("Connectome v_init", lambda v: L(v_init=v), ["foo", 3]),
    ("Connectome neuron", lambda v: L(neuron=v), ["foo", 3]),
    ("calibrate 입력 모양", lambda v: L(outputs=("H", "O")).calibrate(np.ones((2, v), np.float32), {"H": 10}), [5]),
    ("Homeostasis 0차원", lambda v: fd.Homeostasis()(np.float32(v)), [3.0]),
]

ALLOWED = {("compare chance", -0.1), ("compare chance", 1.5)}                  # 점수가 정확도가 아닐 수 있어 범위는 열어 둠
CASES2 = [(name, f, v) for name, f, rule in TABLE2 for v in (BAD2[rule] if isinstance(rule, str) else rule)
          if not (name == "compare chance" and isinstance(v, float) and (name, v) in ALLOWED)]
DATA2 = ("door_odors", "Glomeruli", "KCExpansion", "column_map")


@pytest.mark.parametrize("name,f,v", CASES2, ids=[f"{n}={v!r}" for n, _, v in CASES2])
def test_bad_argument_rejected_2(name, f, v):
    if any(k in name for k in DATA2) and (missing("flywire") or missing("door")):
        pytest.skip("데이터 없음")
    with pytest.raises((ValueError, TypeError, KeyError, IndexError)):
        f(v)


@pytest.mark.parametrize("make,match", [
    (lambda: fd.Signal(np.ones((3, 4), np.float32)).clip(5, 2), "lo"),
    (lambda: fd.Signal(np.ones((3, 4), np.float32)).var(ddof=12), "ddof"),
    (lambda: fd.ganglion.fire(fd.Signal(np.ones(3, np.float32)), 0.0, -1.0), "slope"),
    (lambda: fd.ganglion.inhibit(np.ones((2, 5)), k=0), "k"),
    (lambda: fd.surprise(fd.Signal(np.zeros((3, 2), np.float32)), [0, 1]), "시료 수"),
    (lambda: fd.surprise(fd.Signal(np.zeros((0, 3), np.float32)), np.zeros(0, int)), "빈 배치"),
    (lambda: fd.surprise(fd.Signal(np.zeros(3, np.float32)), [0]), "2차원"),
    (lambda: fd.Activation("foo"), "relu"),
    (lambda: fd.Pathway(3), "부를 수 있는"),
])
def test_op_argument_errors(make, match):
    with pytest.raises((ValueError, TypeError), match=match):
        make()
