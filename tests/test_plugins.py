"""플러그인: 사용자 정의 뉴런 모델 (fd.neurons)과 관찰자 규칙 (fd.STDP)"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from test_genetics import _chain
from test_threefactor import _rec


def _strong():
    c = _rec(feedback_edges=True)
    return fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)


X = np.random.default_rng(1).uniform(50, 200, (4, 6)).astype(np.float32)


def _pair(c, **kw):
    kw = dict(dict(t_ms=80, trainable=True, device="cpu"), **kw)
    return fd.ConnectomeLayer(c, "IN", "O", **kw), fd.ConnectomeLayer(c, "IN", "O", neuron=fd.neurons.LIF(), **kw)


# ─────────────── 플러그인 LIF = 내장 LIF (플러그인 틀의 검증) ───────────────
@pytest.mark.parametrize("mode", ["poisson", "regular"])
def test_plugin_lif_matches_builtin(mode):
    c = _strong()
    a, b = _pair(c, input_mode=mode)
    with fd.quiescent():
        _, ta = a(X, seed=2, record=list(range(c.N)))
        _, tb = b(X, seed=2, record=list(range(c.N)))
    np.testing.assert_array_equal(ta, tb)                                  # 스파이크가 스텝까지 같음
    assert ta.sum() > 100
    grads = []
    for L in (a, b):
        xs = fd.Signal(X, plastic=True)
        (L(xs, seed=2) * 0.01).sum().retrograde()
        grads.append((L.log_scale.retro.copy(), xs.retro.copy()))
    for g1, g2 in zip(*grads):                                             # 손 유도 역전파 = 자동 미분
        np.testing.assert_allclose(g1, g2, rtol=1e-3, atol=1e-4 * np.abs(g1).max())


def test_plugin_lif_with_genetics_and_checkpoint():
    c = _strong()
    a, b = _pair(c)
    b_ck = fd.ConnectomeLayer(c, "IN", "O", t_ms=80, trainable=True, device="cpu", neuron=fd.neurons.LIF(),
                              checkpoint_every=7)
    h = fd.genetics.driver(c, group="H")
    outs = []
    for L in (a, b, b_ck):
        with fd.genetics.activate(L, fd.genetics.Line(c, h.idx[:3], "h3"), hz=150), \
                fd.genetics.block(L, fd.genetics.Line(c, h.idx[5:8], "b")):
            xs = fd.Signal(X, plastic=True)
            o = L(xs, seed=5)
            (o * 0.01).sum().retrograde()
        outs.append((o.numpy(), L.log_scale.retro.copy()))
    np.testing.assert_array_equal(outs[0][0], outs[1][0])
    np.testing.assert_array_equal(outs[1][0], outs[2][0])
    np.testing.assert_allclose(outs[1][1], outs[2][1], rtol=1e-4, atol=1e-6)


# ─────────────── Izhikevich ───────────────
def test_izhikevich_types_differ_and_train():
    c = _strong()
    rs = fd.ConnectomeLayer(c, "IN", "O", t_ms=150, device="cpu", neuron=fd.neurons.Izhikevich(), trainable=True)
    fs = fd.ConnectomeLayer(c, "IN", "O", t_ms=150, device="cpu", neuron=fd.neurons.Izhikevich(a=0.1, d=2.0))
    with fd.quiescent():
        r_rs, r_fs = rs(X, seed=0, return_all=True).numpy(), fs(X, seed=0, return_all=True).numpy()
    hid = c.groups["H"]
    assert r_rs[:, hid].mean() > 1 and r_fs[:, hid].mean() > r_rs[:, hid].mean()   # fast spiking이 더 빠름
    o = rs(X, seed=0)
    (o * 0.01).sum().retrograde()
    assert np.isfinite(rs.log_scale.retro).all() and np.abs(rs.log_scale.retro).sum() > 0


def test_custom_model_with_explain_mosaic_bridge():
    c = _strong()
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=80, device="cpu", neuron=fd.neurons.Izhikevich(), trainable=True)
    rep = fd.explain(lambda L_, s: L_(X, seed=s).sum(), L, verify=1)
    assert set(rep.groups.name) == {"IN", "H", "O"}
    with fd.genetics.mosaic(L, p=0.5):
        a = L(X, seed=1).numpy()
    with fd.quiescent():
        b = L(X, seed=1).numpy()
    assert not np.array_equal(a, b)
    torch = pytest.importorskip("torch")
    m = fd.torch.bridge(L, seed=0)
    out = m(torch.tensor(X))
    out.sum().backward()
    assert m.log_scale.grad is not None


# ─────────────── 저장·등록 ───────────────
def test_custom_model_save_load_and_registry(tmp_path):
    c = _strong()
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=60, device="cpu", neuron=fd.neurons.Izhikevich(a=0.1, d=2.0))
    L.save(tmp_path / "izh")
    M = fd.ConnectomeLayer.load(tmp_path / "izh.npz", device="cpu")
    assert isinstance(M.neuron, fd.neurons.Izhikevich) and M.neuron.params["a"] == 0.1
    with fd.quiescent():
        np.testing.assert_array_equal(L(X, seed=3).numpy(), M(X, seed=3).numpy())

    class Unregistered(fd.neurons.LIF):
        pass
    U = fd.ConnectomeLayer(c, "IN", "O", t_ms=20, device="cpu", neuron=Unregistered())
    with pytest.raises(ValueError, match="등록"):
        U.save(tmp_path / "u")

    class BadInit(fd.neurons.NeuronModel):
        state = ("v",)

        def init(self, ctx):
            return {"w": ctx.zeros()}
    with pytest.raises(ValueError, match="state"):
        fd.ConnectomeLayer(c, "IN", "O", t_ms=20, device="cpu", neuron=BadInit())(X, seed=0)
    with pytest.raises(ValueError, match="neuron은"):
        fd.ConnectomeLayer(c, "IN", "O", neuron="izh")
    with pytest.raises(ValueError, match="bias"):
        fd.ConnectomeLayer(c, "IN", "O", neuron=fd.neurons.LIF(), train_neurons=True)
    with pytest.raises(ValueError, match="스파이킹"):
        fd.ThreeFactor(fd.ConnectomeLayer(c, "IN", "O", trainable=True, neuron=fd.neurons.LIF()))


def test_custom_model_gpu_matches_cpu():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    c = _strong()
    r = []
    for dev in ("cpu", "gpu"):
        L = fd.ConnectomeLayer(c, "IN", "O", t_ms=60, device=dev, neuron=fd.neurons.Izhikevich())
        with fd.quiescent():
            r.append(L(X, seed=0).numpy())
    np.testing.assert_allclose(r[0], r[1], rtol=1e-4, atol=1e-3)


# ─────────────── STDP ───────────────
def test_stdp_causal_potentiation():
    c = fd.Circuit.from_edges([0, 1], [1, 0], [40.0, 1.0], groups={"A": [0], "B": [1]})
    L = fd.ConnectomeLayer(c, "A", "B", t_ms=300, trainable=True, device="cpu")
    stdp = fd.STDP(L)
    stdp(np.full((4, 1), 60.0, np.float32), seed=0)
    d = stdp.assign()
    pre = B.numpy(L.wiring.pre)
    assert d[pre == 0][0] > 0 and d[pre == 1][0] < 0                       # A→B 강화, B→A 약화
    np.testing.assert_allclose(L.log_scale.retro, -d)
    rule = fd.Plasticity(L.synapses(), rate=1.0)
    rule.step()
    assert L.log_scale.numpy()[pre == 0][0] > 0                            # 규칙이 Δ 방향으로
    assert L._observer is None
    with pytest.raises(RuntimeError, match="먼저"):
        stdp.assign()


def test_stdp_works_with_custom_neurons():
    c = _strong()
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=80, device="cpu", neuron=fd.neurons.Izhikevich(), trainable=True)
    stdp = fd.STDP(L)
    stdp(X, seed=0)
    assert np.abs(stdp.assign()).sum() > 0
