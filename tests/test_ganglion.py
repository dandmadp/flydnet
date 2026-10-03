"""flydnet.ganglion (자체 엔진) 테스트

1. 수치 미분: 연산마다 역행성 신호(기울기)를 유한 차분과 비교 (float64, CPU)
2. torch를 정답지로: 같은 입력에서 출력·기울기·가소성 규칙 결과가 같은지 (torch가 있을 때)
3. CPU ↔ GPU: 같은 계산이 양쪽에서 같은지 (CuPy·GPU가 있을 때)
"""
import numpy as np
import pytest

import flydnet as fd
import flydnet.ganglion as G
from flydnet.ganglion import backend as B

needs_gpu = pytest.mark.skipif(not G.gpu_available(), reason="GPU(CuPy) 없음")
try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch 없음")
RNG = np.random.default_rng(0)


def numgrad(f, arrays, eps=1e-6):
    """f(*arrays) → 스칼라 numpy. 각 배열의 수치 기울기"""
    out = []
    for a in arrays:
        g = np.zeros_like(a)
        it = np.nditer(a, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = a[i]
            a[i] = old + eps; hi = f(*arrays)
            a[i] = old - eps; lo = f(*arrays)
            a[i] = old
            g[i] = (hi - lo) / (2 * eps)
        out.append(g)
    return out


def check(fn, *shapes, positive=False, atol=1e-6):
    """fn(*Signals) → Signal. 역행성 신호 = 수치 미분인지 (출력에 무작위 가중을 곱해 스칼라로)"""
    arrays = [RNG.uniform(0.5, 2.0, s) if positive else RNG.normal(size=s) for s in shapes]
    probe = None

    def scalar(*arrs):
        nonlocal probe
        out = fn(*[G.Signal(a.copy(), dtype=np.float64) for a in arrs]).data
        if probe is None:
            probe = RNG.normal(size=out.shape)
        return float((out * probe).sum())
    scalar(*arrays)
    sig = [G.Signal(a.copy(), plastic=True, dtype=np.float64) for a in arrays]
    out = fn(*sig)
    (out * G.Signal(probe)).sum().retrograde()
    for s, ng in zip(sig, numgrad(scalar, arrays)):
        np.testing.assert_allclose(s.retro, ng, atol=atol, rtol=1e-5)


# ─────────────── 1. 수치 미분 ───────────────
@pytest.mark.parametrize("name,fn,shapes", [
    ("add-broadcast", lambda a, b: a + b, [(3, 4), (4,)]),
    ("sub-broadcast", lambda a, b: a - b, [(3, 4), (3, 1)]),
    ("mul", lambda a, b: a * b, [(3, 4), (3, 4)]),
    ("matmul", lambda a, b: a @ b, [(3, 4), (4, 2)]),
    ("tanh", lambda a: a.tanh(), [(3, 4)]),
    ("sigmoid", lambda a: a.sigmoid(), [(3, 4)]),
    ("exp", lambda a: a.exp(), [(3, 4)]),
    ("sum-axis", lambda a: a.sum(axis=1), [(3, 4)]),
    ("mean-keepdims", lambda a: a.mean(axis=0, keepdims=True), [(3, 4)]),
    ("max-axis", lambda a: a.max(axis=1), [(3, 4)]),
    ("reshape-T", lambda a: a.reshape(4, 3).T, [(3, 4)]),
    ("slice", lambda a: a[1:, ::2], [(3, 4)]),
    ("fancy-dup", lambda a: a[np.array([0, 2, 0])], [(3, 4)]),
    ("concat", lambda a, b: G.concat([a, b], axis=1), [(2, 3), (2, 2)]),
    ("where", lambda a, b: G.where(a.data > 0, a, b), [(3, 4), (3, 4)]),
    ("log_softmax", lambda a: G.log_softmax(a), [(3, 5)]),
    ("neg-pow", lambda a: -(a ** 3), [(3, 4)]),
])
def test_gradients_match_numerical(name, fn, shapes):
    check(fn, *shapes)


@pytest.mark.parametrize("fn", [lambda a, b: a / b, lambda a: a.log(), lambda a: a ** 0.5])
def test_gradients_positive_domain(fn):
    n = fn.__code__.co_argcount
    check(fn, *[(3, 4)] * n, positive=True)


def test_surprise_and_transmit_gradients():
    y = np.array([0, 2, 1])
    check(lambda a: G.surprise(a, y), (3, 4))
    rng = np.random.default_rng(1)
    key = rng.choice(6 * 5, 12, replace=False)
    w, order = G.wiring(key // 5, key % 5, 6, 5)
    check(lambda x, v: G.transmit(x, v, w), (3, 5), (12,))
    check(lambda x, v: G.transmit(x, v, w, edge_chunk=4), (3, 5), (12,))     # 연결을 나눠 계산해도 같음


def test_transmit_matches_dense():
    rng = np.random.default_rng(2)
    key = rng.choice(7 * 5, 15, replace=False)
    post, pre = key // 5, key % 5
    w, order = G.wiring(post, pre, 7, 5)
    v = rng.normal(size=15)
    dense = np.zeros((7, 5)); dense[post[order], pre[order]] = v
    x = rng.normal(size=(3, 5))
    np.testing.assert_allclose(G.transmit(G.Signal(x, dtype=np.float64), G.Signal(v, dtype=np.float64), w).data,
                               x @ dense.T, atol=1e-12)
    with pytest.raises(ValueError):
        G.wiring([0, 0], [1, 1], 7, 5)


def test_fire_inhibit_semantics():
    v = G.Signal(np.array([-0.2, 0.0, 0.3]), plastic=True)
    s = G.fire(v, threshold=0.1)
    assert s.numpy().tolist() == [0, 0, 1]
    s.sum().retrograde()
    assert (v.retro > 0).all()                                                 # 대리 기울기
    x = G.Signal(np.array([[3.0, -1.0, 2.0, 0.5]]), plastic=True)
    h = G.inhibit(x, k=2)
    assert h.numpy().tolist() == [[3.0, 0.0, 2.0, 0.0]]
    h.sum().retrograde()
    assert x.retro.tolist() == [[1, 0, 1, 0]]
    assert (G.inhibit(G.Signal(np.ones((1, 4))), k=10).numpy() == 1).all()


# ─────────────── 그래프 동작 ───────────────
def test_retrograde_rules():
    a = G.Synapse(np.ones(3))
    (a * 2).sum().retrograde()
    (a * 3).sum().retrograde()
    assert a.retro.tolist() == [5, 5, 5]                                        # 쌓임
    with G.quiescent():
        y = a * 2
    assert not y.plastic                                                       # 휴지 상태: 기록 안 함
    with pytest.raises(RuntimeError):
        y.sum().retrograde()
    with pytest.raises(RuntimeError):
        (a * 2).retrograde()                                                   # 값이 여러 개
    b = G.Synapse(np.ones(3))
    z = (b * b).sum()                                                          # 같은 신호를 두 번 써도
    z.retrograde()
    assert b.retro.tolist() == [2, 2, 2]


def test_deep_chain_no_recursion_limit():
    a = G.Synapse(np.ones(2))
    x = a
    for _ in range(5000):
        x = x * 1.0
    x.sum().retrograde()
    assert a.retro.tolist() == [1, 1]


# ─────────────── 조직 ───────────────
def _pn_kc(n_pn=30, n_kc=200, per_kc=6, seed=0, signs=False):
    rng = np.random.default_rng(seed)
    pre = np.concatenate([rng.choice(n_pn, per_kc, replace=False) for _ in range(n_kc)])
    post = np.repeat(np.arange(n_pn, n_pn + n_kc), per_kc)
    w = rng.integers(1, 10, len(pre)).astype(np.float32)
    if signs:
        w *= np.where(np.isin(pre, np.arange(0, n_pn, 3)), -1, 1)
    return fd.Circuit(np.arange(n_pn + n_kc), {"PN": np.arange(n_pn), "KC": np.arange(n_pn, n_pn + n_kc)}, pre, post, w)


@pytest.mark.parametrize("train", ["edge", "pair", "free", None])
def test_neuropil_modes_keep_wiring_and_sign(train):
    c = _pn_kc(signs=True)
    layer = G.Neuropil(c, "PN", "KC", train=train, bias=True, device="cpu")
    mask, sign = layer.dense() != 0, np.sign(layer.dense())
    rule = G.Plasticity(layer.synapses(), rate=0.5)
    for _ in range(5):
        loss = ((layer(RNG.normal(size=(8, 30))) - 1) ** 2).mean()
        rule.clear(); loss.retrograde(); rule.step()
    after = layer.dense()
    assert ((after != 0) <= mask).all()
    if train in ("edge", "pair"):
        assert (np.sign(after) == sign).all()
    expect = {"edge": layer.n_connections, "pair": len(layer.pairs), "free": layer.n_connections, None: 0}[train]
    assert layer.n_synapses() == expect + 200


def test_tissue_state_roundtrip_and_wiring_check(tmp_path):
    c = _pn_kc()
    model = G.Pathway(G.Projection(10, 30, seed=0, device="cpu"), G.Neuropil(c, "PN", "KC", device="cpu"),
                      G.LateralInhibition(frac=0.1), G.Projection(200, 3, seed=1, device="cpu"))
    model[1].log_scale.data += RNG.normal(size=model[1].log_scale.shape).astype(np.float32)
    x = RNG.normal(size=(4, 10))
    model.save(tmp_path / "m.npz")
    new = G.Pathway(G.Projection(10, 30, seed=5, device="cpu"), G.Neuropil(c, "PN", "KC", device="cpu"),
                    G.LateralInhibition(frac=0.1), G.Projection(200, 3, seed=6, device="cpu"))
    new.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(new(x).numpy(), model(x).numpy())
    other = G.Pathway(G.Projection(10, 30, device="cpu"), G.Neuropil(_pn_kc(seed=3), "PN", "KC", device="cpu"),
                      G.LateralInhibition(frac=0.1), G.Projection(200, 3, device="cpu"))
    before = other[0].weight.numpy().copy()
    with pytest.raises(RuntimeError, match="배선이 다름"):
        other.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(other[0].weight.numpy(), before)              # 실패하면 아무것도 안 바뀜


def test_pathway_learns_toy_problem():
    c = _pn_kc(n_pn=20, n_kc=300)
    model = G.Pathway(G.Projection(10, 20, seed=0, device="cpu"), G.Activation("relu"),
                      G.Neuropil(c, "PN", "KC", device="cpu"), G.LateralInhibition(frac=0.1),
                      G.Projection(300, 3, seed=1, device="cpu"))
    X = RNG.normal(size=(300, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    rule = G.AdaptivePlasticity(model.synapses(), rate=1e-2)
    first = None
    for _ in range(150):
        loss = G.surprise(model(X), y)
        first = first or loss.item()
        rule.clear(); loss.retrograde(); rule.step()
    assert loss.item() < 0.5 * first


def test_mushroom_body_output_matches_assoc_readout():
    rng = np.random.default_rng(0)
    X = rng.random((300, 40)).astype(np.float32); y = rng.integers(0, 5, 300)
    ref = fd.AssocReadout(40, 5, per_class=3, device="cpu")                    # 자체 엔진판 AssocReadout
    ref.step(X, y)
    mbo = G.MushroomBodyOutput(40, 5, per_class=3, device="cpu").learn(X, y, batch=300)
    np.testing.assert_allclose(mbo.prototypes, ref.W, atol=1e-6)
    assert (mbo.predict(X) == ref.predict(X)).all()
    assert mbo.predict(X, classes=[1, 2]).max() <= 2


@needs_torch
def test_assoc_and_dopamine_readouts_match_torch():
    """자체 엔진판 AssocReadout·DopamineReadout = torch판 (같은 순서로 학습하면 같은 시냅스)"""
    rng = np.random.default_rng(1)
    X = rng.random((200, 30)).astype(np.float32); y = rng.integers(0, 4, 200)
    for make in (lambda m: m.AssocReadout(30, 4, per_class=3, device="cpu"),
                 *[lambda m, mode=mode: m.DopamineReadout(30, 4, mode=mode, device="cpu")
                   for mode in ("bidir", "assoc", "ltd", "ltd_err", "ltp")]):
        a, t = make(fd), make(fd.torch)
        for i in range(0, 200, 50):
            a.step(X[i:i + 50], y[i:i + 50])
            t.step(torch.tensor(X[i:i + 50]), torch.tensor(y[i:i + 50]))
        np.testing.assert_allclose(a.W, t.W.numpy(), atol=1e-5)
        assert (a.predict(X) == t.predict(torch.tensor(X)).numpy()).all()


# ─────────────── 2. torch를 정답지로 ───────────────
@needs_torch
def test_matches_torch_neuropil_and_adam():
    from flydnet.torch.anatomy import Neuropil as TorchNeuropil
    c = _pn_kc(signs=True)
    tn = TorchNeuropil(c, "PN", "KC", train="edge", device="cpu")
    gn = G.Neuropil(c, "PN", "KC", train="edge", device="cpu")
    x = RNG.normal(size=(5, 30)).astype(np.float32)
    np.testing.assert_allclose(gn(x).numpy(), tn(torch.tensor(x)).detach().numpy(), atol=1e-5)
    # 같은 손실로 Adam 3번 → 같은 매개변수
    topt = torch.optim.Adam(tn.parameters(), lr=1e-2)
    gopt = G.AdaptivePlasticity(gn.synapses(), rate=1e-2)
    target = RNG.normal(size=(5, 200)).astype(np.float32)
    for _ in range(3):
        tl = ((tn(torch.tensor(x)) - torch.tensor(target)) ** 2).mean()
        topt.zero_grad(); tl.backward(); topt.step()
        gl = ((gn(x) - target) ** 2).mean()
        gopt.clear(); gl.retrograde(); gopt.step()
        assert abs(tl.item() - gl.item()) < 1e-5
    np.testing.assert_allclose(gn.log_scale.numpy(), tn.log_scale.detach().numpy(), atol=1e-5)


@needs_torch
def test_matches_torch_surprise_and_sgd_momentum():
    logits = RNG.normal(size=(6, 4)).astype(np.float32); y = RNG.integers(0, 4, 6)
    t = torch.tensor(logits, requires_grad=True)
    torch.nn.functional.cross_entropy(t, torch.tensor(y)).backward()
    s = G.Signal(logits, plastic=True)
    l = G.surprise(s, y); l.retrograde()
    np.testing.assert_allclose(s.retro, t.grad.numpy(), atol=1e-6)
    w0 = RNG.normal(size=(3,)).astype(np.float32)
    tw = torch.nn.Parameter(torch.tensor(w0)); gw = G.Synapse(w0.copy())
    topt = torch.optim.SGD([tw], lr=0.1, momentum=0.9, weight_decay=0.01)
    gopt = G.Plasticity([gw], rate=0.1, momentum=0.9, decay=0.01)
    for _ in range(4):
        topt.zero_grad(); (tw ** 2).sum().backward(); topt.step()
        gopt.clear(); (gw ** 2).sum().retrograde(); gopt.step()
    np.testing.assert_allclose(gw.numpy(), tw.detach().numpy(), atol=1e-6)


# ─────────────── 3. CPU ↔ GPU ───────────────
@needs_gpu
def test_gpu_matches_cpu():
    c = _pn_kc(signs=True)
    x = RNG.normal(size=(8, 10)).astype(np.float32); y = RNG.integers(0, 3, 8)
    out = {}
    for dev in ("cpu", "gpu"):
        model = G.Pathway(G.Projection(10, 30, seed=0, device=dev), G.Neuropil(c, "PN", "KC", device=dev),
                          G.LateralInhibition(frac=0.2), G.Projection(200, 3, seed=1, device=dev))
        loss = G.surprise(model(x), y)
        loss.retrograde()
        out[dev] = (loss.item(), [B.numpy(s.retro) for s in model.synapses()])
    assert abs(out["cpu"][0] - out["gpu"][0]) < 1e-5
    for a, b in zip(out["cpu"][1], out["gpu"][1]):
        np.testing.assert_allclose(a, b, atol=1e-5)


@needs_gpu
def test_tissue_to_device_and_back():
    c = _pn_kc()
    m = G.Pathway(G.Projection(10, 30, seed=0, device="cpu"), G.Neuropil(c, "PN", "KC", device="cpu"))
    x = RNG.normal(size=(3, 10)).astype(np.float32)
    ref = m(x).numpy()
    m.to("gpu")
    assert m.device == "gpu" and m[1].wiring.device == "gpu"
    np.testing.assert_allclose(m(x).numpy(), ref, atol=1e-5)
    m.to("cpu")
    np.testing.assert_allclose(m(x).numpy(), ref, atol=1e-6)
    mbo = G.MushroomBodyOutput(10, 3, per_class=2, device="gpu").learn(x, np.array([0, 1, 2]))
    assert mbo.predict(x).tolist() == [0, 1, 2]


def test_flydnet_works_without_torch():
    """torch가 없는 환경 (sys.modules['torch'] = None으로 흉내): import·학습이 되고, torch 기능은 안내 오류"""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import sys; sys.modules["torch"] = None
        import numpy as np, flydnet as fd
        c = fd.Circuit(np.arange(6), {"A": np.arange(3), "B": np.arange(3, 6)},
                       np.array([0, 1, 2, 0]), np.array([3, 4, 5, 5]), np.ones(4, np.float32))
        m = fd.Pathway(fd.Neuropil(c, "A", "B", device="cpu"), fd.Projection(3, 2, device="cpu"))
        loss = fd.surprise(m(np.ones((4, 3), np.float32)), [0, 1, 0, 1]); loss.retrograde()
        assert all(s.retro is not None for s in m.synapses())
        layer = fd.ConnectomeLayer(c, "A", "B", t_ms=10, dt=0.5, gains={"A>B": 30.0}, trainable=True, device="cpu")
        out = layer(np.full((2, 3), 150.0, np.float32), seed=0)
        (out * out).sum().retrograde()                       # 시간 시뮬레이션도 torch 없이 학습
        assert layer.log_scale.retro is not None
        r = fd.AssocReadout(3, 2, per_class=2, device="cpu").fit(np.random.rand(10, 3), np.arange(10) % 2)
        assert r.predict(np.random.rand(4, 3)).shape == (4,)                   # 0.1 기능도 torch 없이
        fd.KCExpansion(c, n_in=4, pre="A", post="B", device="cpu")(np.random.rand(2, 4))
        fd.train_linear(np.random.rand(20, 3), np.arange(20) % 2, np.random.rand(5, 3), np.arange(5) % 2,
                        epochs=2, device="cpu")
        try:
            fd.torch
            raise SystemExit("torch 연동이 불러와짐")
        except ImportError as e:
            assert "flydnet[torch]" in str(e)
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


# ─────────────── ConnectomeLayer (시간 시뮬레이션) ───────────────
def _tiny(n_in=5, n_out=12, n_edges=120, seed=0):
    rng = np.random.default_rng(seed)
    N = n_in + n_out
    key = rng.choice(N * N, n_edges, replace=False)
    pre, post = key // N, key % N
    keep = pre != post
    w = rng.integers(1, 6, keep.sum()) * rng.choice([-1, 1], keep.sum())
    return fd.Circuit(np.arange(N), {"IN": np.arange(n_in), "OUT": np.arange(n_in, N)},
                      pre[keep], post[keep], w.astype(np.float32))


@needs_torch
@pytest.mark.parametrize("neuron,kw", [
    ("lif", dict(gains={"IN>OUT": 30.0}, input_mode="regular", dt=0.5)),
    ("lif", dict(gains={"IN>OUT": 30.0}, input_mode="regular", dt=0.5, bias={"OUT": 3.0}, train_neurons=True,
                 v_init="random", count_from_ms=5, share="pair")),
    ("graded", dict(params={"w_syn": 0.3}, bias={"OUT": 0.2}, train_neurons=True, dt=1.0)),
])
def test_connectome_layer_matches_torch(neuron, kw):
    from flydnet.torch.layers import ConnectomeLayer as TL
    c = _tiny()
    tl = TL(c, "IN", "OUT", t_ms=30, neuron=neuron, trainable=True, device="cpu", **kw)
    gl = G.ConnectomeLayer(c, "IN", "OUT", t_ms=30, neuron=neuron, trainable=True, device="cpu", **kw)
    gl.phase0 = tl.phase0.numpy().ravel().copy()
    gl.v_frac = tl.v_frac.numpy().ravel().copy()
    x = (np.random.default_rng(3).random((4, 5)) * (200 if neuron == "lif" else 1)).astype(np.float32)
    t_out = tl(torch.tensor(x))
    g_out = gl(x)
    np.testing.assert_allclose(g_out.numpy(), t_out.detach().numpy(), atol=1e-4)
    w = np.random.default_rng(4).normal(size=g_out.shape).astype(np.float32)
    (t_out * torch.tensor(w)).sum().backward()
    (g_out * w).sum().retrograde()
    np.testing.assert_allclose(gl.log_scale.retro, tl.log_scale.grad.numpy(), rtol=1e-3, atol=1e-4)
    if "train_neurons" in kw:
        np.testing.assert_allclose(gl.bias.retro, tl.bias.grad.numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(gl.log_t_mbr.retro, tl.log_t_mbr.grad.numpy(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("neuron,mode", [("lif", "poisson"), ("lif", "regular"), ("graded", "regular")])
def test_connectome_layer_checkpoint_same_output_and_retro(neuron, mode):
    c = _tiny()
    kw = dict(t_ms=20, dt=0.5, neuron=neuron, input_mode=mode, trainable=True, device="cpu",
              gains={"IN>OUT": 30.0}, bias={"OUT": 0.2} if neuron == "graded" else None)
    x = np.random.default_rng(5).random((3, 5)).astype(np.float32) * (200 if neuron == "lif" else 1)
    res = []
    for ce in (None, 7):
        layer = G.ConnectomeLayer(c, "IN", "OUT", checkpoint_every=ce, **kw)
        xs = G.Signal(x, plastic=True)
        out = layer(xs, seed=11)
        (out * out).sum().retrograde()
        res.append((out.numpy(), layer.log_scale.retro.copy(), xs.retro.copy()))
    for a, b in zip(*res):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_connectome_layer_poisson_seed_and_options():
    c = _tiny()
    layer = G.ConnectomeLayer(c, "IN", "OUT", t_ms=30, dt=0.5, gains={"IN>OUT": 30.0}, device="cpu")
    x = np.full((2, 5), 150.0, np.float32)
    with G.quiescent():
        a, b, d = layer(x, seed=1).numpy(), layer(x, seed=1).numpy(), layer(x, seed=2).numpy()
        assert np.array_equal(a, b) and not np.array_equal(a, d)
        assert np.array_equal(layer(x[:, None].repeat(4, 1), seed=1).numpy(), a)   # 시간 고정 입력 = (B, T, n)
        out, tr = layer(x, seed=1, record=[5, 6])
        assert tr.shape == (2, 60, 2) and np.array_equal(out.numpy(), a)
        assert layer(np.zeros((2, 5), np.float32)).numpy().sum() == 0
    with pytest.raises(ValueError):
        G.ConnectomeLayer(c, "IN", "OUT", t_ms=10, count_from_ms=10, device="cpu")
    with pytest.raises(ValueError):
        G.ConnectomeLayer(c, "IN", "OUT", trainable=["IN>NOPE"], device="cpu")


def test_connectome_layer_save_load(tmp_path):
    c = _tiny()
    layer = G.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True, share="pair",
                              bias={"OUT": 2.0}, train_neurons=True, gains={"IN>OUT": 30.0}, device="cpu")
    layer.log_scale.data += 0.3; layer.bias.data += 1.0
    layer.save(tmp_path / "l.npz")
    back = G.ConnectomeLayer.load(tmp_path / "l.npz", device="cpu")
    x = np.full((2, 5), 150.0, np.float32)
    with G.quiescent():
        assert np.array_equal(layer(x).numpy(), back(x).numpy())
    assert set(back.scale_of()) == set(layer.scale_of())


@needs_gpu
def test_connectome_layer_gpu_matches_cpu():
    c = _tiny()
    x = np.random.default_rng(6).random((3, 5)).astype(np.float32) * 200
    res = {}
    for dev in ("cpu", "gpu"):
        layer = G.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, trainable=True, gains={"IN>OUT": 30.0},
                                  device=dev, checkpoint_every=10)
        out = layer(x, seed=3)                                                     # 포아송: 해시 난수 → 같음
        (out * out).sum().retrograde()
        res[dev] = (out.numpy(), B.numpy(layer.log_scale.retro))
    np.testing.assert_allclose(res["cpu"][0], res["gpu"][0], atol=1e-4)
    np.testing.assert_allclose(res["cpu"][1], res["gpu"][1], rtol=1e-3, atol=1e-4)


@needs_torch
def test_pair_order_matches_torch_with_unsorted_group_names():
    """그룹 이름이 알파벳순이 아닐 때도 연결 종류 순서가 torch판과 같아야 함 (실제 데이터에서 잡힌 버그)"""
    from flydnet.torch.layers import ConnectomeLayer as TL
    c = _tiny()
    c = fd.Circuit(c.root_ids, {"Zin": c.groups["IN"], "Aout": c.groups["OUT"]}, c.pre, c.post, c.weight)
    kw = dict(t_ms=20, dt=1.0, neuron="graded", params={"w_syn": 0.3}, bias=0.2, trainable=True, share="pair")
    tl = TL(c, "Zin", "Aout", device="cpu", **kw)
    gl = G.ConnectomeLayer(c, "Zin", "Aout", device="cpu", **kw)
    assert gl.train_pairs == list(tl.train_pairs)
    x = np.random.default_rng(0).random((2, 5)).astype(np.float32)
    tl(torch.tensor(x)).sum().backward()
    gl(x).sum().retrograde()
    np.testing.assert_allclose(gl.log_scale.retro, tl.log_scale.grad.numpy(), rtol=1e-3, atol=1e-5)


@needs_gpu
@pytest.mark.parametrize("nb", [1, 3, 8, 16, 32, 40])
def test_gpu_kernels_match_reference(nb):
    """직접 쓴 CUDA 커널 (행 우선 SpMM, 연결별 내적) = 기준 계산. 배치 32 초과(레인 반복)·빈 행 포함"""
    import cupy as cp
    from flydnet.ganglion import kernels as K
    rng = np.random.default_rng(nb)
    n = 300
    key = rng.choice(n * n, 2000, replace=False)
    post, pre = key // n, key % n
    post[post == 7] = 8                                                        # 행 7은 비어 있게
    key = np.unique(post * n + pre); post, pre = key // n, key % n
    w, order = G.wiring(post, pre, n, n, device="gpu")
    vals = cp.asarray(rng.normal(size=len(post)).astype(np.float32))
    M, MT = K.matrices(w, vals)
    x = cp.asarray(rng.normal(size=(n, nb)).astype(np.float32))
    g = cp.asarray(rng.normal(size=(n, nb)).astype(np.float32))
    np.testing.assert_allclose(cp.asnumpy(K.spmm(M, x)), cp.asnumpy(M @ x), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cp.asnumpy(K.spmm(MT, g)), cp.asnumpy(MT @ g), rtol=1e-4, atol=1e-4)
    ref = cp.asnumpy((g[w.post] * x[w.pre]).sum(axis=1))
    np.testing.assert_allclose(cp.asnumpy(K.edge_dot(g, x, w.indptr, w.post, w.pre)), ref, rtol=1e-4, atol=1e-4)
