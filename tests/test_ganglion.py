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
    import torch as _t
    ref = fd.AssocReadout(40, 5, per_class=3, device="cpu")
    ref.step(_t.tensor(X), _t.tensor(y))
    mbo = G.MushroomBodyOutput(40, 5, per_class=3, device="cpu").learn(X, y, batch=300)
    np.testing.assert_allclose(mbo.prototypes, ref.W.numpy(), atol=1e-6)
    assert (mbo.predict(X) == ref.predict(_t.tensor(X)).numpy()).all()
    assert mbo.predict(X, classes=[1, 2]).max() <= 2


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
        try:
            fd.ConnectomeLayer
            raise SystemExit("torch 기능이 불러와짐")
        except ImportError as e:
            assert "flydnet[torch]" in str(e)
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
