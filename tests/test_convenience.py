"""편의 기능과 오류 메시지: 1차원 입력, 알기 쉬운 오류, fd.train / evaluate / door_task, Homeostasis, Pathway seed"""
import numpy as np
import pytest

import flydnet as fd
from test_threefactor import _rec


def _c():
    c = _rec(feedback_edges=True)
    return fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)


X = np.random.default_rng(0).uniform(50, 200, (4, 6)).astype(np.float32)


def _L(**kw):
    return fd.ConnectomeLayer(_c(), "IN", "O", **dict(dict(t_ms=30, device="cpu"), **kw))


# ─────────────── 입력 ───────────────
def test_single_sample_input():
    L = _L()
    with fd.quiescent():
        one = L(X[0], seed=3)
        many = L(X[:1], seed=3)
        out, trace = L(X[0], seed=3, record=[0, 1])
    assert one.shape == (5,) and trace.shape[0] == 300 and out.shape == (5,)
    np.testing.assert_array_equal(one.numpy(), many.numpy()[0])


@pytest.mark.parametrize("make,match", [
    (lambda: _L()(-X), "0 이상"),
    (lambda: _L()(X, seed=1.5), "seed는 정수"),
    (lambda: _L()(X, record=[999]), "record"),
    (lambda: _L(t_ms=0), "t_ms는 양수"),
    (lambda: _L(share="all"), "share는"),
    (lambda: _L(input_mode="burst"), "input_mode는"),
    (lambda: fd.ConnectomeLayer(_c(), "IN", "Q", device="cpu"), "회로에 없는 그룹"),
    (lambda: fd.AdaptivePlasticity(_L(trainable=True).synapses(), rate=-1), "rate"),
    (lambda: fd.Projection(4, 2, device="cpu")(fd.Signal(np.ones((3, 5), np.float32))), "Projection"),
    (lambda: fd.train_linear(np.ones((5, 3)), np.zeros(4), np.ones((2, 3)), np.zeros(2), epochs=1), "개수"),
    (lambda: fd.compare(lambda c, s: 0.5, _c(), controls=["shuffled"], seeds=1, verbose=False), "2개 이상"),
])
def test_clear_errors(make, match):
    with pytest.raises((ValueError, TypeError, KeyError, IndexError), match=match):
        make()


def test_overlapping_io_warns():
    with pytest.warns(UserWarning, match="겹침"):
        fd.ConnectomeLayer(_c(), "IN", ["IN", "O"], device="cpu")


def test_negative_inputs_allowed_for_graded():
    L = fd.ConnectomeLayer(_c(), "IN", "O", t_ms=20, device="cpu", neuron="graded")
    L(-np.ones((2, 6), np.float32))                                        # 연속값 뉴런은 활동값이라 음수 가능


# ─────────────── Homeostasis·Pathway seed ───────────────
def test_homeostasis():
    x = fd.Signal(np.random.default_rng(0).uniform(0, 50, (3, 7)).astype(np.float64), plastic=True)
    y = fd.Homeostasis()(x)
    np.testing.assert_allclose(y.numpy().mean(1), 0, atol=1e-9)
    np.testing.assert_allclose(y.numpy().std(1), 1, rtol=1e-4)
    w = np.random.default_rng(1).standard_normal((3, 7))
    (y * fd.Signal(w)).sum().retrograde()
    f = lambda a: float((fd.Homeostasis()(fd.Signal(a)) * fd.Signal(w)).sum().data)
    g, x0, h = np.zeros((3, 7)), x.numpy(), 1e-5
    for i in np.ndindex(x0.shape):
        a, b = x0.copy(), x0.copy(); a[i] += h; b[i] -= h
        g[i] = (f(a) - f(b)) / (2 * h)
    np.testing.assert_allclose(x.retro, g, atol=1e-7)
    assert np.isfinite(fd.Homeostasis()(np.ones((2, 4))).numpy()).all()   # 모두 같은 값이어도


def test_pathway_passes_seed():
    m = fd.Pathway(fd.Activation("relu"), _L(t_ms=80))
    x = X
    with fd.quiescent():
        a, b, c = m(x, seed=1).numpy(), m(x, seed=1).numpy(), m(x, seed=2).numpy()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    m2 = fd.Pathway(fd.Projection(3, 2, device="cpu"))
    m2(np.ones((2, 3), np.float32), seed=5)                                 # seed를 받지 않는 것만 있어도 됨


# ─────────────── fd.train / evaluate ───────────────
def test_train_and_evaluate():
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((120, 5)).astype(np.float32)
    ys = (Xs[:, 0] + Xs[:, 1] > 0).astype(int)
    m = fd.Pathway(fd.Projection(5, 8, device="cpu", seed=0), fd.Activation("tanh"), fd.Projection(8, 2, device="cpu"))
    h = fd.train(m, Xs[:80], ys[:80], val=(Xs[80:], ys[80:]), epochs=15, rate=0.05, verbose=False)
    assert h["loss"][-1] < h["loss"][0] and h["val_acc"][-1] > 0.8 and len(h["val_acc"]) == 15
    assert fd.evaluate(m, Xs[80:], ys[80:]) == h["val_acc"][-1]
    with pytest.raises(ValueError, match="개수"):
        fd.train(m, Xs, ys[:5], epochs=1, verbose=False)
    with pytest.raises(TypeError, match="synapses"):
        fd.train(lambda x: x, Xs, ys, epochs=1, verbose=False)


def test_train_with_connectome_layer():
    c = _c()
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=40, device="cpu", trainable=True)
    Xs = np.random.default_rng(1).uniform(20, 200, (24, 6)).astype(np.float32)
    ys = (Xs[:, 0] > Xs[:, 1]).astype(int)
    m = fd.Pathway(L, fd.Homeostasis(), fd.Projection(5, 2, device="cpu"))
    h = fd.train(m, Xs, ys, epochs=3, batch=8, verbose=False)
    assert np.isfinite(h["loss"]).all() and L.log_scale.retro is not None


def test_door_task():
    from flydnet.data import missing
    if missing("door"):
        pytest.skip("DoOR 데이터 없음")
    glom = [f"G{i}" for i in range(3)]
    with pytest.raises(ValueError, match="쓸 수 있는 냄새"):                 # 사구체가 적어 측정된 냄새가 없음
        fd.door_task(n_odors=1, glomeruli=glom)
    with pytest.raises(ValueError, match="samples"):
        fd.door_task(samples=0, glomeruli=glom)
    if missing("flywire"):
        pytest.skip("FlyWire 데이터 없음")
    Xtr, ytr, Xte, yte = fd.door_task(n_odors=5, samples=4, seed=1)
    assert Xtr.shape[0] == 20 and Xtr.dtype == np.float32 and 0 <= Xtr.min() and Xtr.max() <= 1
    assert sorted(set(ytr)) == list(range(5)) and len(fd.door_task.names) == 5
    np.testing.assert_array_equal(fd.door_task(n_odors=5, samples=4, seed=1)[0], Xtr)


# ─────────────── 짧은 이름·자동 보정 ───────────────
def test_short_names():
    assert fd.Connectome is fd.ConnectomeLayer and fd.Adaptive is fd.AdaptivePlasticity
    assert fd.Glomeruli is fd.GlomerularEncoder and fd.Inhibition is fd.LateralInhibition
    assert fd.MBON is fd.MushroomBodyOutput and fd.tune is fd.tune_surrogate
    a = fd.Connectome(_c(), "IN", "O", t_ms=40, device="cpu", damp=0.2, ckpt=10)
    assert a.surrogate_damp == 0.2 and a.checkpoint_every == 10
    with pytest.raises(TypeError, match="하나만"):
        fd.Connectome(_c(), "IN", "O", device="cpu", damp=0.2, surrogate_damp=0.3)


def test_calibrate_reaches_targets_and_saves(tmp_path):
    c = _c()
    L = fd.Connectome(c, "IN", ("H", "O"), t_ms=100, device="cpu", input_mode="regular")
    tab = L.calibrate(X, {"H": 30, "O": 15}, iters=12)
    assert set(tab.group) == {"H", "O"}
    with fd.quiescent():
        r = L(X, seed=0, return_all=True).numpy()
    for g, t in (("H", 30), ("O", 15)):
        assert abs(r[:, c.groups[g]].mean() - t) <= 0.25 * t, (g, r[:, c.groups[g]].mean())
    assert any(k.endswith(">H") for k in L.gains) and L.config["gains"] == L.gains
    L.save(tmp_path / "cal")
    M = fd.Connectome.load(tmp_path / "cal.npz", device="cpu")
    assert M.gains == L.gains
    with pytest.raises(ValueError, match="입력 그룹"):
        L.calibrate(X, {"IN": 10})
    with pytest.raises(KeyError, match="회로에 없는"):
        L.calibrate(X, {"Q": 10})
    t2 = fd.Connectome(c, "IN", "O", t_ms=60, device="cpu").calibrate(X, 20)   # 숫자 = 출력 그룹 모두
    assert list(t2.group) == ["O"]


# ─────────────── numpy·torch 습관 ───────────────
def _num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        a, b = x.copy(), x.copy(); a[i] += h; b[i] -= h
        g[i] = (f(a) - f(b)) / (2 * h)
    return g


@pytest.mark.parametrize("name,f", [
    ("min", lambda s: s.min(axis=1).sum()),
    ("var", lambda s: s.var(axis=0, ddof=1).sum()),
    ("std", lambda s: s.std()),
    ("softmax", lambda s: (s.softmax(1) * fd.Signal(np.arange(12.).reshape(3, 4))).sum()),
    ("sqrt·square", lambda s: ((s.square() + 1.0).sqrt()).sum()),
    ("astype·copy·squeeze", lambda s: (s.reshape(1, 3, 4).squeeze().astype(np.float64).copy() ** 2).sum()),
    ("abs", lambda s: abs(s).sum()),
])
def test_new_signal_ops_gradients(name, f):
    x0 = np.random.default_rng(3).standard_normal((3, 4))
    xs = fd.Signal(x0.copy(), plastic=True)
    f(xs).retrograde()
    np.testing.assert_allclose(xs.retro, _num_grad(lambda a: float(f(fd.Signal(a)).data), x0), atol=1e-6)


def test_signal_python_protocols():
    x = fd.Signal(np.array([[1., -2.], [3., 4.]], np.float32))
    assert float(x.sum()) == 6.0 and int(x.max()) == 4 and bool(x.sum() > 0)
    assert x.argmax(1).tolist() == [0, 1] and x.size == 4 and x.tolist() == [[1., -2.], [3., 4.]]
    assert float(np.mean(x).data) == 1.5 and x.any() and not (x > 10).any()
    with pytest.raises(ValueError, match="값 하나인"):
        bool(x)
    with pytest.raises(ValueError, match="값 하나인"):
        float(x)


@pytest.mark.parametrize("code,hint", [
    ("x.grad", ".retro"), ("x.backward()", "retrograde"), ("x.w", ".numpy()"), ("x.requires_grad", "plastic"),
    ("p.parameters()", "synapses()"), ("p.zero_grad()", "clear_retro"), ("p.weigth", "weight"),
    ("rule.zero_grad()", ".clear()"), ("rule.lr", ".rate"), ("c.num_nodes", ".N"),
    ("fd.Linear", "fd.Projection"), ("fd.Adam", "fd.Adaptive"), ("fd.no_grad", "fd.quiescent"),
])
def test_attribute_hints(code, hint):
    x = fd.Signal(np.ones(3, np.float32))
    p = fd.Projection(3, 2, device="cpu")
    rule = fd.Adaptive(p.synapses())
    c = fd.graphs.layered([2, 2, 2], 0.5)
    with pytest.raises(AttributeError, match=None) as e:
        eval(code)
    assert hint in str(e.value), str(e.value)


def test_hints_do_not_break_copy_pickle_hasattr():
    import copy
    import pickle
    p = fd.Pathway(fd.Projection(3, 2, device="cpu"), fd.Activation("relu"))
    x = fd.Signal(np.ones((2, 3), np.float32))
    c = fd.graphs.layered([2, 2, 2], 0.5)
    for obj in (p, x, c, fd.Adaptive(p.synapses())):
        copy.deepcopy(obj)
        pickle.loads(pickle.dumps(obj))
    assert not hasattr(x, "grad") and not hasattr(p, "parameters") and getattr(p, "nope", 7) == 7


# ─────────────── 3차 오류 점검 ───────────────
def test_train_accepts_lists_and_pandas():
    import pandas as pd
    m = fd.Pathway(fd.Projection(6, 2, device="cpu", seed=0))
    Xl = np.random.default_rng(0).random((10, 6)).tolist()
    yl = [0, 1] * 5
    fd.train(m, Xl, yl, epochs=1, verbose=False)
    fd.train(m, pd.DataFrame(Xl), pd.Series(yl), epochs=1, verbose=False)
    assert 0 <= fd.evaluate(m, pd.DataFrame(Xl), yl) <= 1


def test_duplicate_groups_rejected():
    with pytest.raises(ValueError, match="같은 이름이 여러 번"):
        fd.Connectome(_c(), ["IN", "IN"], "O", device="cpu")
    with pytest.raises(ValueError, match="같은 이름이 여러 번"):
        fd.Connectome(_c(), "IN", ["O", "O"], device="cpu")


def test_data_size_check(tmp_path):
    from flydnet import data as D
    if D.missing("door"):
        pytest.skip("DoOR 데이터 없음")
    import shutil
    src = D.data_dir("door")
    for n in D.SOURCES["door"]:
        shutil.copy(src / n, tmp_path / n)
    D.require("door", tmp_path)
    name = next(iter(D.SOURCES["door"]))
    (tmp_path / name).write_bytes((tmp_path / name).read_bytes()[:100])            # 잘린 파일
    with pytest.raises(ValueError, match="크기가 다름"):
        D.require("door", tmp_path)


# ─────────────── 입력 자료형 ───────────────
@pytest.mark.parametrize("make", [
    lambda b: b.astype(np.float32), lambda b: b.astype(np.float16), lambda b: b.astype(np.int64),
    lambda b: b.astype(np.int32), lambda b: np.clip(b, 0, 255).astype(np.uint8), lambda b: b > 100,
    lambda b: b.tolist(), lambda b: __import__("pandas").DataFrame(b),
])
def test_any_input_dtype_computes_in_float32(make):
    base = np.random.default_rng(0).uniform(0, 200, (4, 6))
    x = make(base)
    assert fd.Projection(6, 3, device="cpu")(x).data.dtype == np.float32
    assert fd.Homeostasis()(x).data.dtype == np.float32
    assert _L()(x, seed=0).data.dtype == np.float32
    fd.train(fd.Pathway(fd.Projection(6, 2, device="cpu")), x, [0, 1, 0, 1], epochs=1, verbose=False)


def test_float64_kept_when_chosen():
    x = np.random.default_rng(0).random((2, 6))                             # numpy float64 = 일부러 고른 정밀도
    assert fd.Projection(6, 3, device="cpu")(x).data.dtype == np.float64


def test_torch_dtypes():
    torch = pytest.importorskip("torch")
    base = np.random.default_rng(0).uniform(0, 200, (4, 6))
    for dt in (torch.bfloat16, torch.float16, torch.int64, torch.bool):
        x = torch.tensor(base > 100 if dt == torch.bool else base).to(dt)
        assert fd.Projection(6, 3, device="cpu")(x).data.dtype == np.float32
    m = fd.torch.bridge(_L(trainable=True), seed=0)
    for dt in (torch.bfloat16, torch.float16):
        xt = torch.tensor(base, dtype=dt, requires_grad=True)
        m(xt).sum().backward()
        assert xt.grad.dtype == dt                                          # 기울기는 입력과 같은 자료형으로


def test_label_dtypes():
    z = fd.Signal(np.zeros((3, 2), np.float32))
    for ok in ([0, 1, 1], [0., 1., 1.], np.array([True, False, True]), np.array([0, 1, 1], np.uint8)):
        fd.surprise(z, ok)
    with pytest.raises(ValueError, match="정수가 아닌 값"):
        fd.surprise(z, [0, 1.7, 1])
    with pytest.raises(ValueError, match="np.unique"):
        fd.surprise(z, ["a", "b", "a"])
