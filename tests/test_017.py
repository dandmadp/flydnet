"""0.1.17에서 고친 버그의 회귀 테스트 (연산 경계·Linear 계열·입력 신호)"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from flydnet.ganglion.signal import Signal
from test_threefactor import _rec


def test_transpose_negative_axes_backward():
    """음수 축을 그대로 정렬해 역순열이 틀렸던 것: (2, 3) 신호에 (3, 2) 기울기가 쌓였음"""
    x = np.arange(6.0).reshape(2, 3)
    R = np.random.default_rng(0).standard_normal((3, 2))
    for axes in ((-1, 0), ((1, 0),), (1, -2)):
        s = Signal(x, plastic=True)
        (s.transpose(*axes) * Signal(R)).sum().retrograde()
        np.testing.assert_allclose(s.retro, R.T)
    with pytest.raises(ValueError, match="transpose"):
        Signal(x).transpose(0, 0)


def test_pow_zero_has_zero_gradient():
    s = Signal(np.array([0.0, 2.0]), plastic=True)
    (s ** 0).sum().retrograde()
    np.testing.assert_array_equal(s.retro, [0, 0])


def test_eq_ne_are_elementwise_and_hash_is_identity():
    """예전에는 객체 비교라 (s == 0)이 늘 False 하나 - (spk == 1).sum() 같은 코드가 조용히 틀렸음"""
    s = Signal(np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal((s == 0).data, [True, False, True])
    np.testing.assert_array_equal((s != 0).data, [False, True, False])
    np.testing.assert_array_equal((s == Signal(np.array([0.0, 1, 1]))).data, [True, True, False])
    t = Signal(np.array([0.0, 1.0, 0.0]))
    assert len({s: 1, t: 2}) == 2                                          # 값이 같아도 다른 신호는 다른 키


@pytest.mark.parametrize("a,b", [((3,), (3, 2)), ((2, 3), (3,)), ((3,), (3,))])
def test_matmul_1d_like_numpy(a, b):
    rng = np.random.default_rng(1)
    A, Bm = rng.standard_normal(a), rng.standard_normal(b)
    sa, sb = Signal(A, plastic=True), Signal(Bm, plastic=True)
    out = sa @ sb
    assert out.shape == np.shape(A @ Bm)
    np.testing.assert_allclose(out.data, A @ Bm)
    out.sum().retrograde()
    ones = np.ones(np.shape(A @ Bm))
    ga = (ones @ Bm.T) if A.ndim == 1 and Bm.ndim == 2 else np.outer(ones, Bm) if Bm.ndim == 1 and A.ndim == 2 else Bm * ones
    np.testing.assert_allclose(sa.retro, ga)


def test_number_power_signal():
    s = Signal(np.array([1.0, 2.0]), plastic=True)
    y = 2 ** s
    y.sum().retrograde()
    np.testing.assert_allclose(y.data, [2, 4])
    np.testing.assert_allclose(s.retro, np.log(2) * np.array([2, 4]))
    with pytest.raises(ValueError, match="양수"):
        (-2) ** s


def test_clip_numpy_bounds_on_gpu():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    s = Signal(np.array([[1.0, 5.0], [3.0, -2.0]], np.float32), device="gpu", plastic=True)
    y = s.clip(np.zeros(2), np.full(2, 4.0))
    y.sum().retrograde()
    np.testing.assert_array_equal(y.numpy(), [[1, 4], [3, 0]])
    np.testing.assert_array_equal(B.numpy(s.retro), [[1, 0], [1, 0]])


def test_saturating_input_rate_warns_once():
    """입력 뉴런은 두 스텝에 한 번까지만 발화 (dt 0.1 ms면 5 kHz) - 그 이상은 조용히 잘렸음"""
    layer = fd.Connectome(_rec(strong=True), "IN", "O", t_ms=20, device="cpu")
    with pytest.warns(UserWarning, match="포화"):
        layer(np.full((1, 6), 20000, np.float32), seed=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer(np.full((1, 6), 20000, np.float32), seed=0)                 # 한 번만


def test_calibrate_and_reach_do_not_raise_silent_warning():
    """보정·진단 중에는 '출력이 모두 0' 경고를 내지 않음 (바로 그것을 고치는 중)"""
    import warnings
    c = fd.graphs.layered([20, 100, 100, 10], 0.1, seed=0)
    x = np.full((4, 20), 100, np.float32)
    layer = fd.Connectome(c, "in", "out", device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer.reach(x)
        layer.calibrate(x, {"out": 10})


def test_concat_accepts_arrays():
    from flydnet.ganglion.signal import concat
    a = Signal(np.ones((1, 2)), plastic=True)
    y = concat([a, np.zeros((1, 2))])
    y.sum().retrograde()
    np.testing.assert_array_equal(y.data, [[1, 1], [0, 0]])
    np.testing.assert_array_equal(a.retro, [[1, 1]])


def _colab():
    """Colab에서 실제로 쓴 코드: 입력 2개 → 출력 1개, torch.rand(32, 2) (0~1 값)"""
    c = fd.Circuit([0, 1, 2], {"input_group": [0, 1], "output_group": [2]}, [0, 1], [2, 2], [1.0, 1.0])
    x = np.random.default_rng(0).random((32, 2)).astype(np.float32)
    return c, x


def test_unit_hint_for_normalized_inputs():
    """0~1 입력은 Hz로 해석돼 거의 입력이 없음 - 예전에는 "연결이 약함"으로 잘못 안내했음"""
    c, x = _colab()
    layer = fd.ConnectomeLayer(c, inputs=["input_group"], outputs=["output_group"], device="cpu")
    with pytest.warns(UserWarning, match="발화율\(Hz\)") as rec:
        layer(x)
    assert not any("연결이 약함" in str(w.message) for w in rec)


def test_calibrate_refuses_quiet_inputs_instead_of_huge_gains():
    """예전: 0~1 입력에 배율을 43억 배까지 올리고 목표 20 Hz 대신 0.6 Hz로 조용히 끝남"""
    c, x = _colab()
    layer = fd.ConnectomeLayer(c, inputs=["input_group"], outputs=["output_group"], device="cpu")
    with pytest.raises(ValueError, match="Hz로 바꿀 것"):
        layer.calibrate(x)
    assert layer.gains == {}                                               # 아무것도 바꾸지 않음
    tab = layer.calibrate(x * 100)                                         # Hz로 바꾸면 맞춰짐
    assert abs(float(tab.after_hz[0]) - 20) <= 2 and float(tab.gain_factor[0]) < 1000


def test_calibrate_warns_when_target_missed_and_caps_gain():
    c, x = _colab()
    layer = fd.ConnectomeLayer(c, inputs=["input_group"], outputs=["output_group"], device="cpu")
    with pytest.warns(UserWarning, match="max_gain"):
        tab = layer.calibrate(x * 100, target=400, max_gain=5)           # 입력보다 빠른 출력은 불가능
    assert float(tab.gain_factor[0]) <= 5 + 1e-9


# ─────────────── 한 스텝 합친 GPU 커널 ───────────────
def _gpu_or_skip():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    from flydnet.ganglion import kernels as K
    if K._cuda() is None:
        pytest.skip("전용 커널 없음")
    return K


def test_poisson_kernel_bitwise_equals_hash():
    K = _gpu_or_skip()
    import cupy as cp
    from flydnet.ganglion.physiology import hash_uniform
    p = cp.asarray(np.random.default_rng(0).random((37, 5)).astype(np.float32) * 0.3)
    for seed, step in ((0, 0), (3, 17), (2 ** 40, 999)):
        ref = (hash_uniform(cp, seed, step, (5, 37)).T < p).astype(cp.float32)
        assert bool((K.poisson_spikes(cp, seed, step, p) == ref).all())


@pytest.mark.parametrize("kw,effect", [({}, None), (dict(ckpt=50, truncate=60, noise=0.5), None),
                                       ({}, "activate"), ({}, "mosaic")])
def test_fused_lif_bitwise_equals_elementwise(monkeypatch, kw, effect):
    """합친 커널(순전파·역전파 각각 하나)이 원소별 연산 경로와 비트 단위로 같음 (출력·연결 기울기·입력 기울기)"""
    K = _gpu_or_skip()
    from flydnet.ganglion import physiology as P
    c = _rec(feedback_edges=True, strong=True)
    X = np.random.default_rng(0).uniform(50, 200, (5, 6)).astype(np.float32)

    def run():
        layer = fd.Connectome(c, "IN", "O", t_ms=40, device="gpu", trainable=True, **kw)
        eff = (fd.genetics.activate(layer, fd.genetics.driver(c, group="H"), hz=80) if effect == "activate" else
               fd.genetics.mosaic(layer, p=0.4) if effect == "mosaic" else None)
        x = fd.Signal(X, device="gpu", plastic=True)
        y = layer(x, seed=2)
        (y * y).sum().retrograde()
        if eff is not None:
            eff.remove()
        return [B.numpy(y.data), B.numpy(layer.log_scale.retro), B.numpy(x.retro)]
    fast = run()
    monkeypatch.setattr(K, "_fused_ok", lambda *a, **k: False)
    monkeypatch.setattr(K, "poisson_spikes",
                        lambda xp, seed, step, p: (P.hash_uniform(xp, seed, step, p.shape[::-1]).T < p).astype(p.dtype))
    slow = run()
    assert fast[0].mean() > 0
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a, b)


def test_backward_memory_per_step_stays_small():
    """순전파가 역전파용으로 붙잡는 메모리: 원소·스텝당 약 7바이트 (u·발화·적분 여부·1바이트 스파이크).
    예전에는 막전위·전류·스파이크·누적 사슬을 모두 붙잡아 약 34바이트 (전체 뇌 200스텝 6.9 GB)"""
    _gpu_or_skip()
    import cupy as cp
    c = fd.graphs.erdos_renyi(4000, 0.01, weight=20.0, groups={"in": np.arange(100), "out": np.arange(3900, 4000)})
    X = np.full((16, 100), 80, np.float32)
    layer = fd.Connectome(c, "in", "out", t_ms=40, device="gpu", trainable=True)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    base = pool.used_bytes()
    y = layer(X, seed=0)
    cp.cuda.Device().synchronize()
    per = (pool.used_bytes() - base) / (c.N * 16 * 400)
    assert per < 12, f"원소·스텝당 {per:.1f}바이트"
    y.sum().retrograde()
    assert layer.log_scale.retro is not None and bool(cp.isfinite(layer.log_scale.retro).all())


def test_fused_inverse_cache_not_fooled_by_reused_gpu_memory(monkeypatch):
    """크기가 같은 집단을 차례로 activate하면, GPU 주소로 캐시한 역표가 해제 뒤 같은 주소의 새 목록에 붙어
    두 번째 A 활성화가 B의 뉴런을 자극했음 (0.1.17 합친 커널에서 생겼다가 고침)"""
    K = _gpu_or_skip()
    G = fd.genetics
    c = _rec(feedback_edges=True, strong=True)
    H = c.groups["H"]
    lines = [G.Line(c, H[i:i + 5], str(i)) for i in (0, 5, 10, 15)]

    def run(line):
        layer = fd.Connectome(c, "IN", "O", t_ms=30, device="gpu")
        with G.activate(layer, line, hz=300), fd.quiescent():
            return layer(np.zeros((2, 6), np.float32), seed=1, return_all=True).numpy()
    fast = [run(ln) for ln in lines * 3]
    monkeypatch.setattr(K, "_fused_ok", lambda *a, **k: False)
    slow = [run(ln) for ln in lines * 3]
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a, b)
    assert fast[4][0, H[:5]].mean() > fast[4][0, H[5:10]].mean()          # 두 번째 A: A 뉴런이 자극됨


# ─────────────── 전체 검토 2회차 ───────────────
def test_stdp_sees_post_spikes_of_blocked_neurons():
    """Shibire(block)는 발화는 그대로 두고 전달만 막음 - STDP의 시냅스 후 흔적은 실제 발화로 (예전: 0이 되어 학습 없음)"""
    c = _rec(feedback_edges=True, strong=True)
    X = np.random.default_rng(0).uniform(50, 200, (4, 6)).astype(np.float32)
    layer = fd.Connectome(c, "IN", "O", t_ms=60, device="cpu", trainable=True)
    st = fd.STDP(layer)
    onto = np.isin(B.numpy(layer.wiring.post)[B.numpy(layer.train_pos)], c.groups["O"])
    with fd.genetics.block(layer, fd.genetics.driver(c, group="O")):
        st(X, seed=1)
        d = st.assign()
    assert np.abs(B.numpy(d)[onto]).sum() > 0


def test_tune_surrogate_restores_layer_on_failure():
    layer = fd.Connectome(_rec(strong=True), "IN", "O", t_ms=80, device="cpu", trainable=True, share="pair")
    n = [0]

    def score(l, s):
        n[0] += 1
        if n[0] > 3:
            raise RuntimeError("score 실패")
        return l(np.full((2, 6), 150, np.float32), seed=s).sum()
    with pytest.raises(RuntimeError):
        fd.tune(score, layer, verbose=False)
    assert layer.surrogate_damp == "auto"


def test_load_state_is_all_or_nothing():
    a = fd.Pathway(fd.Projection(4, 3, device="cpu", seed=0), fd.Projection(3, 2, device="cpu", seed=1))
    st = a.state()
    bad = dict(st, **{"0.weight": st["0.weight"] + 1, "1.weight": np.zeros((5, 5), np.float32)})
    with pytest.raises(ValueError, match="1.weight"):
        a.load_state(bad)
    np.testing.assert_array_equal(a.state()["0.weight"], st["0.weight"])  # 앞 값도 그대로


def test_calibrate_restores_gains_when_interrupted():
    c = fd.graphs.layered([20, 100, 10], 0.15, seed=0)
    layer = fd.Connectome(c, "in", "out", device="cpu")
    n, orig = [0], layer.forward

    def flaky(*a, **k):
        n[0] += 1
        if n[0] == 4:
            raise KeyboardInterrupt
        return orig(*a, **k)
    layer.forward = flaky
    with pytest.raises(KeyboardInterrupt):
        layer.calibrate(np.full((4, 20), 100, np.float32), {"out": 10})
    assert layer.gains == {}


def test_option_combinations_fused_equals_elementwise():
    """옵션 조합(체크포인팅·절단·잡음·regular·활성화 둘·block·silence·mosaic·감쇠·ThreeFactor)에서 합친 커널 = 원소별 경로.
    하나씩만 시험하면 놓치는 조합 버그용 (역표 캐시 버그는 같은 크기 집단을 연속 활성화할 때만 나왔음)"""
    _gpu_or_skip()
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).with_name("combo_check.py")
    spec = importlib.util.spec_from_file_location("combo_check_mod", p)
    m = importlib.util.module_from_spec(spec)
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["combo_check", "7", "8"]
    try:
        spec.loader.exec_module(m)
    finally:
        _sys.argv = old
        m.elementwise(False)
    assert m.bad == 0


# ─────────────── 5회 검토 - 2회차 (상태·생명주기) ───────────────
def test_save_keeps_attributes_changed_after_construction(tmp_path):
    """layer.t_ms = 80 등 만든 뒤 바꾼 값이 저장에서 빠지고 불러온 층이 다르게 동작하던 것"""
    c = _rec(strong=True)
    X = np.full((2, 6), 150, np.float32)
    layer = fd.Connectome(c, "IN", "O", t_ms=40, device="cpu")
    layer.t_ms, layer.noise, layer.slope, layer.count_from_ms = 80, 0.5, 5.0, 10.0
    layer.p["w_syn"] = 0.5
    with fd.quiescent():
        a = layer(X, seed=0).numpy()
    m = fd.Connectome.load(layer.save(tmp_path / "l"))
    assert (m.t_ms, m.noise, m.slope, m.count_from_ms, m.p["w_syn"]) == (80, 0.5, 5.0, 10.0, 0.5)
    with fd.quiescent():
        np.testing.assert_array_equal(m(X, seed=0).numpy(), a)


def test_changing_w_syn_after_construction_applies_to_connections():
    """p["w_syn"]을 바꾸면 입력 자극에만 반영되고 연결 세기는 그대로이던 것 (같은 값인데 55 Hz 대 5 Hz)"""
    c = _rec(strong=True)
    X = np.full((2, 6), 150, np.float32)
    a = fd.Connectome(c, "IN", "O", t_ms=60, device="cpu", params={"w_syn": 0.6})
    b = fd.Connectome(c, "IN", "O", t_ms=60, device="cpu")
    b.p["w_syn"] = 0.6
    with fd.quiescent():
        np.testing.assert_array_equal(a(X, seed=0).numpy(), b(X, seed=0).numpy())


def test_threefactor_after_moving_layer_to_gpu():
    _gpu_or_skip()
    c = _rec(feedback_edges=True, strong=True)
    X = np.random.default_rng(0).uniform(50, 200, (3, 6)).astype(np.float32)
    for fb in ("random", "connectome"):
        layer = fd.Connectome(c, "IN", "O", t_ms=40, device="cpu", trainable=True)
        tf = fd.ThreeFactor(layer, feedback=fb, seed=0)
        layer.to("gpu")
        o = tf(X, seed=1)
        o.sum().retrograde()
        assert np.isfinite(B.numpy(tf.assign(o))).all()


def test_bridge_optimizer_still_trains_after_load_state(tmp_path):
    torch = pytest.importorskip("torch")
    from test_genetics import _chain
    layer = fd.ConnectomeLayer(_chain(), "A", ("O",), t_ms=30, trainable=True, device="cpu", neuron="graded")
    br = fd.torch.bridge(layer, seed=0)
    opt = torch.optim.SGD(br.parameters(), lr=0.5)
    layer.load_state(fd.ConnectomeLayer.load(layer.save(tmp_path / "s"), device="cpu").state())
    before = layer.log_scale.numpy().copy()
    br(torch.rand(4, 6)).pow(2).sum().backward()
    opt.step()
    assert not np.allclose(layer.log_scale.numpy(), before)              # 옵티마이저가 바꾼 값이 엔진에 반영
