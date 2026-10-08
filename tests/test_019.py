"""0.1.19 회귀 시험"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from test_threefactor import _rec

needs_gpu = pytest.mark.skipif(not B.gpu_available(), reason="GPU 없음")


@needs_gpu
def test_gpu_scatter_add_matches_numpy_and_is_deterministic():
    """GPU scatter_add = np.add.at (겹치는 칸·튜플·불리언·슬라이스 섞임·음수·연속 아닌 배열), 반복하면 비트까지 같음.
    예전: cupy.add.at (원자적 덧셈)이라 같은 칸에 많이 더하면 실행마다 마지막 자리가 달랐음"""
    import cupy as cp
    r = np.random.default_rng(0)
    for _ in range(200):
        shape = tuple(int(r.integers(1, 6)) for _ in range(int(r.integers(1, 4))))
        c = r.random()
        if c < 0.3:
            idx = r.integers(-shape[0], shape[0], int(r.integers(0, 12)))
        elif c < 0.5:
            idx = tuple(r.integers(0, d, 7) for d in shape)
        elif c < 0.65:
            idx = r.random(shape) < 0.5
        elif c < 0.8 and len(shape) > 1:
            idx = (slice(None), r.integers(0, shape[1], 5))
        else:
            idx = (r.integers(0, shape[0], 9),)
        base = r.standard_normal(shape)
        vals = r.standard_normal(base[idx].shape)
        ref = base.copy()
        np.add.at(ref, idx, vals)
        tg = cp.asarray(base)
        gi = tuple(cp.asarray(p) if isinstance(p, np.ndarray) else p for p in idx) if isinstance(idx, tuple) \
            else cp.asarray(idx)
        B.scatter_add(tg, gi, cp.asarray(vals))
        np.testing.assert_allclose(cp.asnumpy(tg), ref, rtol=1e-12, atol=1e-12)
    t = cp.zeros((6, 4), cp.float32)[:, ::2]                                      # 연속이 아닌 배열
    B.scatter_add(t, cp.array([0, 0, 3, 5]), cp.ones((4, 2), cp.float32))
    assert float(t[0, 0]) == 2 and float(t[3, 1]) == 1
    idx = cp.asarray(r.integers(0, 40, 400_000))
    v = cp.asarray(r.standard_normal(400_000).astype(np.float32))
    outs = []
    for _ in range(4):
        t = cp.zeros(40, cp.float32)
        B.scatter_add(t, idx, v)
        outs.append(cp.asnumpy(t))
    assert all(np.array_equal(outs[0], o) for o in outs)


@needs_gpu
@pytest.mark.parametrize("kw", [dict(share="pair"), dict(train_neurons=True, t_mbr={"H": 12.0})])
def test_gpu_training_gradients_repeat_bitwise(kw):
    """여러 연결이 값 하나를 함께 쓰는 학습 (share="pair", 세포 유형별 매개변수): GPU에서 같은 계산을 다시 하면 기울기가
    비트까지 같음 (예전: 첫 스텝부터 마지막 자리가 달라 학습이 실행마다 갈렸음)"""
    c = _rec(feedback_edges=True, strong=True)
    X = np.random.default_rng(0).uniform(50, 200, (8, 6)).astype(np.float32)
    res = []
    for _ in range(3):
        L = fd.Connectome(c, "IN", "O", t_ms=40, device="gpu", trainable=True, **kw)
        (L(X, seed=1) * 0.01).sum().retrograde()
        res.append([B.numpy(s.retro) for s in L.synapses()])
    assert all(np.array_equal(a, b) for r_ in res[1:] for a, b in zip(res[0], r_))


@pytest.mark.filterwarnings("ignore:보정 뒤에도 목표에")                  # 반복 3번이라 목표에 덜 닿음 - 자료형만 확인
def test_calibrate_accepts_numpy_scalar_target():
    """목표에 numpy 숫자: 예전엔 dict로 보다 'numpy.float32 object is not iterable'"""
    c = fd.graphs.layered([10, 40, 5], 0.3, weight=30, seed=0)
    X = np.full((2, 10), 100, np.float32)
    tabs = []
    for t in (10.0, np.float32(10), np.int64(10)):
        L = fd.Connectome(c, "in", "out", t_ms=40, device="cpu")
        tabs.append(L.calibrate(X, t, iters=3))
    for t in tabs[1:]:
        assert list(t.after_hz) == list(tabs[0].after_hz) and list(t.target_hz) == list(tabs[0].target_hz)


# ─────────────── 음수 입력 (RateEncoder·Learner) ───────────────
def test_rate_encoder_warns_once_on_negative_inputs():
    """음수가 많은 입력 (StandardScaler 등): 예전엔 경고 없이 발화율 0으로 잘려 정확도가 조용히 떨어졌음
    (보고된 예: Iris 0.867 → 0.733). 음수 비율 5% 이상이면 한 번 알림, 적으면 조용히"""
    import warnings
    X = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)        # 음수 약 50%
    for proj in ("random", None):
        enc = fd.RateEncoder(8, 8 if proj is None else 30, projection=proj, device="cpu")
        with pytest.warns(UserWarning, match="음수"):
            enc(X)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            enc(X)                                                                  # 한 번만
    few = np.abs(X)
    few[0, 0] = -1.0                                                                # 음수 1/160 < 5%
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fd.RateEncoder(8, 30, device="cpu")(few)


def test_learner_warns_on_negative_inputs():
    X = np.random.default_rng(0).standard_normal((24, 6)).astype(np.float32)
    y = np.arange(24) % 3
    with pytest.warns(UserWarning, match="음수"):
        fd.Learner(_rec(feedback_edges=True, strong=True), "IN", "O", t_ms=30, device="cpu").learn(X, y)


def test_rate_encoder_onoff_channels():
    """negative="onoff": 부호 있는 특징을 ON(max(x, 0))·OFF(max(-x, 0)) 두 채널로 (시각계 ON/OFF 경로처럼) - 음수 정보가
    남고 경고 없음. 기본("clip")은 예전과 같은 값"""
    import warnings
    X = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
    enc = fd.RateEncoder(4, 8, projection=None, negative="onoff", device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = enc(X).numpy()
    both = np.concatenate([np.maximum(X, 0), np.maximum(-X, 0)], 1)
    np.testing.assert_allclose(r, both / both.max(1, keepdims=True) * 100, rtol=1e-6)
    assert not np.array_equal(enc(X).numpy(), enc(-X).numpy())                       # 부호가 바뀌면 다른 발화율
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip = fd.RateEncoder(4, 4, projection=None, device="cpu")
        assert np.array_equal(clip(np.abs(X)).numpy(), fd.RateEncoder(4, 4, projection=None, device="cpu")(np.abs(X)).numpy())
    e2 = fd.RateEncoder(4, 30, negative="onoff", device="cpu")                      # 투영이면 2 x n_in 채널에서 모음
    assert e2.P.shape == (30, 8) and e2(X).shape == (5, 30)
    with pytest.raises(ValueError, match="2"):
        fd.RateEncoder(4, 4, projection=None, negative="onoff", device="cpu")         # 1:1이면 n_out = 2 x n_in
    with pytest.raises(ValueError, match="negative"):
        fd.RateEncoder(4, 4, negative="abs", device="cpu")
    L = fd.Learner(_rec(feedback_edges=True, strong=True), "IN", "O", t_ms=30, device="cpu", negative="onoff")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L.learn(np.random.default_rng(0).standard_normal((24, 6)).astype(np.float32), np.arange(24) % 3)
    assert L.encoders["default"].negative == "onoff"


# ─────────────── Signal 인자 ───────────────
def test_signal_plastic_must_be_bool():
    """fd.Signal(1, 5): 예전엔 bool(5)로 조용히 학습 신호가 됨 → 참·거짓만 (numpy bool도)"""
    for bad in (5, 0, 1.0, "yes", None):
        with pytest.raises(TypeError, match="plastic"):
            fd.Signal(1.0, bad)
    assert fd.Signal(1.0, np.bool_(True)).plastic is True and fd.Signal(1.0, False).plastic is False
