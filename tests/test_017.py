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
