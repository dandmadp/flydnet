"""전체 코드 검토(0.1.16)에서 찾은 버그의 회귀 테스트 - 모두 예전에는 조용히 틀리거나 엉뚱한 오류를 냈음"""
import warnings

import numpy as np
import pandas as pd
import pytest

import flydnet as fd
from flydnet.ganglion.signal import Signal, checkpoint, quiescent
from test_threefactor import _rec


def _strong():
    c = _rec(feedback_edges=True)
    return fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)


X = np.random.default_rng(0).uniform(50, 200, (3, 6)).astype(np.float32)


# ─────────────── 자동 미분 (signal.py) ───────────────
def test_clip_accepts_array_bounds():
    x = Signal(np.array([[1., 5.], [3., 2.]], np.float32))
    np.testing.assert_array_equal(x.clip(np.zeros(2), np.full(2, 4.0)).data, [[1, 4], [3, 2]])
    with pytest.raises(ValueError, match="lo"):
        x.clip(np.array([0, 5]), np.array([1, 4]))


def test_max_backward_keeps_float32():
    x = Signal(np.array([[1., 5.], [5., 2.]], np.float32), plastic=True)
    x.max(axis=1).sum().retrograde()
    assert x.retro.dtype == np.float32


def test_retro_shape_must_match():
    x = Signal(np.ones((2, 2), np.float32), plastic=True)
    with pytest.raises(ValueError, match="retro 모양"):
        (x * 2).retrograde(np.ones((1, 2), np.float32))


def test_shared_intermediate_freed_by_other_loss_raises():
    """두 손실이 중간값을 공유하면 두 번째 retrograde가 기울기를 조용히 버리던 것 (a.retro가 8이 아니라 2)"""
    a = Signal(np.ones(3, np.float32), plastic=True)
    h = a * 2
    l1, l2 = h.sum(), (h * 3).sum()
    l1.retrograde()
    with pytest.raises(RuntimeError, match="이미 풀었음"):
        l2.retrograde()
    a = Signal(np.ones(3, np.float32), plastic=True)
    h = a * 2
    h.sum().retrograde(keep=True)
    (h * 3).sum().retrograde()
    np.testing.assert_array_equal(a.retro, [8, 8, 8])


def test_checkpoint_backward_inside_quiescent():
    w = Signal(np.ones(3, np.float32), plastic=True)
    loss = checkpoint(lambda v: (v * w * 2,), Signal(np.ones(3, np.float32)))[0].sum()
    with quiescent():
        loss.retrograde()
    np.testing.assert_array_equal(w.retro, [2, 2, 2])


# ─────────────── ConnectomeLayer ───────────────
@pytest.mark.parametrize("tm", [5.0, 5.001, 7.0])
def test_group_t_mbr_equal_to_tau_matches_scalar(tm):
    """그룹별 t_mbr = tau(5 ms)에서 정확한 적분 계수가 0/0이 되어 출력이 0 Hz이던 것"""
    c = _strong()
    x = np.full((4, 6), 150, np.float32)
    ref = fd.Connectome(c, "IN", "O", t_ms=40, device="cpu", params={"t_mbr": tm})(x, seed=1).data
    grp = fd.Connectome(c, "IN", "O", t_ms=40, device="cpu", params={"t_mbr": tm},
                        t_mbr={g: tm for g in c.groups})(x, seed=1).data
    np.testing.assert_allclose(grp, ref, rtol=1e-5)


def test_calibrate_group_without_inputs_raises():
    """들어오는 연결이 없는 그룹은 배율로 보정할 수 없음 - 예전엔 아무것도 못 바꾸고 조용히 끝남"""
    c = fd.Circuit.from_edges([0, 0], [1, 1], [20.0, 20.0], groups={"IN": [0], "H": [1], "L": [2]}, n=3)
    layer = fd.Connectome(c, "IN", ("H", "L"), t_ms=20, device="cpu")
    with pytest.raises(ValueError, match="들어오는 연결"):
        layer.calibrate(np.full((2, 1), 100.0, np.float32), {"L": 10})


def test_projection_accepts_1d_and_nd():
    p = fd.Projection(6, 2, device="cpu", seed=0)
    x = np.random.rand(4, 5, 6).astype(np.float32)
    flat = p(x.reshape(-1, 6)).data.reshape(4, 5, 2)
    np.testing.assert_allclose(p(x).data, flat, rtol=1e-6)
    np.testing.assert_allclose(p(x[0, 0]).data, flat[0, 0], rtol=1e-6)


# ─────────────── Circuit ───────────────
def _small():
    return fd.Circuit(np.arange(4), {"A": [0, 1], "B": [2, 3]}, [0, 1, 2, 3], [2, 3, 0, 1], [1, 2, 3, 4],
                      meta=pd.DataFrame({"gaba": [True, False, True, False], "size": [1.5, 2.0, 3.0, 4.0]}))


def test_subset_rejects_unknown_group():
    c = _small()
    with pytest.raises(KeyError, match="Bx"):
        c.subset(["A", "Bx"])
    assert c.subset("A").N == 2 and c.subset(["A", "A"]).N == 2


def test_ungrouped_neurons_do_not_break_controls():
    c = fd.Circuit(np.arange(4), {"A": [0, 1]}, [0, 1, 2, 3], [2, 3, 0, 1], [1, 2, 3, 4])
    for f in (c.shuffled, c.randomized, c.shuffled_weights):
        assert f().n_edges == 4
    assert "?" in set(c.summary().index.get_level_values(0))


def test_meta_types_survive_save_and_driver_matches(tmp_path):
    c = _small()
    p = fd.Connectome(c, "A", "B", device="cpu", t_ms=5).save(tmp_path / "x")
    c2 = fd.Connectome.load(p).circuit
    assert c2.meta.gaba.dtype == bool and c2.meta["size"].dtype == np.float64
    for cc in (c, c2):
        assert len(fd.genetics.driver(cc, gaba=True)) == 2
        assert len(fd.genetics.driver(cc, size=[1.5, 3])) == 2


def test_normalized_zero_input_is_zero_not_nan():
    z = fd.Circuit(np.arange(3), {"A": [0, 1, 2]}, [0, 1], [2, 2], [0.0, 0.0])
    np.testing.assert_array_equal(z.normalized().weight, [0, 0])


# ─────────────── genetics · explain · gradcheck · compare ───────────────
def test_screen_checks_all_lines_before_running():
    c = _strong()
    layer = fd.Connectome(c, "IN", "O", t_ms=20, device="cpu")
    calls = []

    def measure(l, s):
        calls.append(s)
        return 1.0
    with pytest.raises(ValueError, match="'IN'"):
        fd.genetics.screen(measure, layer, fd.genetics.lines(c, by="group"), effector="activate", seeds=2,
                           verbose=False)
    assert not calls                                                   # 측정을 시작하기 전에 멈춤


def test_holm_keeps_nan():
    from flydnet.genetics import _holm
    out = _holm(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.04])


def test_explain_and_gradcheck_leave_training_gradients_alone():
    layer = fd.Connectome(_strong(), "IN", "O", t_ms=30, device="cpu", trainable=True, share="pair")
    layer(X, seed=0).sum().retrograde()
    before = layer.log_scale.retro.copy()
    fd.explain(lambda l, s: l(X, seed=s).sum(), layer)
    np.testing.assert_array_equal(layer.log_scale.retro, before)
    fd.gradcheck(lambda l, s: l(X, seed=s).sum(), layer)
    np.testing.assert_array_equal(layer.log_scale.retro, before)


def test_sign_flip_p_is_scale_invariant():
    d = np.array([3, 2, 4, 3, 5, 2, 3, 4.0])
    ps = {fd.sign_flip_p(d * s) for s in (1, 1e-9, 1e-14, 1e6)}
    assert len(ps) == 1 and ps.pop() < 0.01


# ─────────────── 인코더 · 확장 · 그래프 · 학습 ───────────────
def test_rate_encoder_1d_is_one_sample():
    """1차원 [1, 2, 3, 4]를 특징 1개짜리 시료 4개로 봐서 모두 100 Hz이던 것"""
    e = fd.RateEncoder(4, 4, projection=None, device="cpu")
    np.testing.assert_allclose(e(np.array([1., 2, 3, 4])).data, [25, 50, 75, 100])
    with pytest.raises(ValueError, match="특징"):
        e(np.ones((2, 5)))


def test_erdos_renyi_exact_edge_count_without_self_loops():
    for n, p in ((5, 0.5), (5, 1.0), (40, 0.2)):
        c = fd.graphs.erdos_renyi(n, p, seed=3)
        assert (c.pre != c.post).all()
        if p == 1.0:
            assert c.n_edges == n * (n - 1)


@pytest.mark.parametrize("bad", [lambda: fd.graphs.layered([5], 0.5),
                                 lambda: fd.graphs.stochastic_block({"A": 3}, [[1.5]]),
                                 lambda: fd.graphs.stochastic_block({"A": 3}, {("A", "Z"): 0.1})])
def test_graph_generators_reject_bad_specs(bad):
    with pytest.raises((ValueError, KeyError)):
        bad()


def test_train_checks_val_before_training():
    model = fd.Pathway(fd.Projection(6, 2, device="cpu"))
    with pytest.raises(ValueError, match="val"):
        fd.train(model, X, [0, 1, 0], val=(X,), epochs=1, verbose=False)
    with pytest.raises(ValueError):
        fd.extract(model, None, np.zeros((0, 6)))
