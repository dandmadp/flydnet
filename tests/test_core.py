"""flydnet 핵심 동작 테스트 (자체 엔진). FlyWire 데이터가 없으면 회로 관련 테스트는 건너뜀

    pytest -q
0.1.17까지는 torch판 복사본(fd.torch.*)으로 같은 것을 시험했음 - 0.1.18에서 torch판을 빼며 자체 엔진으로 옮김
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B

HAS_DATA = not fd.data.missing("flywire")
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"FlyWire 데이터 없음: {fd.data_dir('flywire')}")


@pytest.fixture(scope="module")
def mb():
    return fd.Circuit.from_flywire()


def _np(x):
    return x.numpy() if isinstance(x, fd.Signal) else B.numpy(x)


# ─────────────── Circuit ───────────────
@needs_data
def test_mushroom_body_sizes(mb):
    assert {k: len(v) for k, v in mb.groups.items()} == {"PN": 344, "KC": 2597, "APL": 1, "MBON": 48}
    assert mb.pre.min() >= 0 and mb.post.max() < mb.N
    assert (mb.weight != 0).all()


@needs_data
def test_kc_are_excitatory_apl_inhibitory(mb):
    g = mb.group_of()
    assert (mb.weight[g[mb.pre] == "KC"] > 0).all()
    assert (mb.weight[g[mb.pre] == "APL"] < 0).all()


@needs_data
def test_shuffled_keeps_degrees_per_group_pair(mb):
    sh = mb.shuffled(seed=0)
    g = mb.group_of()

    def degrees(c):
        key = g[c.pre] + ">" + g[c.post]
        out = pd.Series(1, index=pd.MultiIndex.from_arrays([key, c.pre])).groupby(level=[0, 1]).size()
        inn = pd.Series(1, index=pd.MultiIndex.from_arrays([key, c.post])).groupby(level=[0, 1]).size()
        return out.sort_index(), inn.sort_index()

    (o1, i1), (o2, i2) = degrees(mb), degrees(sh)
    pd.testing.assert_series_equal(o1, o2)
    pd.testing.assert_series_equal(i1, i2)
    assert (mb.post != sh.post).mean() > 0.5          # 실제로 섞였는지


# ─────────────── RateEncoder ───────────────
def test_encoder_shape_and_range():
    enc = fd.RateEncoder(784, 344, max_rate=100, device="cpu")
    r = _np(enc(np.random.default_rng(0).random((8, 28, 28)).astype(np.float32)))
    assert r.shape == (8, 344)
    assert r.min() >= 0 and np.allclose(r.max(1), 100.0)


def test_encoder_identity_requires_same_size():
    with pytest.raises(ValueError):
        fd.RateEncoder(10, 20, projection=None)


# ─────────────── ConnectomeLayer ───────────────
@needs_data
def test_layer_shape_and_zero_input(mb):
    layer = fd.ConnectomeLayer(mb, "PN", ("KC", "MBON"), t_ms=20)
    with fd.quiescent():
        r = _np(layer(np.zeros((4, 344), np.float32)))
    assert r.shape == (4, 2597 + 48)
    assert (r == 0).all()


@needs_data
def test_layer_regular_is_deterministic(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=30, gains={"PN>KC": 2.0}, input_mode="regular")
    x = fd.RateEncoder(784, 344)(np.random.default_rng(0).random((4, 784)).astype(np.float32))
    with fd.quiescent():
        a, b = _np(layer(x, seed=1)), _np(layer(x, seed=2))
    assert np.array_equal(a, b) and a.sum() > 0


@needs_data
def test_layer_poisson_seed_reproducible(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=30, gains={"PN>KC": 2.0})
    x = np.full((2, 344), 100.0, np.float32)
    with fd.quiescent():
        assert np.array_equal(_np(layer(x, seed=5)), _np(layer(x, seed=5)))


# ─────────────── DopamineReadout ───────────────
def _toy(n=400, k=50, c=4, seed=0):
    g = np.random.default_rng(seed)
    protos = (g.random((c, k)) < 0.2).astype(np.float32)
    y = g.integers(0, c, n)
    X = ((protos[y] + 0.3 * g.random((n, k))) * 50).astype(np.float32)
    return X, y


@pytest.mark.parametrize("mode", ["bidir", "assoc"])
def test_readout_learns_toy_problem(mode):
    X, y = _toy()
    m = fd.DopamineReadout(X.shape[1], 4, mode, lr=0.1, device="cpu").fit(X, y, epochs=3)
    assert m.accuracy(X, y) > 0.9


def test_bidir_weights_stay_nonnegative():
    X, y = _toy()
    m = fd.DopamineReadout(X.shape[1], 4, "bidir", lr=1.0, device="cpu").fit(X, y, epochs=3)
    assert (_np(m.W) >= 0).all()


def test_assoc_is_order_independent():
    X, y = _toy()
    a = fd.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X, y, seed=0)
    order = np.argsort(y, kind="stable")                # 클래스별로 몰아서 (연속 학습과 같은 순서)
    b = fd.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X[order], y[order], seed=1)
    assert np.allclose(_np(a.W), _np(b.W), atol=1e-5)


def test_invalid_mode():
    with pytest.raises(ValueError):
        fd.DopamineReadout(10, 2, "nope")


# ─────────────── GlomerularEncoder / synthetic_odors ───────────────
@needs_data
def test_glomerular_encoder(mb):
    enc = fd.GlomerularEncoder(mb, device="cpu")
    P = _np(enc.P)
    assert enc.n_glomeruli == 56
    assert P.sum() == 139                              # 단일 사구체형 PN 139개, 각자 사구체 하나
    assert (P.sum(1) <= 1).all()
    odor = np.zeros((1, 56), np.float32); odor[0, enc.glomeruli.index("DM1")] = 1.0
    r = _np(enc(odor))[0]
    on = mb.meta.iloc[mb.groups["PN"]].cell_type.str.startswith("DM1_").values
    assert (r[on] == 100).all() and (r[~on] == 0).all()


@needs_data
def test_shuffled_keeps_meta(mb):
    pd.testing.assert_frame_equal(mb.shuffled(0).meta, mb.meta)


def test_synthetic_odors_shapes_and_determinism():
    a = fd.synthetic_odors(5, 56, 3, 4, protos_per_class=2, seed=1)
    b = fd.synthetic_odors(5, 56, 3, 4, protos_per_class=2, seed=1)
    Xtr, ytr, Xte, yte = a
    assert Xtr.shape == (15, 56) and Xte.shape == (20, 56)
    assert (Xtr >= 0).all() and np.array_equal(np.bincount(ytr), np.full(5, 3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@needs_data
def test_shuffled_only_selected_pairs(mb):
    g = mb.group_of()
    key = g[mb.pre] + ">" + g[mb.post]
    sh = mb.shuffled(seed=0, pairs=["PN>KC"])
    changed = mb.post != sh.post
    assert changed[key == "PN>KC"].mean() > 0.5
    assert not changed[key != "PN>KC"].any()
    ex = mb.shuffled(seed=0, exclude=["PN>KC"])
    assert not (mb.post != ex.post)[key == "PN>KC"].any()
    with pytest.raises(ValueError):
        mb.shuffled(pairs=["XX>YY"])


@needs_data
def test_shuffled_has_no_duplicate_or_self_edges(mb):
    sh = mb.shuffled(seed=0)
    key = pd.Series(sh.pre * sh.N + sh.post)
    assert not key.duplicated().any()
    assert not (sh.pre == sh.post).any()


@needs_data
@pytest.mark.skipif(bool(fd.data.missing("door")), reason="DoOR 데이터 없음")
def test_door_odors(mb):
    enc = fd.GlomerularEncoder(mb, device="cpu")
    d = fd.door_odors(enc.glomeruli, min_measured=20)
    assert d["X"].shape == (len(d["names"]), 56)
    assert (d["X"] >= 0).all() and (d["X"] <= 1).all()
    assert (d["X"][~d["measured"]] == 0).all()           # 측정 안 된 칸은 0
    assert (d["measured"].sum(1) >= 20).all()
    assert "arom" not in set(d["classes"]) and "terpenes" not in set(d["classes"])


def test_assoc_readout_k1_matches_dopamine_assoc():
    X, y = _toy()
    a = fd.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X, y)
    b = fd.AssocReadout(X.shape[1], 4, per_class=1, device="cpu").fit(X, y)
    assert np.allclose(_np(a.W), _np(b.W), atol=1e-5)


def test_assoc_readout_never_changes_other_classes():
    X, y = _toy()
    m = fd.AssocReadout(X.shape[1], 4, per_class=3, device="cpu")
    m.fit(X[y < 2], y[y < 2])
    before = _np(m.W)[:2 * 3].copy()
    m.fit(X[y >= 2], y[y >= 2])                         # 나중 클래스 학습
    assert np.array_equal(_np(m.W)[:2 * 3], before)     # 앞 클래스 출력은 그대로
    assert (_np(m.count).reshape(4, 3) > 0).all()       # 모든 원형이 채워짐
    assert m.accuracy(X, y) > 0.9


def test_train_linear_is_reproducible():
    X, y = _toy(n=100)
    r = [_np(fd.train_linear(X[:60], y[:60], X[60:], y[60:], epochs=5, device="cpu")["model"].weight.data)
         for _ in range(2)]
    assert np.array_equal(r[0], r[1])


def test_assoc_readout_uses_all_prototypes_in_one_batch():
    X, y = _toy(n=40)
    m = fd.AssocReadout(X.shape[1], 4, per_class=3, device="cpu").fit(X, y, batch=64)   # 한 묶음에 전부
    assert (_np(m.count).reshape(4, 3) > 0).all()
    assert _np(m.count).sum() == 40


# ─────────────── 역전파 ───────────────
@needs_data
def test_trainable_layer_gradients_flow(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=20, dt=0.5, gains={"PN>KC": 2.0},
                               input_mode="regular", trainable=True)
    assert layer.log_scale.size == layer.w_syn.size
    x = fd.Signal(np.full((4, 344), 80.0, np.float32), plastic=True, device=layer.device)
    layer(x).sum().retrograde()
    g = _np(layer.log_scale.retro)
    assert np.isfinite(g).all() and (g != 0).any()
    assert x.retro is not None and (_np(x.retro) != 0).any()       # 입력까지 기울기가 흐름


@needs_data
def test_trainable_subset_and_sign_kept(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "MBON", trainable=["KC>MBON"])
    assert set(layer.edge_key[_np(layer.train_pos)]) == {"KC>MBON"}
    layer.log_scale.data = B.to(np.random.default_rng(0).normal(0, 2, layer.log_scale.shape).astype(np.float32),
                                layer.device)
    w0, w = _np(layer.w_base), layer.weights()
    assert np.array_equal(np.sign(w), np.sign(w0))                 # 부호(흥분/억제)는 그대로
    with pytest.raises(ValueError):
        fd.ConnectomeLayer(mb, "PN", "KC", trainable=["XX>YY"])


@needs_data
def test_untrained_trainable_layer_matches_fixed(mb):
    kw = dict(t_ms=20, dt=0.5, gains={"PN>KC": 2.0}, input_mode="regular")
    x = fd.RateEncoder(784, 344)(np.random.default_rng(1).random((4, 784)).astype(np.float32))
    with fd.quiescent():
        a = _np(fd.ConnectomeLayer(mb, "PN", "KC", **kw)(x))
        b = _np(fd.ConnectomeLayer(mb, "PN", "KC", trainable=True, **kw)(x))
    assert np.allclose(a, b)


def test_surrogate_spike():
    x = fd.Signal(np.array([-1.0, -0.01, 0.01, 1.0]), plastic=True)
    y = fd.fire(x, 0.0, 10.0)
    assert y.numpy().tolist() == [0, 0, 1, 1]
    y.sum().retrograde()
    assert (x.retro > 0).all() and x.retro[1] > x.retro[0]      # 문턱 근처에서 기울기가 큼


# ─────────────── 데이터 경로 ───────────────
def test_data_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(fd.data, "CONFIG", tmp_path / "config.json")
    monkeypatch.setattr(fd.data, "DEFAULT_ROOT", tmp_path / "default")
    for env in ("FLYDNET_FLYWIRE", "FLYDNET_DATA", "FLYDNET_DOOR"):
        monkeypatch.delenv(env, raising=False)
    assert fd.data_dir("flywire") == tmp_path / "default" / "flywire"          # 4. 기본값
    fd.set_data_dir(flywire=tmp_path / "cfg")
    assert fd.data_dir("flywire") == (tmp_path / "cfg").resolve()             # 3. 설정 파일
    assert fd.data_dir("door") == tmp_path / "default" / "door"              #    다른 묶음은 그대로
    monkeypatch.setenv("FLYDNET_DATA", str(tmp_path / "legacy"))
    assert fd.data_dir("flywire") == tmp_path / "legacy"                      # 2. 예전 환경변수
    monkeypatch.setenv("FLYDNET_FLYWIRE", str(tmp_path / "env"))
    assert fd.data_dir("flywire") == tmp_path / "env"                         #    새 환경변수가 우선
    assert fd.data_dir("flywire", tmp_path / "arg") == tmp_path / "arg"       # 1. 직접 준 경로
    with pytest.raises(ValueError):
        fd.data_dir("nope")


def test_require_explains_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        fd.data.require("door", tmp_path)
    for name in fd.data.SOURCES["door"]:
        (tmp_path / name).write_text("x")
    with pytest.raises(ValueError, match="크기가 다름"):                     # 있지만 잘린·다른 파일
        fd.data.require("door", tmp_path)


# ─────────────── 저장 / 불러오기 ───────────────
@needs_data
def test_layer_save_load_roundtrip(mb, tmp_path):
    layer = fd.ConnectomeLayer(mb, "PN", ("KC", "MBON"), t_ms=20, dt=0.5, gains={"PN>KC": 2.0},
                               input_mode="regular", trainable=["KC>MBON"])
    layer.log_scale.data = B.to(np.random.default_rng(0).normal(0, 0.5, layer.log_scale.shape).astype(np.float32),
                                layer.device)                       # '학습된' 상태 흉내
    layer.set_gain("PN>KC", 2.5)                                    # 만든 뒤 바꾼 배율도 저장되는지
    p = layer.save(tmp_path / "layer")
    new = fd.ConnectomeLayer.load(p)
    x = fd.RateEncoder(784, 344)(np.random.default_rng(2).random((3, 784)).astype(np.float32))
    with fd.quiescent():
        assert np.array_equal(_np(layer(x)), _np(new(x)))
    assert np.array_equal(new.weights(), layer.weights())
    assert new.gains == {"PN>KC": 2.5} and new.circuit.N == mb.N
    pd.testing.assert_frame_equal(new.circuit.meta.astype(str), mb.meta.astype(str))
    cpu = fd.ConnectomeLayer.load(p, device="cpu")
    assert cpu.log_scale.device == "cpu"


@needs_data
def test_model_state_roundtrip(mb, tmp_path):
    make = lambda: fd.Pathway(fd.ConnectomeLayer(mb, "PN", "KC", t_ms=10, dt=0.5, trainable=True),
                              fd.Projection(2597, 3, seed=0))
    a = make()
    a[0].log_scale.data = B.to(np.random.default_rng(0).uniform(-1, 1, a[0].log_scale.shape).astype(np.float32),
                               a[0].device)
    p = a.save(tmp_path / "m")
    b = make()
    b.load(p)
    assert np.array_equal(a[0].weights(), b[0].weights())


@pytest.mark.parametrize("cls,kw", [(fd.AssocReadout, dict(per_class=3)),
                                    (fd.DopamineReadout, dict(mode="bidir", lr=0.1))])
def test_readout_save_load(cls, kw, tmp_path):
    X, y = _toy()
    m = cls(X.shape[1], 4, device="cpu", **kw).fit(X, y)
    p = m.save(tmp_path / "r")
    n = cls.load(p, device="cpu")
    assert np.array_equal(m.predict(X), n.predict(X))
    with pytest.raises(ValueError):
        (fd.DopamineReadout if cls is fd.AssocReadout else fd.AssocReadout).load(p)


# ─────────────── 그래디언트 체크포인팅 ───────────────
@needs_data
@pytest.mark.parametrize("mode", ["regular", "poisson"])
def test_checkpointing_gives_same_output_and_grads(mb, mode):
    def run(ce):
        layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=15, dt=0.5, gains={"PN>KC": 2.0}, input_mode=mode,
                                   trainable=True, checkpoint_every=ce)
        layer.log_scale.data = B.to(np.linspace(-0.3, 0.3, layer.log_scale.size, dtype=np.float32), layer.device)
        x = fd.Signal(np.full((4, 344), 90.0, np.float32), plastic=True, device=layer.device)
        out = layer(x, seed=7)
        (out * out).mean().retrograde()
        return _np(out.data), _np(layer.log_scale.retro), _np(x.retro)
    a, b = run(None), run(4)
    assert np.array_equal(a[0], b[0])                             # 순전파는 완전히 같음 (포아송 난수 포함)
    for ga, gb in zip(a[1:], b[1:]):                              # 기울기는 GPU 합산 순서 오차 안에서 같음
        assert np.allclose(ga, gb, rtol=1e-4, atol=1e-5)


# ─────────────── 작은 가짜 회로 (데이터 없이) ───────────────
def _tiny_circuit(n_in=5, n_out=12, n_edges=60, seed=0):
    rng = np.random.default_rng(seed)
    N = n_in + n_out
    key = rng.choice(N * N, n_edges, replace=False)
    pre, post = key // N, key % N
    keep = pre != post
    w = rng.integers(1, 6, keep.sum()) * rng.choice([-1, 1], keep.sum())
    return fd.Circuit(np.arange(N), {"IN": np.arange(n_in), "OUT": np.arange(n_in, N)},
                      pre[keep], post[keep], w.astype(np.float32))


def test_tiny_circuit_layer_trains_without_flywire_data():
    c = _tiny_circuit(n_edges=120)
    layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True,
                               device="cpu", gains={"IN>OUT": 30.0})
    out = layer(np.full((2, 5), 200.0, np.float32))
    out.sum().retrograde()
    assert out.shape == (2, 12) and np.isfinite(layer.log_scale.retro).all()


# ─────────────── 뉴런 매개변수, 연결 종류 공유, 시간 변화 입력 (시각계용) ───────────────
def test_with_sign_flips_only_selected_sender():
    c = _tiny_circuit()
    s = c.with_sign("IN", -1)
    m = np.isin(c.pre, c.groups["IN"])
    assert (s.weight[m] < 0).all() and np.array_equal(s.weight[~m], c.weight[~m])
    assert np.array_equal(np.abs(s.weight), np.abs(c.weight))


def test_pair_sharing_has_one_scale_per_edge_type():
    c = _tiny_circuit(n_edges=120)
    layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True,
                               share="pair", device="cpu", gains={"IN>OUT": 30.0})
    assert layer.log_scale.size == len(set(layer.edge_key))
    layer(np.full((2, 5), 200.0, np.float32)).sum().retrograde()
    assert np.isfinite(layer.log_scale.retro).all()
    layer.log_scale.data = np.arange(layer.log_scale.size, dtype=np.float32) * 0.1
    w = layer.weights() / _np(layer.w_base)
    for k, v in layer.scale_of().items():
        assert np.allclose(w[layer.edge_key == k], v)


def test_bias_makes_neurons_fire_without_input_and_trains():
    c = _tiny_circuit()
    x = np.zeros((2, 5), np.float32)
    quiet = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=50, dt=0.5, device="cpu")
    with fd.quiescent():
        assert _np(quiet(x)).sum() == 0
    layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=50, dt=0.5, device="cpu", bias={"OUT": 12.0},
                               t_mbr={"OUT": 10.0}, train_neurons=True, v_init="random")
    out = layer(x)
    assert (_np(out.data) > 0).all()
    out.sum().retrograde()
    assert np.abs(layer.bias.retro).sum() > 0 and np.abs(layer.log_t_mbr.retro).sum() > 0
    tab = layer.neuron_table()
    assert tab.loc["OUT", "bias_mV"] == 12.0 and abs(tab.loc["OUT", "t_mbr_ms"] - 10.0) < 1e-4
    assert tab.loc["IN", "bias_mV"] == 0.0 and abs(tab.loc["IN", "t_mbr_ms"] - 20.0) < 1e-4


def test_neutral_neuron_params_match_plain_layer():
    c = _tiny_circuit(n_edges=120)
    kw = dict(t_ms=30, dt=0.5, input_mode="regular", device="cpu", gains={"IN>OUT": 30.0})
    x = np.full((2, 5), 150.0, np.float32)
    with fd.quiescent():
        a = _np(fd.ConnectomeLayer(c, "IN", "OUT", **kw)(x))
        b = _np(fd.ConnectomeLayer(c, "IN", "OUT", bias=0.0, **kw)(x))
    assert np.allclose(a, b)


def test_time_varying_input_constant_matches_static():
    c = _tiny_circuit(n_edges=120)
    layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=30, dt=0.5, input_mode="regular", device="cpu",
                               gains={"IN>OUT": 30.0})
    x = (np.random.default_rng(0).random((3, 5)) * 200).astype(np.float32)
    with fd.quiescent():
        assert np.array_equal(_np(layer(x)), _np(layer(np.repeat(x[:, None], 7, 1))))
        off_late = np.repeat(x[:, None], 2, 1); off_late[:, 1] = 0          # 후반부 입력 끔 → 반응 줄어듦
        assert _np(layer(off_late)).sum() < _np(layer(x)).sum()


def test_count_from_ms_counts_only_late_window():
    c = _tiny_circuit(n_edges=120)
    kw = dict(t_ms=40, dt=0.5, input_mode="regular", device="cpu", gains={"IN>OUT": 30.0})
    x = np.full((1, 5), 200.0, np.float32)
    with fd.quiescent(), pytest.warns(UserWarning, match="겹침"):           # 입력 = 출력 그룹 (일부러)
        full = _np(fd.ConnectomeLayer(c, "IN", "IN", **kw)(x))
    with fd.quiescent(), pytest.warns(UserWarning, match="겹침"):
        late = _np(fd.ConnectomeLayer(c, "IN", "IN", count_from_ms=20, **kw)(x))
    assert np.allclose(late, full, rtol=0.2)                            # 일정 입력이면 발화율(Hz)은 비슷
    with pytest.raises(ValueError):
        fd.ConnectomeLayer(c, "IN", "OUT", count_from_ms=40, **kw)


def test_new_options_save_load_roundtrip(tmp_path):
    c = _tiny_circuit(n_edges=120)
    layer = fd.ConnectomeLayer(c, ["IN"], "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True,
                               share="pair", bias={"OUT": 3.0}, train_neurons=True, v_init="random",
                               count_from_ms=5, device="cpu", gains={"IN>OUT": 30.0})
    layer.log_scale.data = layer.log_scale.data + 0.3
    layer.bias.data = layer.bias.data + 1.0
    back = fd.ConnectomeLayer.load(layer.save(tmp_path / "l"), device="cpu")
    x = np.full((2, 5), 150.0, np.float32)
    with fd.quiescent():
        assert np.array_equal(_np(layer(x)), _np(back(x)))


def test_drifting_grating():
    xy = np.random.default_rng(0).normal(size=(30, 2)).astype(np.float32) * 5
    lum = fd.drifting_grating(xy, [0, 90], t_ms=100, frames=20, onset_ms=20, contrast=0.5)
    assert lum.shape == (2, 20, 30)
    assert (lum[:, :4] == 0).all() and np.abs(lum).max() <= 0.5 + 1e-6 and np.abs(lum[:, 4:]).max() > 0.3


@needs_data
def test_visual_circuit_and_direction_wiring():
    vc = fd.visual_circuit()
    assert {"T4a", "Mi1", "R1-6", "HSE"} <= set(vc.groups) and vc.pos is not None
    assert (vc.weight[np.isin(vc.pre, vc.groups["R1-6"])] < 0).all()
    xy = fd.column_map(vc)
    assert np.isfinite(xy[vc.groups["Mi1"]]).all() and np.isnan(xy[vc.groups["HSE"]]).all()
    real = fd.direction_offsets(vc, xy)
    with pytest.warns(UserWarning):
        sh = vc.shuffled(seed=0)
    rand = fd.direction_offsets(sh, xy)
    assert all(real[t][1] > 0.3 > rand[t][1] for t in real)       # 실제 배선에만 방향 구조


def test_normalized_inputs_sum_to_one():
    c = _tiny_circuit(n_edges=120).normalized()
    tot = np.bincount(c.post, weights=np.abs(c.weight), minlength=c.N)
    assert np.allclose(tot[tot > 0], 1.0)


def test_graded_neurons_respond_and_train():
    c = _tiny_circuit(n_edges=120).normalized()
    layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=30, dt=1.0, neuron="graded", params={"w_syn": 2.0},
                               trainable=True, share="pair", bias={"OUT": 0.2}, train_neurons=True, device="cpu")
    x = fd.Signal(np.random.default_rng(0).random((3, 5)).astype(np.float32), plastic=True)
    out = layer(x)
    assert out.shape == (3, 12) and (out.numpy() >= 0).all()
    out.sum().retrograde()
    assert np.isfinite(layer.log_scale.retro).all() and np.abs(layer.bias.retro).sum() > 0
    assert x.retro is not None and np.abs(x.retro).sum() > 0
    with fd.quiescent():                                     # 입력 뉴런 활동 = 입력값
        assert np.allclose(layer(x.numpy(), return_all=True).numpy()[:, _np(layer.in_idx)], x.numpy())


def test_graded_checkpoint_matches():
    c = _tiny_circuit(n_edges=120).normalized()
    kw = dict(t_ms=30, dt=1.0, neuron="graded", params={"w_syn": 2.0}, trainable=True, bias=0.2, device="cpu")
    a = fd.ConnectomeLayer(c, "IN", "OUT", **kw)
    b = fd.ConnectomeLayer(c, "IN", "OUT", checkpoint_every=7, **kw)
    x = np.random.default_rng(1).random((2, 5)).astype(np.float32)
    ya, yb = a(x), b(x)
    ya.sum().retrograde(); yb.sum().retrograde()
    assert np.allclose(ya.numpy(), yb.numpy()) and np.allclose(a.log_scale.retro, b.log_scale.retro, atol=1e-6)


@pytest.mark.parametrize("neuron", ["lif", "graded"])
def test_record_returns_traces(neuron):
    c = _tiny_circuit(n_edges=120)
    layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, neuron=neuron, input_mode="regular",
                               device="cpu", gains={"IN>OUT": 30.0}, bias=0.2 if neuron == "graded" else None)
    x = np.full((2, 5), 0.8 if neuron == "graded" else 200.0, np.float32)
    with fd.quiescent():
        out, tr = layer(x, record=[5, 6, 0])
        assert tr.shape == (2, 40, 3)
        assert np.allclose(_np(out), _np(layer(x)))


def test_subset_keeps_only_selected_groups():
    c = _tiny_circuit(n_edges=120)
    s = c.subset(["OUT"])
    assert s.N == 12 and list(s.groups) == ["OUT"]
    m = np.isin(c.pre, c.groups["OUT"]) & np.isin(c.post, c.groups["OUT"])
    assert s.n_edges == m.sum() and np.array_equal(np.sort(s.weight), np.sort(c.weight[m]))


def test_local_shuffle_stays_within_bins_and_merge_mixes_subtypes():
    rng = np.random.default_rng(0)
    N = 60
    groups = {"A": np.arange(20), "Ba": np.arange(20, 40), "Bb": np.arange(40, 60)}
    pre = rng.integers(0, 20, 400); post = rng.integers(20, 60, 400)
    key = np.unique(pre * N + post); pre, post = key // N, key % N
    c = fd.Circuit(np.arange(N), groups, pre, post, np.ones(len(pre), np.float32))
    xy = np.stack([np.arange(N) % 10, np.zeros(N)], 1).astype(np.float32)
    s = c.shuffled(seed=1, local=(xy, 5.0))
    assert np.array_equal(np.floor(xy[s.post, 0] / 5), np.floor(xy[c.post, 0] / 5))   # 칸은 그대로
    g = c.group_of()
    assert np.array_equal(g[s.post], g[c.post])                                      # merge 없으면 아형 유지
    m = c.shuffled(seed=1, local=(xy, 5.0), merge={"Ba": "B", "Bb": "B"})
    assert (g[m.post] != g[c.post]).any()                                            # merge면 아형 섞임
    assert np.array_equal(np.bincount(m.post, minlength=N), np.bincount(c.post, minlength=N))


def test_cli_status_and_unknown_command(capsys):
    from flydnet.__main__ import main
    assert main(["status"]) == 0 and "flydnet" in capsys.readouterr().out
    assert main(["bogus"]) == 1


# ─────────────── 데이터 버전 고정 ───────────────
def test_data_sources_are_pinned_with_checksums():
    for kind, files in fd.data.SOURCES.items():
        for name, (url, size, sha) in files.items():
            assert "/main/" not in url and "/master/" not in url, url     # 커밋 해시로 고정
            assert size > 0 and len(sha) == 64


def test_download_checks_content(tmp_path, monkeypatch):
    good = b"hello flydnet"
    import hashlib
    monkeypatch.setitem(fd.data.SOURCES, "door",
                        {"a.csv": ("http://x/a.csv", len(good), hashlib.sha256(good).hexdigest())})
    payload = {"data": b"tampered!!!!!"}                                   # 크기는 같고 내용만 다름
    monkeypatch.setattr(fd.data, "_fetch", lambda url, dest, quiet: Path(dest).write_bytes(payload["data"]))
    with pytest.raises(IOError):
        fd.download("door", tmp_path, quiet=True)
    assert not (tmp_path / "a.csv").exists() and not (tmp_path / "a.csv.part").exists()
    payload["data"] = good
    fd.download("door", tmp_path, quiet=True)
    assert fd.data.verify("door", tmp_path) == {"a.csv": "ok"}
    (tmp_path / "a.csv").write_bytes(b"old version!!")                     # 예전 버전 파일 → 다시 받음
    assert fd.data.verify("door", tmp_path) == {"a.csv": "sha256"}
    fd.download("door", tmp_path, quiet=True)
    assert (tmp_path / "a.csv").read_bytes() == good


@needs_data
def test_local_data_matches_pinned_version():
    assert set(fd.data.verify("flywire").values()) == {"ok"}


# ─────────────── KCExpansion (앞먹임 확장 층) ───────────────
def _pn_kc_circuit(n_pn=30, n_kc=200, per_kc=6, seed=0):
    rng = np.random.default_rng(seed)
    pre = np.concatenate([rng.choice(n_pn, per_kc, replace=False) for _ in range(n_kc)])
    post = np.repeat(np.arange(n_pn, n_pn + n_kc), per_kc)
    w = rng.integers(1, 10, len(pre)).astype(np.float32)
    return fd.Circuit(np.arange(n_pn + n_kc), {"PN": np.arange(n_pn), "KC": np.arange(n_pn, n_pn + n_kc)}, pre, post, w)


def test_kc_expansion_sparsity_and_weights():
    c = _pn_kc_circuit()
    kc = fd.KCExpansion(c, n_in=50, k_frac=0.1, device="cpu")
    assert kc.W.shape == (200, 30) and _np(kc.W).sum() == c.weight.sum()
    x = np.random.default_rng(0).random((7, 50)).astype(np.float32)
    out = _np(kc(x))
    assert out.shape == (7, 200) and ((out != 0).sum(1) <= 20).all()
    assert np.array_equal(out, _np(kc(x)))                                # 결정론적
    b = _np(fd.KCExpansion(c, n_in=50, k_frac=0.1, binary=True, device="cpu")(x))
    assert set(np.unique(b).tolist()) <= {0.0, 1.0} and (b.sum(1) == 20).all()


def test_kc_expansion_similar_inputs_share_codes():
    c = _pn_kc_circuit(n_pn=60, n_kc=1000)
    kc = fd.KCExpansion(c, n_in=100, k_frac=0.05, binary=True, device="cpu")
    g = np.random.default_rng(0)
    x = g.random((1, 100)).astype(np.float32)
    near, far = (x + 0.02 * g.standard_normal((1, 100))).astype(np.float32), g.random((1, 100)).astype(np.float32)
    ov = lambda a, b: float((_np(kc(a)) * _np(kc(b))).sum()) / kc.k
    assert ov(x, near) > 0.6 > 0.3 > ov(x, far)


# ─────────────── 구조물 · 작용 ───────────────
def _two_layer_circuit(n_a=6, n_b=9, n_c=4, seed=0):
    """A → B → C, 일부 억제, 같은 연결 중복 포함"""
    rng = np.random.default_rng(seed)
    pre = np.r_[rng.integers(0, n_a, 30), rng.integers(n_a, n_a + n_b, 20)]
    post = np.r_[rng.integers(n_a, n_a + n_b, 30), rng.integers(n_a + n_b, n_a + n_b + n_c, 20)]
    w = (rng.integers(1, 5, 50) * rng.choice([-1, 1], 50)).astype(np.float32)
    N = n_a + n_b + n_c
    return fd.Circuit(np.arange(N), {"A": np.arange(n_a), "B": np.arange(n_a, n_a + n_b),
                                     "C": np.arange(n_a + n_b, N)}, pre, post, w)


def test_neuropil_matches_circuit_wiring():
    c = _two_layer_circuit()
    layer = fd.Neuropil(c, "A", "B", train=None, init="counts", device="cpu")
    W = np.zeros((9, 6))
    m = np.isin(c.pre, c.groups["A"]) & np.isin(c.post, c.groups["B"])
    np.add.at(W, (c.post[m] - 6, c.pre[m]), c.weight[m])                      # 중복은 합쳐짐
    assert np.allclose(layer.dense(), W)
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    assert np.allclose(layer(x).numpy(), x @ W.T.astype(np.float32), atol=1e-5)
    assert layer.in_features == 6 and layer.out_features == 9 and layer.n_connections == (W != 0).sum()
    with pytest.raises(ValueError):
        fd.Neuropil(c, "C", "A", device="cpu")                                # 연결 없음


@pytest.mark.parametrize("train", ["pair", "edge", "free"])
def test_neuropil_training_modes(train):
    c = _two_layer_circuit()
    layer = fd.Neuropil(c, ["A", "B"], ["B", "C"], train=train, bias=True, device="cpu")
    mask = layer.dense() != 0
    sign = np.sign(layer.dense())
    rule = fd.Plasticity(layer.named_synapses(), rate=0.5)
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = layer(rng.standard_normal((8, 15)).astype(np.float32)) - 1
        rule.clear(); (y * y).mean().retrograde(); rule.step()
    after = layer.dense()
    assert ((after != 0) <= mask).all()                                       # 커넥톰에 없는 연결은 생기지 않음
    if train != "free":
        assert np.array_equal(np.sign(after), sign)                           # 부호 유지 (Dale)
    n_param = layer.n_synapses() - 13                                         # bias 제외
    assert n_param == {"pair": len(layer.pairs), "edge": layer.n_connections, "free": layer.n_connections}[train]
    if train == "pair":
        assert set(layer.scale_of()) == set(layer.pairs)


def test_neuropil_fan_in_keeps_signal_scale():
    c = _pn_kc_circuit(n_pn=50, n_kc=400, per_kc=8)
    layer = fd.Neuropil(c, "PN", "KC", train=None, device="cpu")
    y = layer(np.random.default_rng(0).standard_normal((2000, 50)).astype(np.float32)).numpy()
    assert 0.7 < y.std() < 1.3


def test_lateral_inhibition_and_axon_hillock():
    x = fd.Signal(np.array([[3.0, -1.0, 2.0, 0.5]]), plastic=True)
    y = fd.LateralInhibition(k=2)(x)
    assert np.array_equal(y.numpy(), [[3.0, 0.0, 2.0, 0.0]])
    y.sum().retrograde()
    assert np.array_equal(x.retro, [[1.0, 0.0, 1.0, 0.0]])                   # 기울기는 남은 것에만
    assert (fd.LateralInhibition(frac=0.5)(x).numpy() != 0).sum() == 2
    with pytest.raises(ValueError):
        fd.LateralInhibition()
    v = fd.Signal(np.array([-0.2, 0.0, 0.3]), plastic=True)
    s = fd.AxonHillock(threshold=0.1)(v)
    assert np.array_equal(s.numpy(), [0.0, 0.0, 1.0])
    s.sum().retrograde()
    assert (v.retro > 0).all()                                                # 대리 기울기


def test_mushroom_body_output_matches_assoc_readout(tmp_path):
    X, y = _toy(n=300, k=40, c=5)
    ref = fd.AssocReadout(40, 5, per_class=3, device="cpu").fit(X, y, batch=50)
    mbo = fd.MushroomBodyOutput(40, 5, per_class=3, device="cpu")
    perm = np.random.default_rng(0).permutation(len(X))                       # AssocReadout.fit과 같은 순서
    mbo.learn(X[perm], y[perm], batch=50)
    assert np.allclose(_np(mbo.prototypes), _np(ref.W)) and np.array_equal(_np(mbo.count), _np(ref.count))
    assert np.array_equal(mbo.predict(X), ref.predict(X))
    p = mbo.save(tmp_path / "mbo")
    new = fd.MushroomBodyOutput(40, 5, per_class=3, device="cpu")
    new.load(p)
    assert np.array_equal(new(X).numpy(), mbo(X).numpy())
    assert np.isinf(fd.MushroomBodyOutput(40, 5, device="cpu")(X).numpy()).all()   # 아직 아무것도 안 배움


def test_kenyon_code_matches_kc_expansion():
    from flydnet.ganglion.physiology import kenyon_code
    c = _pn_kc_circuit()
    kc = fd.KCExpansion(c, n_in=50, k_frac=0.1, device="cpu")
    x = np.random.default_rng(3).random((6, 50)).astype(np.float32)
    assert np.array_equal(_np(kc(x)), _np(kenyon_code(x, kc.W, kc.k, kc.proj)))


def test_anatomy_in_pathway_learns_toy_problem():
    c = _pn_kc_circuit(n_pn=20, n_kc=300, per_kc=6)
    model = fd.Pathway(fd.Projection(10, 20, seed=0), fd.Activation("relu"),
                       fd.Neuropil(c, "PN", "KC", train="edge", device="cpu"),
                       fd.LateralInhibition(frac=0.1), fd.Projection(300, 3, seed=1))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    rule = fd.Adaptive(model.named_synapses(), rate=1e-2)
    first = None
    for _ in range(150):
        loss = fd.surprise(model(X), y)
        first = first or float(loss.data)
        rule.clear(); loss.retrograde(); rule.step()
    assert float(loss.data) < 0.5 * first


def test_neuropil_load_state_checks_wiring():
    c = _pn_kc_circuit()
    a = fd.Neuropil(c, "PN", "KC", device="cpu")
    a.log_scale.data = np.random.default_rng(0).standard_normal(a.log_scale.shape).astype(np.float32)
    b = fd.Neuropil(c, "PN", "KC", device="cpu")
    b.load_state(a.state())                                                   # 같은 회로 → 됨
    x = np.random.default_rng(1).standard_normal((3, 30)).astype(np.float32)
    assert np.array_equal(a(x).numpy(), b(x).numpy())
    other = fd.Neuropil(_pn_kc_circuit(seed=1), "PN", "KC", device="cpu")      # 다른 배선
    with pytest.raises(RuntimeError, match="배선이 다름"):
        other.load_state(a.state())


def test_physiology_edge_cases():
    x = np.random.default_rng(0).standard_normal((2, 4))
    assert np.array_equal(fd.inhibit(x, k=10).numpy(), x)                     # k가 원소 수보다 크면 전부
    v = fd.Signal(np.random.default_rng(1).standard_normal(5), plastic=True)
    s = fd.fire(v)
    assert s.data.dtype == np.float64
    s.sum().retrograde()
    assert v.retro.dtype == np.float64
