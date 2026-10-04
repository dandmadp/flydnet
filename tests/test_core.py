"""flydnet 핵심 동작 테스트. FlyWire 데이터가 없으면 회로 관련 테스트는 건너뜀

    pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip("torch")  # flydnet의 torch 연동 기능 테스트 (torch 없으면 건너뜀)

import flydnet as fd

HAS_DATA = not fd.data.missing("flywire")
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"FlyWire 데이터 없음: {fd.data_dir('flywire')}")


@pytest.fixture(scope="module")
def mb():
    return fd.Circuit.from_flywire()


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
    enc = fd.torch.RateEncoder(784, 344, max_rate=100)
    r = enc(torch.rand(8, 28, 28))
    assert r.shape == (8, 344)
    assert r.min() >= 0 and torch.allclose(r.amax(1), torch.full((8,), 100.0))


def test_encoder_identity_requires_same_size():
    with pytest.raises(ValueError):
        fd.torch.RateEncoder(10, 20, projection=None)


# ─────────────── ConnectomeLayer ───────────────
@needs_data
def test_layer_shape_and_zero_input(mb):
    layer = fd.torch.ConnectomeLayer(mb, "PN", ("KC", "MBON"), t_ms=20)
    r = layer(torch.zeros(4, 344))
    assert r.shape == (4, 2597 + 48)
    assert (r == 0).all()


@needs_data
def test_layer_regular_is_deterministic(mb):
    layer = fd.torch.ConnectomeLayer(mb, "PN", "KC", t_ms=30, gains={"PN>KC": 2.0}, input_mode="regular")
    x = fd.torch.RateEncoder(784, 344)(torch.rand(4, 784))
    assert torch.equal(layer(x, seed=1), layer(x, seed=2))
    assert layer(x).sum() > 0


@needs_data
def test_layer_poisson_seed_reproducible(mb):
    layer = fd.torch.ConnectomeLayer(mb, "PN", "KC", t_ms=30, gains={"PN>KC": 2.0})
    x = torch.full((2, 344), 100.0)
    assert torch.equal(layer(x, seed=5), layer(x, seed=5))


# ─────────────── DopamineReadout ───────────────
def _toy(n=400, k=50, c=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    protos = (torch.rand(c, k, generator=g) < 0.2).float()
    y = torch.randint(0, c, (n,), generator=g)
    X = (protos[y] + 0.3 * torch.rand(n, k, generator=g)) * 50
    return X, y


@pytest.mark.parametrize("mode", ["bidir", "assoc"])
def test_readout_learns_toy_problem(mode):
    X, y = _toy()
    m = fd.torch.DopamineReadout(X.shape[1], 4, mode, lr=0.1, device="cpu").fit(X, y, epochs=3)
    assert m.accuracy(X, y) > 0.9


def test_bidir_weights_stay_nonnegative():
    X, y = _toy()
    m = fd.torch.DopamineReadout(X.shape[1], 4, "bidir", lr=1.0, device="cpu").fit(X, y, epochs=3)
    assert (m.W >= 0).all()


def test_assoc_is_order_independent():
    X, y = _toy()
    a = fd.torch.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X, y, seed=0)
    order = torch.argsort(y)                           # 클래스별로 몰아서 (연속 학습과 같은 순서)
    b = fd.torch.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X[order], y[order], seed=1)
    assert torch.allclose(a.W, b.W, atol=1e-5)


def test_invalid_mode():
    with pytest.raises(ValueError):
        fd.torch.DopamineReadout(10, 2, "nope")


# ─────────────── GlomerularEncoder / synthetic_odors ───────────────
@needs_data
def test_glomerular_encoder(mb):
    enc = fd.torch.GlomerularEncoder(mb)
    assert enc.n_glomeruli == 56
    assert enc.P.sum() == 139                          # 단일 사구체형 PN 139개, 각자 사구체 하나
    assert (enc.P.sum(1) <= 1).all()
    odor = torch.zeros(1, 56); odor[0, enc.glomeruli.index("DM1")] = 1.0
    r = enc(odor)[0]
    on = mb.meta.iloc[mb.groups["PN"]].cell_type.str.startswith("DM1_").values
    assert (r[torch.tensor(on)] == 100).all() and (r[torch.tensor(~on)] == 0).all()


@needs_data
def test_shuffled_keeps_meta(mb):
    pd.testing.assert_frame_equal(mb.shuffled(0).meta, mb.meta)


def test_synthetic_odors_shapes_and_determinism():
    a = fd.torch.synthetic_odors(5, 56, 3, 4, protos_per_class=2, seed=1)
    b = fd.torch.synthetic_odors(5, 56, 3, 4, protos_per_class=2, seed=1)
    Xtr, ytr, Xte, yte = a
    assert Xtr.shape == (15, 56) and Xte.shape == (20, 56)
    assert (Xtr >= 0).all() and torch.equal(ytr.bincount(), torch.full((5,), 3))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


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
    enc = fd.torch.GlomerularEncoder(mb)
    d = fd.torch.door_odors(enc.glomeruli, min_measured=20)
    assert d["X"].shape == (len(d["names"]), 56)
    assert (d["X"] >= 0).all() and (d["X"] <= 1).all()
    assert (d["X"][~d["measured"]] == 0).all()           # 측정 안 된 칸은 0
    assert (d["measured"].sum(1) >= 20).all()
    assert "arom" not in set(d["classes"]) and "terpenes" not in set(d["classes"])


def test_assoc_readout_k1_matches_dopamine_assoc():
    X, y = _toy()
    a = fd.torch.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X, y)
    b = fd.torch.AssocReadout(X.shape[1], 4, per_class=1, device="cpu").fit(X, y)
    assert torch.allclose(a.W, b.W, atol=1e-5)


def test_assoc_readout_never_changes_other_classes():
    X, y = _toy()
    m = fd.torch.AssocReadout(X.shape[1], 4, per_class=3, device="cpu")
    m.fit(X[y < 2], y[y < 2])
    before = m.W[:2 * 3].clone()
    m.fit(X[y >= 2], y[y >= 2])                         # 나중 클래스 학습
    assert torch.equal(m.W[:2 * 3], before)             # 앞 클래스 출력은 그대로
    assert (m.count.view(4, 3) > 0).all()               # 모든 원형이 채워짐
    assert m.accuracy(X, y) > 0.9


def test_train_linear_is_reproducible():
    X, y = _toy(n=100)
    r = [fd.torch.train_linear(X[:60], y[:60], X[60:], y[60:], epochs=5, device="cpu")["model"].weight for _ in range(2)]
    assert torch.equal(r[0], r[1])


def test_assoc_readout_uses_all_prototypes_in_one_batch():
    X, y = _toy(n=40)
    m = fd.torch.AssocReadout(X.shape[1], 4, per_class=3, device="cpu").fit(X, y, batch=64)   # 한 묶음에 전부
    assert (m.count.view(4, 3) > 0).all()
    assert m.count.sum() == 40


# ─────────────── 역전파 ───────────────
@needs_data
def test_trainable_layer_gradients_flow(mb):
    layer = fd.torch.ConnectomeLayer(mb, "PN", "KC", t_ms=20, dt=0.5, gains={"PN>KC": 2.0},
                               input_mode="regular", trainable=True)
    assert layer.log_scale.numel() == layer.w_syn.numel()
    x = torch.full((4, 344), 80.0, requires_grad=True)
    layer(x).sum().backward()
    g = layer.log_scale.grad
    assert torch.isfinite(g).all() and (g != 0).any()
    assert x.grad is not None and (x.grad != 0).any()       # 입력까지 기울기가 흐름


@needs_data
def test_trainable_subset_and_sign_kept(mb):
    layer = fd.torch.ConnectomeLayer(mb, "PN", "MBON", trainable=["KC>MBON"])
    assert set(layer.edge_key[layer.train_pos.cpu().numpy()]) == {"KC>MBON"}
    with torch.no_grad():
        layer.log_scale.normal_(0, 2)
    w0, w = layer.w_base, layer.weights()
    assert torch.equal(torch.sign(w), torch.sign(w0))       # 부호(흥분/억제)는 그대로
    with pytest.raises(ValueError):
        fd.torch.ConnectomeLayer(mb, "PN", "KC", trainable=["XX>YY"])


@needs_data
def test_untrained_trainable_layer_matches_fixed(mb):
    kw = dict(t_ms=20, dt=0.5, gains={"PN>KC": 2.0}, input_mode="regular")
    x = fd.torch.RateEncoder(784, 344)(torch.rand(4, 784))
    with torch.no_grad():
        a = fd.torch.ConnectomeLayer(mb, "PN", "KC", **kw)(x)
        b = fd.torch.ConnectomeLayer(mb, "PN", "KC", trainable=True, **kw)(x)
    assert torch.allclose(a, b)


def test_surrogate_spike():
    x = torch.tensor([-1.0, -0.01, 0.01, 1.0], requires_grad=True)
    y = fd.torch.layers.SpikeFn.apply(x, 10.0)
    assert y.tolist() == [0, 0, 1, 1]
    y.sum().backward()
    assert (x.grad > 0).all() and x.grad[1] > x.grad[0]      # 문턱 근처에서 기울기가 큼


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
    layer = fd.torch.ConnectomeLayer(mb, "PN", ("KC", "MBON"), t_ms=20, dt=0.5, gains={"PN>KC": 2.0},
                               input_mode="regular", trainable=["KC>MBON"])
    with torch.no_grad():
        layer.log_scale.normal_(0, 0.5)                     # '학습된' 상태 흉내
    layer.set_gain("PN>KC", 2.5)                            # 만든 뒤 바꾼 배율도 저장되는지
    layer.save(tmp_path / "layer.pt")
    new = fd.torch.ConnectomeLayer.load(tmp_path / "layer.pt")
    x = fd.torch.RateEncoder(784, 344)(torch.rand(3, 784))
    with torch.no_grad():
        assert torch.equal(layer(x), new(x))
    assert torch.equal(new.weights(), layer.weights())
    assert new.gains == {"PN>KC": 2.5} and new.circuit.N == mb.N
    pd.testing.assert_frame_equal(new.circuit.meta.astype(str), mb.meta.astype(str))
    cpu = fd.torch.ConnectomeLayer.load(tmp_path / "layer.pt", device="cpu")
    assert cpu.log_scale.device.type == "cpu"


@needs_data
def test_model_state_dict_roundtrip(mb, tmp_path):
    make = lambda: torch.nn.Sequential(fd.torch.ConnectomeLayer(mb, "PN", "KC", t_ms=10, dt=0.5, trainable=True),
                                       torch.nn.Linear(2597, 3))
    a = make()
    with torch.no_grad():
        a[0].log_scale.uniform_(-1, 1)
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b = make()
    b.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    assert torch.equal(a[0].weights(), b[0].weights())


@pytest.mark.parametrize("cls,kw", [(fd.torch.AssocReadout, dict(per_class=3)),
                                    (fd.torch.DopamineReadout, dict(mode="bidir", lr=0.1))])
def test_readout_save_load(cls, kw, tmp_path):
    X, y = _toy()
    m = cls(X.shape[1], 4, device="cpu", **kw).fit(X, y)
    m.save(tmp_path / "r.pt")
    n = cls.load(tmp_path / "r.pt", device="cpu")
    assert torch.equal(m.predict(X), n.predict(X))
    with pytest.raises(ValueError):
        (fd.torch.DopamineReadout if cls is fd.torch.AssocReadout else fd.torch.AssocReadout).load(tmp_path / "r.pt")


# ─────────────── 그래디언트 체크포인팅 ───────────────
@needs_data
@pytest.mark.parametrize("mode", ["regular", "poisson"])
def test_checkpointing_gives_same_output_and_grads(mb, mode):
    def run(ce):
        layer = fd.torch.ConnectomeLayer(mb, "PN", "KC", t_ms=15, dt=0.5, gains={"PN>KC": 2.0}, input_mode=mode,
                                   trainable=True, checkpoint_every=ce)
        with torch.no_grad():
            layer.log_scale.copy_(torch.linspace(-0.3, 0.3, layer.log_scale.numel()))
        x = torch.full((4, 344), 90.0, requires_grad=True)
        out = layer(x, seed=7)
        out.pow(2).mean().backward()
        return out.detach().cpu(), layer.log_scale.grad.cpu(), x.grad.cpu()
    a, b = run(None), run(4)
    assert torch.equal(a[0], b[0])                            # 순전파는 완전히 같음 (포아송 난수 포함)
    for ga, gb in zip(a[1:], b[1:]):                          # 기울기는 GPU 합산 순서 오차 안에서 같음
        assert torch.allclose(ga, gb, rtol=1e-4, atol=1e-5)


# ─────────────── 연결 단위 역전파 (데이터 없이 작은 가짜 회로) ───────────────
def _tiny_circuit(n_in=5, n_out=12, n_edges=60, seed=0):
    rng = np.random.default_rng(seed)
    N = n_in + n_out
    key = rng.choice(N * N, n_edges, replace=False)
    pre, post = key // N, key % N
    keep = pre != post
    w = rng.integers(1, 6, keep.sum()) * rng.choice([-1, 1], keep.sum())
    return fd.Circuit(np.arange(N), {"IN": np.arange(n_in), "OUT": np.arange(n_in, N)},
                      pre[keep], post[keep], w.astype(np.float32))


def test_sparse_propagate_matches_dense_reference():
    from flydnet.torch.layers import SparsePropagate
    c = _tiny_circuit()
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", trainable=True, device="cpu")
    N = c.N
    v = (layer.w_base * torch.linspace(0.5, 2, layer.w_base.numel())).double().requires_grad_(True)
    s = torch.rand(N, 3, dtype=torch.double, requires_grad=True)
    args = (layer.crow, layer.col, layer.crow_t, layer.col_t, layer.perm_t)
    assert torch.autograd.gradcheck(lambda v, s: SparsePropagate.apply(v, s, *args), (v, s))
    dense = torch.zeros(N, N, dtype=torch.double).index_put((layer.w_idx[0], layer.w_idx[1]), v)
    assert torch.allclose(SparsePropagate.apply(v, s, *args), dense @ s)


def test_tiny_circuit_layer_trains_without_flywire_data():
    c = _tiny_circuit(n_edges=120)
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True,
                               device="cpu", gains={"IN>OUT": 30.0})
    x = torch.full((2, 5), 200.0)
    out = layer(x)
    out.sum().backward()
    assert out.shape == (2, 12) and torch.isfinite(layer.log_scale.grad).all()


# ─────────────── 뉴런 매개변수, 연결 종류 공유, 시간 변화 입력 (시각계용) ───────────────
def test_with_sign_flips_only_selected_sender():
    c = _tiny_circuit()
    s = c.with_sign("IN", -1)
    m = np.isin(c.pre, c.groups["IN"])
    assert (s.weight[m] < 0).all() and np.array_equal(s.weight[~m], c.weight[~m])
    assert np.array_equal(np.abs(s.weight), np.abs(c.weight))


def test_pair_sharing_has_one_scale_per_edge_type():
    c = _tiny_circuit(n_edges=120)
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True,
                               share="pair", device="cpu", gains={"IN>OUT": 30.0})
    assert len(layer.log_scale) == len(set(layer.edge_key))
    layer(torch.full((2, 5), 200.0)).sum().backward()
    assert torch.isfinite(layer.log_scale.grad).all()
    with torch.no_grad():
        layer.log_scale[:] = torch.arange(len(layer.log_scale)).float() * 0.1
    w = layer.weights() / layer.w_base
    for k, v in layer.scale_of().items():
        assert torch.allclose(w[torch.tensor(layer.edge_key == k)], torch.tensor(v))


def test_bias_makes_neurons_fire_without_input_and_trains():
    c = _tiny_circuit()
    x = torch.zeros(2, 5)
    quiet = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=50, dt=0.5, device="cpu")
    assert quiet(x).sum() == 0
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=50, dt=0.5, device="cpu", bias={"OUT": 12.0},
                               t_mbr={"OUT": 10.0}, train_neurons=True, v_init="random")
    out = layer(x)
    assert (out > 0).all()
    out.sum().backward()
    assert layer.bias.grad.abs().sum() > 0 and layer.log_t_mbr.grad.abs().sum() > 0
    tab = layer.neuron_table()
    assert tab.loc["OUT", "bias_mV"] == 12.0 and abs(tab.loc["OUT", "t_mbr_ms"] - 10.0) < 1e-4
    assert tab.loc["IN", "bias_mV"] == 0.0 and abs(tab.loc["IN", "t_mbr_ms"] - 20.0) < 1e-4


def test_neutral_neuron_params_match_plain_layer():
    c = _tiny_circuit(n_edges=120)
    kw = dict(t_ms=30, dt=0.5, input_mode="regular", device="cpu", gains={"IN>OUT": 30.0})
    x = torch.full((2, 5), 150.0)
    a = fd.torch.ConnectomeLayer(c, "IN", "OUT", **kw)(x)
    b = fd.torch.ConnectomeLayer(c, "IN", "OUT", bias=0.0, **kw)(x)
    assert torch.allclose(a, b)


def test_time_varying_input_constant_matches_static():
    c = _tiny_circuit(n_edges=120)
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=30, dt=0.5, input_mode="regular", device="cpu",
                               gains={"IN>OUT": 30.0})
    x = torch.rand(3, 5) * 200
    assert torch.equal(layer(x), layer(x[:, None].expand(-1, 7, -1)))
    off_late = x[:, None].repeat(1, 2, 1); off_late[:, 1] = 0          # 후반부 입력 끔 → 반응 줄어듦
    assert layer(off_late).sum() < layer(x).sum()


def test_count_from_ms_counts_only_late_window():
    c = _tiny_circuit(n_edges=120)
    kw = dict(t_ms=40, dt=0.5, input_mode="regular", device="cpu", gains={"IN>OUT": 30.0})
    x = torch.full((1, 5), 200.0)
    full = fd.torch.ConnectomeLayer(c, "IN", "IN", **kw)(x)
    late = fd.torch.ConnectomeLayer(c, "IN", "IN", count_from_ms=20, **kw)(x)
    assert torch.allclose(late, full, rtol=0.2)                         # 일정 입력이면 발화율(Hz)은 비슷
    with pytest.raises(ValueError):
        fd.torch.ConnectomeLayer(c, "IN", "OUT", count_from_ms=40, **kw)


def test_new_options_save_load_roundtrip(tmp_path):
    c = _tiny_circuit(n_edges=120)
    layer = fd.torch.ConnectomeLayer(c, ["IN"], "OUT", t_ms=20, dt=0.5, input_mode="regular", trainable=True,
                               share="pair", bias={"OUT": 3.0}, train_neurons=True, v_init="random",
                               count_from_ms=5, device="cpu", gains={"IN>OUT": 30.0})
    with torch.no_grad():
        layer.log_scale.add_(0.3); layer.bias.add_(1.0)
    layer.save(tmp_path / "l.pt")
    back = fd.torch.ConnectomeLayer.load(tmp_path / "l.pt", device="cpu")
    x = torch.full((2, 5), 150.0)
    assert torch.equal(layer(x), back(x))


def test_drifting_grating():
    xy = np.random.default_rng(0).normal(size=(30, 2)).astype(np.float32) * 5
    lum = fd.torch.drifting_grating(xy, [0, 90], t_ms=100, frames=20, onset_ms=20, contrast=0.5)
    assert lum.shape == (2, 20, 30)
    assert (lum[:, :4] == 0).all() and lum.abs().max() <= 0.5 + 1e-6 and lum[:, 4:].abs().max() > 0.3


@needs_data
def test_visual_circuit_and_direction_wiring():
    vc = fd.visual_circuit()
    assert {"T4a", "Mi1", "R1-6", "HSE"} <= set(vc.groups) and vc.pos is not None
    assert (vc.weight[np.isin(vc.pre, vc.groups["R1-6"])] < 0).all()
    xy = fd.torch.column_map(vc)
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
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=30, dt=1.0, neuron="graded", params={"w_syn": 2.0},
                               trainable=True, share="pair", bias={"OUT": 0.2}, train_neurons=True, device="cpu")
    x = torch.rand(3, 5, requires_grad=True)
    out = layer(x)
    assert out.shape == (3, 12) and (out >= 0).all()
    out.sum().backward()
    assert torch.isfinite(layer.log_scale.grad).all() and layer.bias.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0
    with torch.no_grad():                                    # 입력 뉴런 활동 = 입력값
        assert torch.allclose(layer(x, return_all=True)[:, layer.in_idx], x)


def test_graded_checkpoint_matches():
    c = _tiny_circuit(n_edges=120).normalized()
    kw = dict(t_ms=30, dt=1.0, neuron="graded", params={"w_syn": 2.0}, trainable=True, bias=0.2, device="cpu")
    a = fd.torch.ConnectomeLayer(c, "IN", "OUT", **kw)
    b = fd.torch.ConnectomeLayer(c, "IN", "OUT", checkpoint_every=7, **kw)
    x = torch.rand(2, 5)
    a(x).sum().backward(); b(x).sum().backward()
    assert torch.allclose(a(x), b(x)) and torch.allclose(a.log_scale.grad, b.log_scale.grad, atol=1e-6)


@pytest.mark.parametrize("neuron", ["lif", "graded"])
def test_record_returns_traces(neuron):
    c = _tiny_circuit(n_edges=120)
    layer = fd.torch.ConnectomeLayer(c, "IN", "OUT", t_ms=20, dt=0.5, neuron=neuron, input_mode="regular",
                               device="cpu", gains={"IN>OUT": 30.0}, bias=0.2 if neuron == "graded" else None)
    with torch.no_grad():
        out, tr = layer(torch.full((2, 5), 0.8 if neuron == "graded" else 200.0), record=[5, 6, 0])
    assert tr.shape == (2, 40, 3)
    assert torch.allclose(out, layer(torch.full((2, 5), 0.8 if neuron == "graded" else 200.0)))


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
    kc = fd.torch.KCExpansion(c, n_in=50, k_frac=0.1, device="cpu")
    assert kc.W.shape == (200, 30) and kc.W.sum() == c.weight.sum()
    x = torch.rand(7, 50)
    out = kc(x)
    assert out.shape == (7, 200) and ((out != 0).sum(1) <= 20).all()
    assert torch.equal(out, kc(x))                                       # 결정론적
    b = fd.torch.KCExpansion(c, n_in=50, k_frac=0.1, binary=True, device="cpu")(x)
    assert set(b.unique().tolist()) <= {0.0, 1.0} and (b.sum(1) == 20).all()


def test_kc_expansion_similar_inputs_share_codes():
    c = _pn_kc_circuit(n_pn=60, n_kc=1000)
    kc = fd.torch.KCExpansion(c, n_in=100, k_frac=0.05, binary=True, device="cpu")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 100, generator=g)
    near, far = x + 0.02 * torch.randn(1, 100, generator=g), torch.rand(1, 100, generator=g)
    ov = lambda a, b: (kc(a) * kc(b)).sum().item() / kc.k
    assert ov(x, near) > 0.6 > 0.3 > ov(x, far)


# ─────────────── anatomy (구조물) · physiology (작용) ───────────────
import torch.nn as nn
import flydnet.torch.physiology as P
from flydnet.torch.anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput


def _two_layer_circuit(n_a=6, n_b=9, n_c=4, seed=0):
    """A → B → C, 일부 억제, 같은 연결 중복 포함"""
    rng = np.random.default_rng(seed)
    pre = np.r_[rng.integers(0, n_a, 30), rng.integers(n_a, n_a + n_b, 20)]
    post = np.r_[rng.integers(n_a, n_a + n_b, 30), rng.integers(n_a + n_b, n_a + n_b + n_c, 20)]
    w = (rng.integers(1, 5, 50) * rng.choice([-1, 1], 50)).astype(np.float32)
    N = n_a + n_b + n_c
    return fd.Circuit(np.arange(N), {"A": np.arange(n_a), "B": np.arange(n_a, n_a + n_b),
                                     "C": np.arange(n_a + n_b, N)}, pre, post, w)


def test_transmit_rectangular_matches_dense_and_gradcheck():
    rng = np.random.default_rng(1)
    key = rng.choice(7 * 5, 15, replace=False)
    post, pre = key // 5, key % 5
    w, order = P.wiring(post, pre, 7, 5)
    v = torch.randn(15, dtype=torch.double)[torch.tensor(order)].requires_grad_(True)
    x = torch.randn(3, 2, 5, dtype=torch.double, requires_grad=True)          # 앞 차원 여러 개
    dense = torch.zeros(7, 5, dtype=torch.double).index_put((torch.tensor(post[order]), torch.tensor(pre[order])), v)
    assert P.transmit(x, v, w).shape == (3, 2, 7)
    assert torch.allclose(P.transmit(x, v, w), x @ dense.T)
    assert torch.autograd.gradcheck(lambda v, x: P.transmit(x, v, w), (v, x))
    with pytest.raises(ValueError):
        P.wiring([0, 0], [1, 1], 7, 5)                                        # 중복 연결
    with pytest.raises(ValueError):
        P.transmit(torch.randn(2, 4, dtype=torch.double), v, w)               # 입력 크기 다름


def test_neuropil_matches_circuit_wiring():
    c = _two_layer_circuit()
    layer = Neuropil(c, "A", "B", train=None, init="counts", device="cpu")
    W = np.zeros((9, 6))
    m = np.isin(c.pre, c.groups["A"]) & np.isin(c.post, c.groups["B"])
    np.add.at(W, (c.post[m] - 6, c.pre[m]), c.weight[m])                      # 중복은 합쳐짐
    assert np.allclose(layer.dense().numpy(), W)
    x = torch.randn(4, 6)
    assert torch.allclose(layer(x), x @ torch.tensor(W, dtype=torch.float32).T, atol=1e-5)
    assert layer.in_features == 6 and layer.out_features == 9 and layer.n_synapses == (W != 0).sum()
    with pytest.raises(ValueError):
        Neuropil(c, "C", "A", device="cpu")                                   # 연결 없음


@pytest.mark.parametrize("train", ["pair", "edge", "free"])
def test_neuropil_training_modes(train):
    c = _two_layer_circuit()
    layer = Neuropil(c, ["A", "B"], ["B", "C"], train=train, bias=True, device="cpu")
    mask = layer.dense() != 0
    sign = torch.sign(layer.dense())
    opt = torch.optim.SGD(layer.parameters(), lr=0.5)
    for _ in range(5):
        loss = (layer(torch.randn(8, 15)) - 1).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = layer.dense()
    assert ((after != 0) <= mask).all()                                       # 커넥톰에 없는 연결은 생기지 않음
    if train != "free":
        assert torch.equal(torch.sign(after), sign)                           # 부호 유지 (Dale)
    n_param = sum(p.numel() for p in layer.parameters()) - 13                 # bias 제외
    assert n_param == {"pair": len(layer.pairs), "edge": layer.n_synapses, "free": layer.n_synapses}[train]
    if train == "pair":
        assert set(layer.scale_of()) == set(layer.pairs)


def test_neuropil_fan_in_keeps_signal_scale():
    c = _pn_kc_circuit(n_pn=50, n_kc=400, per_kc=8)
    layer = Neuropil(c, "PN", "KC", train=None, device="cpu")
    y = layer(torch.randn(2000, 50))
    assert 0.7 < y.std().item() < 1.3


def test_lateral_inhibition_and_axon_hillock():
    x = torch.tensor([[3.0, -1.0, 2.0, 0.5]], requires_grad=True)
    y = LateralInhibition(k=2)(x)
    assert torch.equal(y, torch.tensor([[3.0, 0.0, 2.0, 0.0]]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))           # 기울기는 남은 것에만
    assert LateralInhibition(frac=0.5)(x).ne(0).sum() == 2
    with pytest.raises(ValueError):
        LateralInhibition()
    v = torch.tensor([-0.2, 0.0, 0.3], requires_grad=True)
    s = AxonHillock(threshold=0.1)(v)
    assert torch.equal(s, torch.tensor([0.0, 0.0, 1.0]))
    s.sum().backward()
    assert (v.grad > 0).all()                                                 # 대리 기울기


def test_mushroom_body_output_matches_assoc_readout(tmp_path):
    X, y = _toy(n=300, k=40, c=5)
    ref = fd.torch.AssocReadout(40, 5, per_class=3, device="cpu").fit(X, y, batch=50)
    mbo = MushroomBodyOutput(40, 5, per_class=3)
    g = torch.Generator().manual_seed(0)                                      # AssocReadout.fit과 같은 순서
    perm = torch.randperm(len(X), generator=g)
    mbo.learn(X[perm], y[perm], batch=50)
    assert torch.allclose(mbo.prototypes, ref.W) and torch.equal(mbo.count, ref.count)
    assert torch.equal(mbo.predict(X), ref.predict(X).cpu())
    torch.save(mbo.state_dict(), tmp_path / "mbo.pt")
    new = MushroomBodyOutput(40, 5, per_class=3)
    new.load_state_dict(torch.load(tmp_path / "mbo.pt", weights_only=True))
    assert torch.equal(new(X), mbo(X)) and new.learned_classes == list(range(5))
    assert torch.isinf(MushroomBodyOutput(40, 5)(X)).all()                    # 아직 아무것도 안 배움


def test_kenyon_code_matches_kc_expansion():
    c = _pn_kc_circuit()
    kc = fd.torch.KCExpansion(c, n_in=50, k_frac=0.1, device="cpu")
    x = torch.rand(6, 50)
    assert torch.equal(kc(x), P.kenyon_code(x, kc.W, kc.k, kc.proj))


def test_anatomy_in_sequential_learns_toy_problem():
    c = _pn_kc_circuit(n_pn=20, n_kc=300, per_kc=6)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), Neuropil(c, "PN", "KC", train="edge", device="cpu"),
                          LateralInhibition(frac=0.1), nn.Linear(300, 3))
    X = torch.randn(300, 10); y = (X[:, 0] > 0).long() + (X[:, 1] > 0).long()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        loss = nn.functional.cross_entropy(model(X), y)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first


def test_neuropil_state_dict_checks_wiring():
    c = _pn_kc_circuit()
    a = Neuropil(c, "PN", "KC", device="cpu")
    with torch.no_grad():
        a.log_scale.normal_()
    b = Neuropil(c, "PN", "KC", device="cpu")
    b.load_state_dict(a.state_dict())                                         # 같은 회로 → 됨
    x = torch.randn(3, 30)
    assert torch.equal(a(x), b(x))
    other = Neuropil(_pn_kc_circuit(seed=1), "PN", "KC", device="cpu")         # 다른 배선
    with pytest.raises(RuntimeError, match="배선이 다름"):
        other.load_state_dict(a.state_dict())


def test_physiology_edge_cases():
    x = torch.randn(2, 4)
    assert torch.equal(P.inhibit(x, k=10), x)                                 # k가 원소 수보다 크면 전부
    v = torch.randn(5, dtype=torch.double, requires_grad=True)
    s = P.fire(v)
    assert s.dtype == torch.double
    s.sum().backward()
    assert v.grad.dtype == torch.double
