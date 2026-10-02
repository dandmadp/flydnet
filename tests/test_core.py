"""flydnet 핵심 동작 테스트. FlyWire 데이터가 없으면 회로 관련 테스트는 건너뜀

    pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import flydnet as fd

HAS_DATA = all((fd.DEFAULT_DATA / f).exists() for f in
               ("Completeness_783.csv", "Connectivity_783.parquet", "flywire_annotations.tsv"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"FlyWire 데이터 없음: {fd.DEFAULT_DATA}")


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
    enc = fd.RateEncoder(784, 344, max_rate=100)
    r = enc(torch.rand(8, 28, 28))
    assert r.shape == (8, 344)
    assert r.min() >= 0 and torch.allclose(r.amax(1), torch.full((8,), 100.0))


def test_encoder_identity_requires_same_size():
    with pytest.raises(ValueError):
        fd.RateEncoder(10, 20, projection=None)


# ─────────────── ConnectomeLayer ───────────────
@needs_data
def test_layer_shape_and_zero_input(mb):
    layer = fd.ConnectomeLayer(mb, "PN", ("KC", "MBON"), t_ms=20)
    r = layer(torch.zeros(4, 344))
    assert r.shape == (4, 2597 + 48)
    assert (r == 0).all()


@needs_data
def test_layer_regular_is_deterministic(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=30, gains={"PN>KC": 2.0}, input_mode="regular")
    x = fd.RateEncoder(784, 344)(torch.rand(4, 784))
    assert torch.equal(layer(x, seed=1), layer(x, seed=2))
    assert layer(x).sum() > 0


@needs_data
def test_layer_poisson_seed_reproducible(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=30, gains={"PN>KC": 2.0})
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
    m = fd.DopamineReadout(X.shape[1], 4, mode, lr=0.1, device="cpu").fit(X, y, epochs=3)
    assert m.accuracy(X, y) > 0.9


def test_bidir_weights_stay_nonnegative():
    X, y = _toy()
    m = fd.DopamineReadout(X.shape[1], 4, "bidir", lr=1.0, device="cpu").fit(X, y, epochs=3)
    assert (m.W >= 0).all()


def test_assoc_is_order_independent():
    X, y = _toy()
    a = fd.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X, y, seed=0)
    order = torch.argsort(y)                           # 클래스별로 몰아서 (연속 학습과 같은 순서)
    b = fd.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X[order], y[order], seed=1)
    assert torch.allclose(a.W, b.W, atol=1e-5)


def test_invalid_mode():
    with pytest.raises(ValueError):
        fd.DopamineReadout(10, 2, "nope")


# ─────────────── GlomerularEncoder / synthetic_odors ───────────────
@needs_data
def test_glomerular_encoder(mb):
    enc = fd.GlomerularEncoder(mb)
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
    a = fd.synthetic_odors(5, 56, 3, 4, protos_per_class=2, seed=1)
    b = fd.synthetic_odors(5, 56, 3, 4, protos_per_class=2, seed=1)
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


DOOR = Path(__file__).resolve().parents[1] / "data" / "door"


@needs_data
@pytest.mark.skipif(not (DOOR / "door_response_matrix.csv").exists(), reason="DoOR 데이터 없음")
def test_door_odors(mb):
    enc = fd.GlomerularEncoder(mb)
    d = fd.door_odors(enc.glomeruli, DOOR, min_measured=20)
    assert d["X"].shape == (len(d["names"]), 56)
    assert (d["X"] >= 0).all() and (d["X"] <= 1).all()
    assert (d["X"][~d["measured"]] == 0).all()           # 측정 안 된 칸은 0
    assert (d["measured"].sum(1) >= 20).all()
    assert "arom" not in set(d["classes"]) and "terpenes" not in set(d["classes"])


def test_assoc_readout_k1_matches_dopamine_assoc():
    X, y = _toy()
    a = fd.DopamineReadout(X.shape[1], 4, "assoc", device="cpu").fit(X, y)
    b = fd.AssocReadout(X.shape[1], 4, per_class=1, device="cpu").fit(X, y)
    assert torch.allclose(a.W, b.W, atol=1e-5)


def test_assoc_readout_never_changes_other_classes():
    X, y = _toy()
    m = fd.AssocReadout(X.shape[1], 4, per_class=3, device="cpu")
    m.fit(X[y < 2], y[y < 2])
    before = m.W[:2 * 3].clone()
    m.fit(X[y >= 2], y[y >= 2])                         # 나중 클래스 학습
    assert torch.equal(m.W[:2 * 3], before)             # 앞 클래스 출력은 그대로
    assert (m.count.view(4, 3) > 0).all()               # 모든 원형이 채워짐
    assert m.accuracy(X, y) > 0.9


def test_train_linear_is_reproducible():
    X, y = _toy(n=100)
    r = [fd.train_linear(X[:60], y[:60], X[60:], y[60:], epochs=5, device="cpu")["model"].weight for _ in range(2)]
    assert torch.equal(r[0], r[1])


def test_assoc_readout_uses_all_prototypes_in_one_batch():
    X, y = _toy(n=40)
    m = fd.AssocReadout(X.shape[1], 4, per_class=3, device="cpu").fit(X, y, batch=64)   # 한 묶음에 전부
    assert (m.count.view(4, 3) > 0).all()
    assert m.count.sum() == 40


# ─────────────── 역전파 ───────────────
@needs_data
def test_trainable_layer_gradients_flow(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "KC", t_ms=20, dt=0.5, gains={"PN>KC": 2.0},
                               input_mode="regular", trainable=True)
    assert layer.log_scale.numel() == layer.w_syn.numel()
    x = torch.full((4, 344), 80.0, requires_grad=True)
    layer(x).sum().backward()
    g = layer.log_scale.grad
    assert torch.isfinite(g).all() and (g != 0).any()
    assert x.grad is not None and (x.grad != 0).any()       # 입력까지 기울기가 흐름


@needs_data
def test_trainable_subset_and_sign_kept(mb):
    layer = fd.ConnectomeLayer(mb, "PN", "MBON", trainable=["KC>MBON"])
    assert set(layer.edge_key[layer.train_pos.cpu().numpy()]) == {"KC>MBON"}
    with torch.no_grad():
        layer.log_scale.normal_(0, 2)
    w0, w = layer.w_base, layer.weights()
    assert torch.equal(torch.sign(w), torch.sign(w0))       # 부호(흥분/억제)는 그대로
    with pytest.raises(ValueError):
        fd.ConnectomeLayer(mb, "PN", "KC", trainable=["XX>YY"])


@needs_data
def test_untrained_trainable_layer_matches_fixed(mb):
    kw = dict(t_ms=20, dt=0.5, gains={"PN>KC": 2.0}, input_mode="regular")
    x = fd.RateEncoder(784, 344)(torch.rand(4, 784))
    with torch.no_grad():
        a = fd.ConnectomeLayer(mb, "PN", "KC", **kw)(x)
        b = fd.ConnectomeLayer(mb, "PN", "KC", trainable=True, **kw)(x)
    assert torch.allclose(a, b)


def test_surrogate_spike():
    x = torch.tensor([-1.0, -0.01, 0.01, 1.0], requires_grad=True)
    y = fd.layers.SpikeFn.apply(x, 10.0)
    assert y.tolist() == [0, 0, 1, 1]
    y.sum().backward()
    assert (x.grad > 0).all() and x.grad[1] > x.grad[0]      # 문턱 근처에서 기울기가 큼
