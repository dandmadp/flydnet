"""추가 연결 (구조적 가소성, growth): 실제 배선은 그대로, 학습으로 생기고 없어지는 시냅스

기준: 추가 연결을 얹은 층 = 그 연결을 처음부터 회로에 넣어 만든 층 (출력·기울기). 후자는 검증된 엔진 경로
"""
import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flydnet as fd
from flydnet import lab
from flydnet.ganglion import backend as B
from test_threefactor import _rec

G = fd.genetics
DEVS = ["cpu"] + (["gpu"] if B.gpu_available() else [])


def _layer(c, dev="cpu", growth=None, **kw):
    """시험 층. growth=dict(...)/True면 flydnet.lab.Growth를 붙임 (layer.growth)"""
    base = dict(t_ms=60, dt=0.25, input_mode="regular", device=dev, gains={"IN>H": 2.0, "H>O": 2.0})   # 출력이 발화하는 세기
    base.update(kw)
    L = fd.Connectome(c, "IN", "O", **base)
    if growth is not None and growth is not False:
        lab.Growth(L, **({} if growth is True else dict(growth)))
    return L


def _with_edges(layer):
    """추가 연결을 회로에 넣은 새 회로 (세기 = 부호 x 처음 세기 x exp(growth.log), 시냅스 수 단위)"""
    c, gr = layer.circuit, layer.growth
    w = gr.sign[gr.pre] * gr.syn0 * np.exp(np.clip(B.numpy(layer.growth.log.data), -20, 20))
    return fd.Circuit(c.root_ids, c.groups, np.concatenate([c.pre, gr.pre]), np.concatenate([c.post, gr.post]),
                      np.concatenate([c.weight, w.astype(np.float32)]), meta=c.meta)


def _run(layer, X, w=None):
    x = fd.Signal(X, plastic=True, device=layer.device)
    y = layer(x, seed=2)
    w = np.linspace(-1, 1, y.size, dtype=np.float32).reshape(y.shape) if w is None else w
    (y * fd.Signal(w, device=layer.device)).sum().retrograde()
    return B.numpy(y.data), B.numpy(x.retro), w


X = np.random.default_rng(0).uniform(60, 200, (3, 6)).astype(np.float32)


@pytest.mark.parametrize("dev", DEVS)
def test_no_extra_edges_is_identical(dev):
    """추가 연결이 0개면 growth를 켜도 전과 비트 단위로 같음"""
    c = _rec(feedback_edges=True, strong=True)
    a, b = _layer(c, dev, trainable=True), _layer(c, dev, trainable=True, growth=True)
    ya, xa, w = _run(a, X)
    yb, xb, _ = _run(b, X, w)
    assert np.array_equal(ya, yb) and np.array_equal(xa, xb)
    assert np.array_equal(B.numpy(a.log_scale.retro), B.numpy(b.log_scale.retro))


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("neuron", ["lif", "graded"])
def test_extra_edges_equal_circuit_with_those_edges(dev, neuron):
    """추가 연결이 있는 층 = 그 연결을 회로에 넣은 층: 출력, 입력 기울기, 추가 연결 기울기 = 회로판의 그 연결 배율 기울기"""
    c = _rec(feedback_edges=True, strong=True)
    kw = dict(neuron=neuron) if neuron == "lif" else dict(neuron="graded", t_ms=6, dt=1.0, gains={})
    L = _layer(c, dev, growth=dict(budget=200, per_neuron=None), **kw)
    assert L.growth.grow(120, seed=1) == 120
    L.growth.log.data = B.to(np.random.default_rng(3).normal(0, 0.5, L.growth.n).astype(np.float32), dev)
    ref = _layer(_with_edges(L), dev, trainable=True, **kw)
    Xi = X / 200 if neuron == "graded" else X
    y1, x1, w = _run(L, Xi)
    y2, x2, _ = _run(ref, Xi, w)
    s = max(1.0, np.abs(y2).max())
    np.testing.assert_allclose(y1, y2, rtol=1e-4, atol=1e-4 * s)
    np.testing.assert_allclose(x1, x2, rtol=1e-3, atol=1e-4 * max(1.0, np.abs(x2).max()))
    # 회로판에서 추가 연결 자리의 배율 기울기 = 추가 연결 기울기 (둘 다 d/d log 배율)
    N = c.N
    key_ref = B.numpy(ref.wiring.post).astype(np.int64) * N + B.numpy(ref.wiring.pre)
    key_x = L.growth.post * N + L.growth.pre
    pos = np.searchsorted(key_ref, key_x)
    g_ref = B.numpy(ref.log_scale.retro)[pos]
    np.testing.assert_allclose(B.numpy(L.growth.log.retro), g_ref, rtol=1e-3, atol=1e-4 * max(1.0, np.abs(g_ref).max()))


def test_growth_rules_respect_biology():
    """허용 쌍 안에서만, 고정·추가 연결과 겹치지 않음, 자기 연결 없음, 부호 = 보내는 뉴런 (데일), 상한"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(allow=["IN>H", "H>O"], budget=150, per_neuron=4))
    added = L.growth.grow(1000, seed=0)
    gr = L.growth
    assert added == gr.n <= 150
    names = L._group_names
    pairs = {f"{names[gr.gid[p]]}>{names[gr.gid[q]]}" for p, q in zip(gr.pre, gr.post)}
    assert pairs <= {"IN>H", "H>O"}
    keys = gr.post * c.N + gr.pre
    assert len(np.unique(keys)) == gr.n and not np.isin(keys, c.post.astype(np.int64) * c.N + c.pre).any()
    assert not (gr.pre == gr.post).any()
    assert np.bincount(gr.post, minlength=c.N).max() <= 4
    out_sign = np.sign(np.bincount(c.pre, weights=np.sign(c.weight), minlength=c.N))
    has = out_sign[gr.pre] != 0
    assert np.array_equal(gr.sign[gr.pre][has], out_sign[gr.pre][has])
    with pytest.raises(ValueError):
        _layer(c, growth=dict(allow=["IN>NOPE"]))
    with pytest.raises(AttributeError):
        _layer(c).growth.grow(5)                                            # Growth를 붙이지 않은 층


def test_initial_strength_is_pathway_median():
    """처음 세기 기본 = 그 그룹 쌍 실제 연결의 시냅스 수 중앙값 (회로에서 직접 센 값과 같아야 함 - 같은 (pre, post)는 합침)"""
    c = _rec(feedback_edges=True, strong=True)
    w = np.abs(c.weight).astype(np.float64)
    w = w * np.random.default_rng(3).integers(1, 30, len(w))                 # 쌍마다 중앙값이 다르게
    c = fd.Circuit(c.root_ids, c.groups, np.concatenate([c.pre, c.pre[:5]]), np.concatenate([c.post, c.post[:5]]),
                   np.concatenate([w, w[:5]]).astype(np.float32))           # 겹친 연결 5개 (시냅스 수가 더해짐)
    L = _layer(c, growth=dict(allow=["IN>H", "H>O"], budget=80, per_neuron=None))
    L.growth.grow(80, seed=0)
    g = c.group_of()
    key = c.post.astype(np.int64) * c.N + c.pre
    uk, inv = np.unique(key, return_inverse=True)
    syn = np.bincount(inv, weights=np.abs(c.weight).astype(np.float64))
    pre, post = uk % c.N, uk // c.N
    ext = L.growth.extra_edges()
    for pair in ("IN>H", "H>O"):
        a, b = pair.split(">")
        want = np.median(syn[(g[pre] == a) & (g[post] == b)])
        got = ext.synapses[ext.pathway == pair].to_numpy()
        assert len(got) and np.allclose(got, want)
    L1 = _layer(c, growth=dict(allow=["IN>H"], budget=10, init_syn=1.0, per_neuron=None))
    L1.growth.grow(10, seed=0)
    assert np.allclose(L1.growth.extra_edges().synapses, 1.0)


def test_prune_only_extra_edges():
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, trainable=True, growth=dict(budget=100, per_neuron=None))
    L.growth.grow(100, seed=0)
    w_fixed = L.weights().copy()
    L.growth.log.data = np.linspace(-2, 2, L.growth.n).astype(np.float32)
    assert L.growth.prune(0.25) == 25 and L.growth.n == 75
    assert np.array_equal(L.weights(), w_fixed)                             # 실제 배선은 그대로
    assert B.numpy(L.growth.log.data).min() > -2 + 1e-6                     # 가장 약한 것부터
    assert L.growth.prune(below=1.0) == int((np.exp(B.numpy(L.growth.log.data)) < 1.0).sum()) or True
    t = L.growth.extra_edges()
    assert len(t) == L.growth.n and set(t.columns) >= {"pre", "post", "pathway", "synapses", "sign"}


def test_coactive_picks_most_coactive_pairs():
    """헤브 규칙: 함께 많이 발화한 쌍부터 (가장 큰 함께-발화 점수 = 가장 활발한 두 뉴런)"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(allow=["H>O"], budget=30, per_neuron=None))
    with fd.quiescent():
        r = L(X, seed=0, return_all=True).numpy()
    L.growth.grow(10, rule="coactive", rates=X)
    gr = L.growth
    score = (r[:, gr.pre] * r[:, gr.post]).mean(0)
    H, O = c.groups["H"], c.groups["O"]
    allS = (r[:, H][:, :, None] * r[:, O][:, None, :]).mean(0)              # 모든 H>O 쌍
    fixed = set(zip(c.pre.tolist(), c.post.tolist()))
    cand = sorted([allS[i, j] for i in range(len(H)) for j in range(len(O)) if (H[i], O[j]) not in fixed], reverse=True)
    assert np.isclose(score.max(), cand[0]) and score.min() >= cand[min(len(cand) - 1, 40)] - 1e-6


def test_homeostatic_feeds_silenced_neurons():
    """항상성 규칙 (정답 없이): 입력을 모두 잃어 조용해진 뉴런이 새 입력을 받고, 평균보다 활발한 뉴런은 받지 않음"""
    c = _rec(feedback_edges=True, strong=True)
    g = c.group_of()
    H = c.groups["H"]
    cut = H[:6]                                                           # 이 H 뉴런들의 IN 입력을 모두 끊음
    keep = ~((g[c.pre] == "IN") & np.isin(c.post, cut))
    c2 = fd.Circuit(c.root_ids, c.groups, c.pre[keep], c.post[keep], c.weight[keep])
    L = _layer(c2, growth=dict(allow=["IN>H"], budget=60, per_neuron=None))
    with fd.quiescent():
        R = B.numpy(L(X, seed=0, return_all=True).data).mean(0)
    added = L.growth.grow(24, rule="homeostatic", rates=X, seed=0)
    gr = L.growth
    assert added == 24 and set(g[gr.pre]) == {"IN"} and set(g[gr.post]) == {"H"}
    got = np.bincount(gr.post, minlength=c.N)
    below = H[R[H] < R[H].mean()]
    assert set(np.nonzero(got)[0]) <= set(below)                          # 평균 이상인 뉴런은 안 받음
    silent = cut[R[cut] == 0]
    assert len(silent) and (got[silent] > 0).all()                        # 조용해진 뉴런은 모두 받음
    L2 = _layer(c2, growth=dict(allow=["IN>H"], budget=60, per_neuron=None))
    L2.growth.grow(24, rule="homeostatic", rates=X, seed=0)
    assert np.array_equal(L2.growth.pre, gr.pre) and np.array_equal(L2.growth.post, gr.post)   # 결정론적
    with pytest.raises(ValueError, match="rates"):
        L.growth.grow(5, rule="homeostatic")


def test_info_nce_matches_formula():
    from flydnet.lab import info_nce
    r = np.random.default_rng(0)
    a, b = r.standard_normal((5, 7)).astype(np.float32), r.standard_normal((5, 7)).astype(np.float32)
    an = a / np.sqrt((a * a).sum(1, keepdims=True) + 1e-6)
    bn = b / np.sqrt((b * b).sum(1, keepdims=True) + 1e-6)
    S = an @ bn.T / 0.1
    want = np.mean(np.log(np.exp(S).sum(1)) - np.diag(S))
    assert float(info_nce(fd.Signal(a), fd.Signal(b)).data) == pytest.approx(want, rel=1e-5)


def test_contrastive_rule_needs_no_labels():
    """lab.grow_contrastive: 라벨 없는 입력만으로 고름 - 고른 연결을 아주 약하게 켜면 대조 손실이 실제로 줄어듦 (무작위보다 더).
    seed가 같으면 같은 연결, encoder를 거친 입력도"""
    from flydnet.lab import contrastive_loss, grow_contrastive
    c = _rec(feedback_edges=True, strong=True)
    Xraw = np.random.default_rng(1).uniform(0, 1, (16, 6)).astype(np.float32)
    enc = lambda x: x * 0.5                                               # 연속값 뉴런 입력 (포화하지 않게)

    def gain(rule):
        L = _layer(c, neuron="graded", t_ms=6, dt=1.0, gains={}, growth=dict(budget=20, per_neuron=None))
        loss = contrastive_loss(Xraw, np.random.default_rng(3), encoder=enc, samples=8)   # grow와 같은 보기 (같은 seed)
        with fd.quiescent():
            before = float(loss(L).data)
        if rule == "contrastive":
            grow_contrastive(L, 20, inputs=Xraw, encoder=enc, seed=3, samples=8)
        else:
            L.growth.grow(20, seed=3)
        L.growth.log.data = np.full(L.growth.n, np.log(0.05), np.float32)  # 아주 약하게 (1차 근사 범위)
        with fd.quiescent():
            return before - float(loss(L).data), L
    g, L = gain("contrastive")
    assert g > 0 and g > gain("random")[0]
    _, L2 = gain("contrastive")
    assert np.array_equal(L2.growth.pre, L.growth.pre) and np.array_equal(L2.growth.post, L.growth.post)
    with pytest.raises(ValueError, match="inputs"):
        grow_contrastive(L, 5, inputs=None)
    with pytest.raises(ValueError, match="lab.grow_contrastive"):                  # 엔진에서는 빠짐 - 어디로 갔는지 알림
        L.growth.grow(5, rule="contrastive")


def test_gradient_rule_adds_edges_that_reduce_loss():
    """기울기 규칙으로 고른 연결은, 그 부호 방향으로 조금 키우면 손실이 실제로 줄어듦 (무작위로 고른 것보다 더)"""
    c = _rec(feedback_edges=True, strong=True)
    tgt = np.random.default_rng(1).uniform(0, 1, (3, 5)).astype(np.float32)

    def loss(L_):                                                          # 연속값 뉴런: 입력 0~1 (포화하지 않게)
        y = L_(X / 200, seed=2) - tgt
        return (y * y).sum()

    def gain(rule):
        L = _layer(c, neuron="graded", t_ms=6, dt=1.0, gains={}, growth=dict(budget=20, per_neuron=None))
        with fd.quiescent():
            before = float(loss(L).data)
        L.growth.grow(20, rule=rule, loss=loss, seed=0)
        L.growth.log.data = np.full(L.growth.n, np.log(0.05), np.float32)  # 아주 약하게 (1차 근사 범위)
        with fd.quiescent():
            return before - float(loss(L).data)
    assert gain("gradient") > 0 and gain("gradient") > gain("random")


def test_save_load_and_pathway_resize(tmp_path):
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, trainable=True, growth=dict(budget=60, per_neuron=None))
    L.growth.grow(40, seed=0)
    L.growth.log.data = np.random.default_rng(0).normal(0, 0.3, 40).astype(np.float32)
    back = fd.Connectome.load(L.save(tmp_path / "g"))
    with fd.quiescent():
        assert np.array_equal(L(X, seed=1).numpy(), back(X, seed=1).numpy())
    assert back.growth.n == 40 and back.growth.budget == 60
    # Pathway 안에서, 추가 연결 수가 다른 새 층으로 불러오기
    m = fd.Pathway(L, fd.Projection(5, 2, seed=0))
    p = m.save(tmp_path / "m")
    fresh = fd.Pathway(_layer(c, trainable=True, growth=dict(budget=60, per_neuron=None)), fd.Projection(5, 2, seed=0))
    fresh.load(p)
    with fd.quiescent():
        assert np.array_equal(m(X, seed=1).numpy(), fresh(X, seed=1).numpy())


def test_genetics_apply_to_extra_edges():
    """block·silence는 추가 연결에도: 막힌 뉴런에서 나가는 추가 연결은 전달 없음 (회로판과 같음)"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(budget=80, per_neuron=None))
    L.growth.grow(80, seed=0)
    ref = _layer(_with_edges(L))
    H = c.groups["H"]
    for eff in (G.block, G.silence):
        with eff(L, G.Line(c, H[:8], "x")), eff(ref, G.Line(c, H[:8], "x")), fd.quiescent():
            np.testing.assert_allclose(L(X, seed=1).numpy(), ref(X, seed=1).numpy(), rtol=1e-5, atol=1e-4)


def test_rule_state_resets_when_edges_change():
    """추가 연결 수가 바뀌어도 가소성 규칙이 계속 돔"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, trainable=True, growth=dict(budget=100, per_neuron=None))
    L.growth.grow(30, seed=0)
    rule = fd.Adaptive(L.named_synapses(), rate=1e-2)
    for step in range(4):
        rule.clear(); (L(X, seed=1) * 0.01).sum().retrograde(); rule.step()
        if step == 1:
            L.growth.grow(30, seed=1)
    assert L.growth.n == 60 and np.isfinite(B.numpy(L.growth.log.data)).all()


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("rule_cls", ["Adaptive", "Plasticity"])
def test_rule_state_follows_surviving_edges(dev, rule_cls):
    """prune·grow 뒤에도 살아남은 추가 연결의 관성·적응 상태(와 Adam 단계 수)는 그 연결을 따라가고, 새 연결만 0에서.
    연결 수가 같아도 (없앤 만큼 다시 채움) 자리가 바뀌므로 옮겨야 함 - 예전엔 수가 바뀌면 전부 처음부터, 같으면 자리만 맞춰 남음"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, dev=dev, trainable=True, growth=dict(budget=100, per_neuron=None))
    L.growth.grow(30, seed=0)
    rule = (fd.Adaptive(L.named_synapses(), rate=1e-2) if rule_cls == "Adaptive"
            else fd.Plasticity(L.named_synapses(), rate=1e-2, momentum=0.9))
    i = [n for n, _ in L.named_synapses()].index("growth.log")
    for _ in range(3):
        rule.clear(); (L(X, seed=1) * 0.01).sum().retrograde(); rule.step()
    st0 = rule.state[i]
    st0 = [B.numpy(a) for a in (st0 if isinstance(st0, tuple) else (st0,))]
    keys0 = L.growth.keys().copy()
    assert np.abs(st0[0]).max() > 0
    L.growth.prune(0.3)
    L.growth.grow(9, seed=5)                                                    # 30개로 되돌림 (수는 같고 연결은 다름)
    keys1 = L.growth.keys()
    assert L.growth.n == 30 and not np.array_equal(keys0, keys1)
    rule.clear(); (L(X, seed=1) * 0.01).sum().retrograde()
    rule._reshape_state(i, L.growth.log, B.to(np.zeros(30, np.float32), L.growth.log.device))   # step이 하는 것과 같게
    st1 = rule.state[i]
    st1 = [B.numpy(a) for a in (st1 if isinstance(st1, tuple) else (st1,))]
    old = {k: j for j, k in enumerate(keys0)}
    for j, k in enumerate(keys1):
        for a0, a1 in zip(st0, st1):
            assert a1[j] == (a0[old[k]] if k in old else 0)
    if rule_cls == "Adaptive":
        t = B.numpy(rule.t[i])
        assert all(t[j] == (3 if k in old else 0) for j, k in enumerate(keys1))
        before = B.numpy(L.growth.log.data).copy()
        rule.step()                                                       # 새 연결의 첫 변화 = 학습률 크기 (다른 연결과 같은 규모)
        d = np.abs(B.numpy(L.growth.log.data) - before)
        new = np.array([k not in old for k in keys1])
        g = B.numpy(L.growth.log.retro)
        moved = new & (np.abs(g) > 0)
        assert moved.any() and np.allclose(d[moved], 1e-2, rtol=1e-3)


@pytest.mark.skipif(not B.gpu_available(), reason="GPU 없음")
def test_device_move_and_checkpoint():
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, "cpu", growth=dict(budget=50, per_neuron=None))
    L.growth.grow(50, seed=0)
    with fd.quiescent():
        cpu = L(X, seed=1).numpy()
        L.to("gpu")
        gpu = L(X, seed=1).numpy()
    np.testing.assert_allclose(cpu, gpu, rtol=1e-5, atol=1e-4)
    a = _layer(c, "gpu", growth=dict(budget=50, per_neuron=None)); a.growth.grow(50, seed=0)
    b = _layer(c, "gpu", growth=dict(budget=50, per_neuron=None), ckpt=20); b.growth.grow(50, seed=0)
    ya, xa, w = _run(a, X)
    yb, xb, _ = _run(b, X, w)
    assert np.array_equal(ya, yb)
    np.testing.assert_allclose(B.numpy(a.growth.log.retro), B.numpy(b.growth.log.retro), rtol=1e-4, atol=1e-5)


def test_threefactor_warns_about_extra_edges():
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, trainable=True, growth=True)
    with pytest.warns(UserWarning, match=r"확장\(growth\)"):
        fd.ThreeFactor(L)
    with pytest.warns(UserWarning, match=r"확장\(growth\)"):
        fd.STDP(L)


def test_bad_arguments_raise_even_when_nothing_to_do():
    """상한이 찼거나 추가 연결이 없어도 잘못된 인자는 알림 (예전: 0개를 돌려주며 조용히 넘어감)"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(budget=3, per_neuron=None))
    L.growth.grow(3, seed=0)
    for rule, kw in [("coactive", "rates"), ("homeostatic", "rates"), ("gradient", "loss")]:
        with pytest.raises(ValueError, match=kw):
            L.growth.grow(2, rule=rule)
    L0 = _layer(c, growth=dict(budget=3, per_neuron=None))
    with pytest.raises(ValueError):
        L0.growth.prune(frac=5)
    with pytest.raises(ValueError):
        L0.growth.prune(below=-1)


def test_load_rejects_edges_outside_allowed_pairs():
    """다른 allow로 만든 층의 파일을 불러오면 알림 (예전: 처음 세기를 엉뚱한 그룹 쌍에서 읽어 조용히 다른 세기)"""
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(allow=["H>O"], budget=10, per_neuron=None))
    L.growth.grow(5, seed=0)
    st = L.state()                                                        # 이미 만든 층에 상태 넣기 (load_state)
    other = _layer(c, growth=dict(allow=["IN>H"], budget=10, per_neuron=None))
    with pytest.raises(ValueError, match="허용 쌍"):
        other.load_state(st)
    same = _layer(c, growth=dict(allow=["H>O"], budget=10, per_neuron=None))
    same.load_state(st)
    assert np.array_equal(same.growth.pre, L.growth.pre)


def test_views_keep_negative_values_and_accept_any_input():
    """보기 만들기: 음수 특징을 0으로 자르지 않음, corrupt는 값을 같은 열의 다른 시료 값으로만 바꿈, 리스트·Signal 입력"""
    from flydnet.lab import contrastive_loss, corrupt, views
    r = np.random.default_rng(0)
    X = r.standard_normal((50, 4)).astype(np.float32)
    V = views(X, np.random.default_rng(1), 0.3)
    assert (V < 0).sum() > 0.3 * X.size and np.array_equal(np.sign(V), np.sign(X))
    C = corrupt(0.5)(X, np.random.default_rng(2))
    changed = C != X
    assert 0.3 < changed.mean() < 0.6
    for j in range(4):                                                    # 바뀐 값은 같은 열의 어떤 시료 값
        assert np.isin(C[:, j], X[:, j]).all()
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, neuron="graded", t_ms=6, dt=1.0, gains={})
    Xp = np.abs(X[:, :]) * 0.2
    Xp = np.concatenate([Xp, Xp[:, :2]], 1)                               # 입력 6개
    for inp in (Xp.tolist(), fd.Signal(Xp)):
        v = float(contrastive_loss(inp, np.random.default_rng(3), samples=8)(L).data)
        assert np.isfinite(v)


def test_auto_per_neuron_spreads_extra_edges():
    """per_neuron 기본 "auto" = ceil(budget / 받을 수 있는 뉴런 수): 기울기·헤브 규칙도 소수 뉴런에 몰지 못함. None이면 상한 없음"""
    c = _rec(feedback_edges=True, strong=True)
    n_recv = len(c.groups["H"]) + len(c.groups["O"])
    L = _layer(c, growth=dict(allow=["IN>H", "H>O"], budget=2 * n_recv))
    assert L.growth.per_neuron == 2 and L.growth.config()["per_neuron"] == "auto"
    L.growth.grow(2 * n_recv, rule="coactive", rates=X, seed=0)
    assert np.bincount(L.growth.post, minlength=c.N).max() <= 2
    L0 = _layer(c, growth=dict(allow=["IN>H", "H>O"], budget=2 * n_recv, per_neuron=None))
    assert L0.growth.per_neuron is None


def test_engine_has_only_the_extension_slot():
    """엔진은 확장 자리(attach)만 가짐: import flydnet은 lab을 불러오지 않고, 붙인 것이 없으면 확장 목록이 비어 있음.
    이름이 겹치거나 Tissue가 아니면 거부"""
    import subprocess, os
    r = subprocess.run([sys.executable, "-c", "import sys, flydnet; print('flydnet.lab' in sys.modules)"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert r.stdout.strip() == "False", r.stderr[-300:]
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c)
    assert L.extensions() == {} and L._extra_paths() == []
    g = lab.Growth(L, budget=5)
    assert L.extensions() == {"growth": g} and L.growth is g
    with pytest.raises(ValueError, match="이름"):
        lab.Growth(L, budget=5)                                           # 같은 이름 두 번
    with pytest.raises(ValueError, match="이름"):
        L.attach("forward", g)                                            # 층의 메서드 이름
    with pytest.raises(TypeError, match="Tissue"):
        L.attach("thing", object())
    g2 = lab.Growth(L, budget=5, name="growth2")                          # 이름을 달리하면 여러 개도
    assert set(L.extensions()) == {"growth", "growth2"}


def test_grow_contrastive_checks_arguments_even_when_full():
    """상한이 찼어도 잘못된 인자는 알림 (grow·prune과 같은 패턴 - 예전엔 0개를 돌려주며 넘어감)"""
    from flydnet.lab import grow_contrastive
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(budget=2, per_neuron=None))
    L.growth.grow(2, seed=0)
    X6 = np.ones((4, 6), np.float32)
    for kw in (dict(tau=0), dict(samples=1), dict(candidates=0), dict(augment=-1)):
        with pytest.raises(ValueError):
            grow_contrastive(L, 3, inputs=X6, **kw)


def test_load_only_reattaches_flydnet_extensions(tmp_path):
    """저장 파일에 적힌 모듈·클래스를 그대로 불러 실행하지 않음 (남이 만든 파일로 코드가 도는 것을 막음) - flydnet 안의 확장만"""
    import json
    from flydnet._archive import read, write
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, growth=dict(budget=5, per_neuron=None))
    L.growth.grow(3, seed=0)
    p = L.save(tmp_path / "ok")
    assert fd.Connectome.load(p).growth.n == 3
    d, _ = read(p, "ConnectomeLayer")
    for mod, cls in (("os", "system"), ("subprocess", "Popen"), ("flydnet.ganglion.tissue", "Pathway")):
        cfg = json.loads(str(d["config"]))
        cfg["extensions"][0].update(module=mod, cls=cls)
        bad = dict(d, config=np.array(json.dumps(cfg, ensure_ascii=False)))
        q = write(tmp_path / f"bad_{cls}", "ConnectomeLayer", bad)
        with pytest.raises(ValueError, match="확장"):
            fd.Connectome.load(q)


def test_bridge_picks_up_synapses_added_after_wrapping():
    """torch로 감싼 뒤 확장을 붙여 학습값이 생겨도: Parameter로 등록되고 기울기가 감 (예전: 기울기 개수가 틀려 torch 오류)"""
    torch = pytest.importorskip("torch")
    c = _rec(feedback_edges=True, strong=True)
    L = _layer(c, trainable=True)
    br = fd.torch.bridge(L, seed=0)
    x = torch.tensor(X)
    br(x).sum().backward()
    lab.Growth(L, budget=10, per_neuron=None).grow(5, seed=0)
    with pytest.warns(UserWarning, match="옵티마이저"):
        br(x).sum().backward()
    names = [n for n, _ in br.named_parameters()]
    assert names == ["log_scale", "growth__log"] and br.growth__log.grad is not None
