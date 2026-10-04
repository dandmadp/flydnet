"""fd.ThreeFactor (e-prop 방식 3요소 규칙): 출력으로 들어오는 연결은 역전파와 같은 기울기, 피드백 방식, 오류"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from test_genetics import _chain


def _rec(seed=0, n_in=6, n_h=20, n_out=5, feedback_edges=False):
    """입력 → 숨은(되먹임 포함) → 출력. feedback_edges면 출력 → 숨은 연결도"""
    rng = np.random.default_rng(seed)
    N = n_in + n_h + n_out
    H, O = (n_in, n_in + n_h), (n_in + n_h, N)
    blocks = [((0, n_in), H, .5, 6), (H, H, .15, 3), (H, O, .5, 6)] + ([(O, H, .4, 3)] if feedback_edges else [])
    pre, post, w = [], [], []
    for a, b, p, s in blocks:
        for i in range(*a):
            for j in range(*b):
                if i != j and rng.random() < p:
                    pre.append(i); post.append(j)
                    w.append(rng.integers(1, 8) * (1 if rng.random() < .8 else -1) * s / 3)
    return fd.Circuit(np.arange(N), {"IN": np.arange(n_in), "H": np.arange(*H), "O": np.arange(*O)},
                      pre, post, np.array(w, np.float32))


def _grads(c, inp, feedback, device="cpu", t_ms=100):
    x = np.random.default_rng(1).uniform(50, 200, (8, len(c.groups[inp]))).astype(np.float32)
    tgt = np.random.default_rng(2).uniform(0, 1, (8, len(c.groups["O"]))).astype(np.float32)
    L = fd.ConnectomeLayer(c, inp, "O", t_ms=t_ms, trainable=True, device=device)
    out = L(x, seed=3)
    ((out * 0.02 - tgt) ** 2).sum().retrograde()
    bptt = B.numpy(L.log_scale.retro).copy(); L.log_scale.retro = None
    tf = fd.ThreeFactor(L, feedback=feedback)
    o = tf(x, seed=3)
    ((o * 0.02 - tgt) ** 2).sum().retrograde()
    g_values = B.numpy(tf.assign(o))
    tf3 = B.numpy(L.log_scale.retro).copy()
    onto_out = np.isin(B.numpy(L.wiring.post), B.numpy(L.out_idx))
    return bptt, tf3, onto_out, g_values, L


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.mark.parametrize("circuit,inp", [(_chain(), "A"), (_rec(), "IN")])
def test_output_synapses_match_bptt(circuit, inp):
    """출력 뉴런으로 들어오는 연결: 역전파도 리셋에서 기울기를 끊으므로 3요소 규칙과 같은 값"""
    bptt, tf3, onto, _, _ = _grads(circuit, inp, "none")
    assert _cos(tf3[onto], bptt[onto]) > 0.999
    np.testing.assert_allclose(tf3[onto], bptt[onto], rtol=1e-3, atol=1e-6 * np.abs(bptt).max())
    assert np.abs(tf3[~onto]).max() == 0                                  # feedback="none": 숨은 연결은 그대로


def test_feedback_kinds():
    c = _rec(feedback_edges=True)
    _, none, onto, _, _ = _grads(c, "IN", "none")
    _, rnd, _, _, _ = _grads(c, "IN", "random")
    _, con, _, _, _ = _grads(c, "IN", "connectome")
    for g in (rnd, con):
        np.testing.assert_allclose(g[onto], none[onto], rtol=1e-5)        # 출력 쪽은 피드백과 무관
        assert np.abs(g[~onto]).max() > 0                                  # 숨은 연결도 학습 신호를 받음
    assert not np.allclose(rnd[~onto], con[~onto])
    c0 = _rec(feedback_edges=False)                                        # 실제 피드백 경로가 없으면 connectome = none
    _, con0, onto0, _, _ = _grads(c0, "IN", "connectome")
    assert np.abs(con0[~onto0]).max() == 0


def test_three_factor_learns():
    """3요소 규칙만으로 출력 발화율을 목표에 가깝게 (역전파 없이)"""
    c = _rec(feedback_edges=True)
    c = fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)    # 출력이 발화하도록 세게
    x = np.random.default_rng(1).uniform(50, 200, (8, 6)).astype(np.float32)
    tgt = np.random.default_rng(2).uniform(0.2, 1, (8, 5)).astype(np.float32)
    L = fd.ConnectomeLayer(c, "IN", "O", t_ms=100, trainable=True, device="cpu")
    tf = fd.ThreeFactor(L, feedback="random")
    rule = fd.AdaptivePlasticity(L.synapses(), rate=0.05)
    losses = []
    for step in range(40):
        o = tf(x, seed=step)
        loss = ((o * 0.02 - tgt) ** 2).mean()
        rule.clear(); loss.retrograde(); tf.assign(o); rule.step()
        losses.append(float(loss.data))
    assert np.mean(losses[-5:]) < 0.6 * np.mean(losses[:5])


def test_three_factor_gpu_matches_cpu():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    c = _rec(feedback_edges=True)
    a = _grads(c, "IN", "connectome", device="cpu")[1]
    b = _grads(c, "IN", "connectome", device="gpu")[1]
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6 * np.abs(a).max())


def test_three_factor_errors():
    c = _chain()
    with pytest.raises(ValueError, match="학습하는 연결"):
        fd.ThreeFactor(fd.ConnectomeLayer(c, "A", "O", t_ms=20, device="cpu"))
    with pytest.raises(ValueError, match="스파이킹"):
        fd.ThreeFactor(fd.ConnectomeLayer(c, "A", "O", t_ms=20, trainable=True, timing="legacy", device="cpu"))
    L = fd.ConnectomeLayer(c, "A", "O", t_ms=20, trainable=True, device="cpu")
    with pytest.raises(ValueError, match="feedback"):
        fd.ThreeFactor(L, feedback="dopamine")
    tf = fd.ThreeFactor(L)
    with pytest.raises(RuntimeError, match="먼저"):
        tf.assign(fd.Signal(np.zeros((1, 6), np.float32), plastic=True))
    o = tf(np.ones((1, 6), np.float32) * 100, seed=0)
    with pytest.raises(RuntimeError, match="retro"):
        tf.assign(o)
    assert L._observer is None
