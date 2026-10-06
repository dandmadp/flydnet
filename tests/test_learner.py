"""fd.Learner: 한 번 훑기 학습, 이어서 배우기(앞의 것 유지), 다른 종류 데이터(source), 저장·불러오기"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion.physiology import recall, reinforce_
from test_threefactor import _rec


def _c():
    c = _rec(feedback_edges=True)
    return fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)


def _data(n=60, d=6, classes=4, seed=0):
    r = np.random.default_rng(seed)
    centers = r.uniform(0, 1, (classes, d))
    y = np.arange(n) % classes
    return (centers[y] + 0.05 * r.standard_normal((n, d))).astype(np.float32), y


def _L(**kw):
    return fd.Learner(_c(), "IN", "O", **dict(dict(t_ms=30, device="cpu"), **kw))


@pytest.mark.parametrize("rule", ["lda", "assoc", "backprop"])
def test_learns_and_keeps_old_classes(rule):
    X, y = _data()
    L = _L(rule=rule)
    L.learn(X[y < 2], y[y < 2])
    before = L.scores(X[y < 2])
    L.learn(X[y >= 2], y[y >= 2])                                         # 새 클래스를 이어서
    assert L.classes == [0, 1, 2, 3]
    after = L.scores(X[y < 2])[:, :2]
    if rule == "assoc":                                                   # 연합: 앞 클래스 기억은 한 비트도 안 바뀜
        np.testing.assert_array_equal(before, after)
    elif rule == "lda":                                                   # lda: 앞 클래스 평균은 그대로 (공유 공분산만 갱신)
        assert np.isfinite(after).all()
    else:                                                                 # 역전파: 이번에 나온 클래스 가중치만 바뀜
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
    assert L.score(X, y) > 0.5                                            # 찍기 0.25


def test_assoc_matches_assoc_readout():
    """Learner(assoc)의 기억 = 같은 특징에 AssocReadout을 쓴 것 (같은 규칙, 같은 묶음)"""
    X, y = _data()
    L = _L(rule="assoc", per_class=2).learn(X, y)
    F = L.features(X)
    r = fd.AssocReadout(F.shape[1], 4, per_class=2, device="cpu")
    for i in range(0, len(F), 32):
        r.step(F[i:i + 32], y[i:i + 32])
    np.testing.assert_allclose(L.scores(X), r.scores(F), rtol=1e-6, atol=1e-6)


def test_backprop_new_class_starts_at_its_mean():
    X, y = _data()
    L = _L(rule="backprop", epochs=1, rate=1e-12)                         # 학습률 ~0: 초기화만 보임
    L.learn(X, y)
    x = L._center(L.features(X))
    for c in range(4):
        w = L.head.W.numpy()[c]
        m = x[y == c].mean(0)
        assert np.dot(w, m) / (np.linalg.norm(w) * np.linalg.norm(m)) > 0.999


def test_labels_any_type_and_single_sample():
    X, y = _data()
    names = np.array(["a", "b", "c", "d"])[y]
    L = _L().learn(X, names)
    assert L.predict(X[0]) in {"a", "b", "c", "d"} and isinstance(L.predict(X[0]), str)
    p = _L().learn(X, y).predict(X)
    assert p.dtype == np.int64 and p.shape == (len(X),)
    assert L.predict(X, classes=["a", "b"]).tolist() == [v for v in L.predict(X, classes=["a", "b"])]
    assert set(L.predict(X, classes=["a", "b"])) <= {"a", "b"}
    with pytest.raises(ValueError, match="배운 적 없는"):
        L.predict(X, classes=["z"])


def test_sources_separate_encoders_shared_memory():
    X, y = _data()
    Z = np.concatenate([X, X], 1)                                         # 다른 모양의 데이터 (특징 12개)
    L = _L().learn(X[y < 2], y[y < 2])
    with pytest.raises(ValueError, match="source"):
        L.learn(Z[y >= 2], y[y >= 2])
    L.learn(Z[y >= 2], y[y >= 2], source="wide")
    assert set(L.encoders) == {"default", "wide"} and L.classes == [0, 1, 2, 3]
    gains = dict(L.layer.gains)
    L.learn(Z[:4], y[:4], source="wide")
    assert L.layer.gains == gains                                         # 보정은 처음 한 번만 (기억과 특징이 어긋나지 않게)


@pytest.mark.parametrize("rule", ["lda", "assoc", "backprop"])
def test_save_load_roundtrip(tmp_path, rule):
    X, y = _data()
    c = _c()
    L = fd.Learner(c, "IN", "O", rule=rule, t_ms=30, device="cpu", epochs=2).learn(X, ["p", "q", "r", "s"] * 15)
    L.learn(np.concatenate([X, X], 1)[:8], ["t"] * 8, source="wide")
    p = L.save(tmp_path / "l")
    M = fd.Learner.load(p, circuit=c, device="cpu")
    for src, data in [("default", X), ("wide", np.concatenate([X, X], 1))]:
        np.testing.assert_array_equal(M.scores(data, src), L.scores(data, src))
    assert M.classes == L.classes and M.rule == rule
    M.learn(X[:4], ["u"] * 4)                                            # 불러온 뒤 계속 배움
    assert M.classes[-1] == "u"
    other = fd.Circuit(c.root_ids, c.groups, c.pre[1:], c.post[1:], c.weight[1:])     # 연결 하나가 다른 회로
    with pytest.raises(RuntimeError, match="배선"):
        fd.Learner.load(p, circuit=other, device="cpu")


def test_errors():
    X, y = _data()
    with pytest.raises(ValueError, match="rule"):
        _L(rule="sgd")
    with pytest.raises(RuntimeError, match="learn"):
        _L().predict(X)
    with pytest.raises(ValueError, match="개수"):
        _L().learn(X, y[:-1])
    with pytest.raises(ValueError, match="NaN"):
        _L().learn(np.full((4, 6), np.nan, np.float32), [0, 1, 0, 1])
    with pytest.raises(RuntimeError, match="저장할 것"):
        _L().save("x")
    with pytest.raises(ValueError, match="shrink"):
        _L(shrink=-1)
    assert _L().learn(X, y[:, None]).classes == [0, 1, 2, 3]            # (n, 1) 라벨은 폄
    with pytest.raises(ValueError, match="1차원"):
        _L().learn(X, np.c_[y, y])


def test_lda_equals_batch_formula_and_is_order_free():
    """흐름 LDA = 한 번에 계산한 LDA (클래스 평균 + 클래스 안 공분산) - 묶음을 나누거나 순서를 바꿔도 같음. 앞 클래스 평균은 그대로"""
    from flydnet.learner import _StreamingLDA
    r = np.random.default_rng(0)
    F = r.standard_normal((90, 7)) + np.repeat(r.standard_normal((3, 7)) * 3, 30, 0)
    y = np.repeat(np.arange(3), 30)
    a = _StreamingLDA(7, shrink=0.1)
    a.learn(F, y, 3)
    b = _StreamingLDA(7, shrink=0.1)
    perm = r.permutation(90)
    for k in range(0, 90, 7):                                             # 섞어서 7개씩
        b.learn(F[perm[k:k + 7]], y[perm[k:k + 7]], 3)
    np.testing.assert_allclose(b.S, a.S, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(b.mu, a.mu, rtol=1e-12, atol=1e-12)
    mu = np.stack([F[y == c].mean(0) for c in range(3)])
    S = sum((F[y == c] - mu[c]).T @ (F[y == c] - mu[c]) for c in range(3))
    np.testing.assert_allclose(a.S, S, rtol=1e-9)
    Sig = S / (90 - 3)
    A = Sig + 0.1 * np.trace(Sig) / 7 * np.eye(7)
    W = np.linalg.solve(A, mu.T).T
    np.testing.assert_allclose(a.scores(F), F @ W.T - 0.5 * (W * mu).sum(1), rtol=1e-6, atol=1e-6)
    c = _StreamingLDA(7)                                                  # 새 클래스를 더해도 앞 클래스 평균은 그대로
    c.learn(F[y < 2], y[y < 2], 2)
    m0 = c.mu.copy()
    c.learn(F[y == 2], y[y == 2], 3)
    np.testing.assert_array_equal(c.mu[:2], m0)
    assert c.shrinkage() >= 7 / 90                                        # auto: 적어도 특징 수 / 시료 수


@pytest.mark.parametrize("rule", ["lda", "assoc", "backprop"])
def test_continue_after_load_equals_no_save(tmp_path, rule):
    """저장 → 불러오기 → 이어 배우기 = 저장 없이 이어 배우기 (backprop은 Adam 관성까지 저장 - 예전엔 잃어 점수가 0.08 달라짐)"""
    X, y = _data()
    c = _c()
    a = fd.Learner(c, "IN", "O", rule=rule, t_ms=30, device="cpu", epochs=2).learn(X[:30], y[:30])
    b = fd.Learner.load(a.save(tmp_path / "l"), circuit=c, device="cpu")
    a.learn(X[30:], y[30:])
    b.learn(X[30:], y[30:])
    np.testing.assert_array_equal(a.scores(X), b.scores(X))


def test_failed_learn_rolls_back_new_labels():
    """learn이 배우다 실패하면 새 라벨을 되돌림 - 예전엔 라벨만 남아, 나중에 그중 일부만 배우면 backprop의 빈 클래스 가중치가 NaN"""
    X, y = _data()
    L = _L(rule="backprop", epochs=1).learn(X[y < 2], y[y < 2])
    orig = L._learn_backprop

    def boom(F, yi):
        raise RuntimeError("boom")
    L._learn_backprop = boom
    with pytest.raises(RuntimeError):
        L.learn(X[y >= 2], y[y >= 2])
    L._learn_backprop = orig
    assert L.classes == [0, 1]
    L.learn(X[y == 2], y[y == 2])                                         # 3만 배우고 4는 아직
    assert L.classes == [0, 1, 2] and np.isfinite(L.head.W.numpy()).all()
