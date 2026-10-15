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
    X = centers[y] + 0.05 * r.standard_normal((n, d))
    return np.clip(X, 0, None).astype(np.float32), y                     # 입력은 0 이상 (음수는 발화율 0으로 잘림 - 0.1.19부터 경고)


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
    L = _L(rule="assoc", per_class=2, input_weight=0).learn(X, y)       # KC 경로만 (원래 입력 없이)
    F = L.features(X)
    r = fd.AssocReadout(F.shape[1], 4, per_class=2, device="cpu")
    for i in range(0, len(F), 32):
        r.step(F[i:i + 32], y[i:i + 32])
    np.testing.assert_allclose(L.scores(X), r.scores(F), rtol=1e-6, atol=1e-6)


def test_backprop_new_class_starts_at_its_mean():
    X, y = _data()
    L = _L(rule="backprop", epochs=1, rate=1e-12, input_weight=0)        # 학습률 ~0: 초기화만 보임 (KC 경로만)
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
    with pytest.raises(ValueError, match="input_weight=0"):               # 원래 입력을 붙이면 종류는 하나만
        _L().learn(X[y < 2], y[y < 2]).learn(Z[y >= 2], y[y >= 2], source="wide")
    L = _L(input_weight=0).learn(X[y < 2], y[y < 2])
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
    L = fd.Learner(c, "IN", "O", rule=rule, t_ms=30, device="cpu", epochs=2, input_weight=0).learn(X, ["p", "q", "r", "s"] * 15)
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


@pytest.mark.parametrize("rule", ["lda", "assoc", "backprop"])
def test_input_weight_memory_and_roundtrip(tmp_path, rule):
    """input_weight > 0: 기억 = KC 특징 + 표준화한 원래 입력 (기준은 처음 배울 때 고정, 저장·불러오기에 같이).
    0이면 KC만 (0.1.19까지와 같음), 0.1.19까지 저장한 파일(input_weight 없음)은 KC만으로 불러옴"""
    X, y = _data()
    c = _c()
    L = fd.Learner(c, "IN", "O", rule=rule, t_ms=30, device="cpu", epochs=2).learn(X[:40], y[:40])
    F = L._memory_features(X, "default")
    assert F.shape[1] == L.layer.n_out + X.shape[1]
    saved = L.inp.pop("default")                                         # 기준이 없으면 예측 데이터로 정하지 않고 오류
    with pytest.raises(RuntimeError, match="learn"):
        L.scores(X)
    L.inp["default"] = saved
    mu = L.inp["default"][0].copy()
    L.learn(X[40:] * 1.5, y[40:])                                        # 이어 배워도 기준은 그대로
    np.testing.assert_array_equal(L.inp["default"][0], mu)
    M = fd.Learner.load(L.save(tmp_path / "w"), circuit=c, device="cpu")
    np.testing.assert_array_equal(M.scores(X), L.scores(X))
    K = fd.Learner(c, "IN", "O", rule=rule, t_ms=30, device="cpu", epochs=2, input_weight=0).learn(X, y)
    assert K._memory_features(X, "default").shape[1] == K.layer.n_out
    import json
    from flydnet._archive import read, write
    d, _ = read(K.save(tmp_path / "k"), "Learner")                      # 0.1.19식 파일: 설정에 input_weight 없음
    cfg = json.loads(str(d["config"])); cfg.pop("input_weight")
    old = write(tmp_path / "old", "Learner", dict(d, config=np.array(json.dumps(cfg))))
    O = fd.Learner.load(old, circuit=c, device="cpu")
    assert O.input_weight == 0 and np.array_equal(O.scores(X), K.scores(X))
    with pytest.raises(ValueError, match="input_weight"):
        fd.Learner(input_weight=-1)


def test_input_standardization_is_bounded():
    """학습 때 늘 0이던 특징이 시험 때 켜져도 원래 입력 부분이 폭발하지 않음 (예전: 표준편차 0 → 1e-6으로 나눠 수십만 배 -
    MNIST 가장자리 픽셀, 시험 2,000장 중 2장). 첫 학습이 시료 1개여도 크기 기준이 섬 (예전: KC 분산 0 → 원래 입력이 사실상 꺼짐)"""
    X, y = _data()
    X[:, 0] = 0.0                                                        # 학습에선 늘 0
    L = _L().learn(X, y)
    probe = X[:3].copy()
    probe[:, 0] = 1.0
    mu, sd, scale = L.inp["default"]
    raw = L._memory_features(probe, "default")[:, L.layer.n_out:]
    assert np.isfinite(raw).all() and np.abs(raw).max() <= 10 * scale * 1.0001
    import warnings
    with warnings.catch_warnings(record=True) as rec:                    # 처음 1개: 기준을 정하기엔 적다고 알림 (보정 경고도 같이 남)
        warnings.simplefilter("always")
        one = _L().learn(X[:1], y[:1])
    assert any("시료가 1개" in str(w.message) for w in rec)
    assert one.inp["default"][2] > 1e-3 and np.isfinite(one._memory_features(X, "default")).all()


def test_failed_learn_does_not_keep_input_stats():
    """학습이 실패하면 그 데이터로 정한 원래 입력 기준도 남기지 않음 (새 라벨처럼)"""
    X, y = _data()
    L = _L()
    wide = np.random.default_rng(0).random((12, 12001)).astype(np.float32)    # 기억 특징이 12,000개를 넘어 lda가 거부
    with pytest.raises(ValueError, match="메모리"):
        L.learn(wide, [0, 1] * 6)
    assert "default" not in L.inp and L.classes == []


@pytest.mark.parametrize("w", [0, 4])
def test_lda_same_result_for_any_data_order(w):
    """같은 데이터를 순서만 섞어 한 번에 배워도 같은 결과 (lda는 순서와 무관하다고 문서에 적힘). 예전: 처음 보정의 평균을
    시료 순서대로 더해 배율의 마지막 자리가 달라지고 (스파이크 몇 개가 바뀜), 시료가 256개를 넘으면 보정용 시료를 위치로 골라
    순서가 바뀌면 다른 시료로 보정 - MNIST 1만 개에서 시험 시료 9.1%의 예측이 바뀌었음"""
    X, y = _data(n=300)                                                  # 256개 넘게 (보정용 시료를 고르는 경우)
    p = np.random.default_rng(1).permutation(len(X))
    a = _L(input_weight=w).learn(X, y)
    b = _L(input_weight=w).learn(X[p], y[p])
    assert a.layer.gains == b.layer.gains
    col = [b.classes.index(c) for c in a.classes]                        # 점수 열 = 라벨을 처음 본 순서 (섞으면 바뀜)
    np.testing.assert_allclose(a.scores(X), b.scores(X)[:, col], rtol=1e-9, atol=1e-9)
    assert (a.predict(X) == b.predict(X)).all()


def test_connectome_model_calibration_same_for_any_data_order():
    """ConnectomeModel.prepare (fd.train의 자동 보정)도 데이터 순서와 무관 (예전: 256개 넘으면 위치로 골라 다른 시료로 보정 -
    순서만 섞어도 배율 11.07 대 10.72). pandas·리스트·torch 입력도 같은 결과"""
    import pandas as pd
    X, y = _data(n=600)
    p = np.random.default_rng(1).permutation(len(X))
    c = _c()
    gains = []
    for data in (X, X[p], pd.DataFrame(X[p]), X[p].tolist(), fd.Signal(X[p])):
        m = fd.ConnectomeModel(c, "IN", "O", 6, 4, device="cpu")
        m.prepare(data)
        gains.append(m.layer.gains)
    assert all(g == gains[0] for g in gains)


def test_input_stats_memory_stays_small():
    """원래 입력 기준·내용 순서를 잘라서 계산 (예전: 전체를 float64로 몇 번 복사 - MNIST 6만 장에서 2.9 GB 더 씀)"""
    import tracemalloc
    from flydnet.ganglion import backend as B
    from flydnet.learner import Learner
    X = np.random.default_rng(0).random((20000, 300)).astype(np.float32)
    F = np.random.default_rng(1).random((20000, 500)).astype(np.float32)
    L = Learner.__new__(Learner)
    L.input_weight = 4.0
    tracemalloc.start()
    B.content_order(X)
    L._input_stats(X, F)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 0.5 * (X.nbytes + F.nbytes), peak
    mu, var, msq = B.ordered_moments(X, B.content_order(X))              # 값은 numpy와 같음
    np.testing.assert_allclose(mu, X.astype(np.float64).mean(0), rtol=1e-12)
    np.testing.assert_allclose(var, X.astype(np.float64).var(0), rtol=1e-10)


def test_streaming_lda_memory_stays_small():
    """흐름 LDA의 학습·점수가 특징 전체를 float64로 복사하지 않음 (예전: 3만 x 2,000에서 학습 627 MB·점수 547 MB 더)"""
    import tracemalloc
    from flydnet.learner import _StreamingLDA
    F = np.random.default_rng(0).random((20000, 1000)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, 20000)
    L = _StreamingLDA(1000)
    tracemalloc.start()
    L.learn(F, y, 10)
    L.scores(F)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 0.6 * F.nbytes + 3 * 1000 * 1000 * 8, peak          # 공분산 몇 개 + 클래스 하나 정도
