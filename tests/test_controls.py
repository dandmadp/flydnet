"""fd.compare와 대조군"""
import numpy as np
import pytest

import flydnet as fd


def _circuit(seed=0, n_a=20, n_b=30, n_e=200):
    rng = np.random.default_rng(seed)
    key = rng.choice(n_a * n_b, n_e, replace=False)
    pre, post = key // n_b, key % n_b + n_a
    bb = rng.choice(n_b * n_b, 80, replace=False)
    bpre, bpost = bb // n_b + n_a, bb % n_b + n_a
    keep = bpre != bpost
    pre, post = np.r_[pre, bpre[keep]], np.r_[post, bpost[keep]]
    w = rng.integers(1, 9, len(pre)).astype(np.float32)
    return fd.Circuit(np.arange(n_a + n_b), {"A": np.arange(n_a), "B": np.arange(n_a, n_a + n_b)}, pre, post, w)


def _pairs(c):
    g = c.group_of()
    return g[c.pre] + ">" + g[c.post]


def test_randomized_keeps_pair_counts_and_weights_but_changes_degrees():
    c = _circuit()
    r = c.randomized(seed=1)
    assert np.array_equal(np.sort(_pairs(c)), np.sort(_pairs(r)))                # 그룹 쌍별 연결 수
    assert np.array_equal(r.weight, c.weight)                                     # 세기 목록 (같은 자리)
    assert len(np.unique(r.pre * 1000 + r.post)) == len(r.pre) and (r.pre != r.post).all()
    assert not np.array_equal(np.bincount(r.post, minlength=c.N), np.bincount(c.post, minlength=c.N))
    assert np.array_equal(c.randomized(seed=1).post, r.post)                       # seed 재현


def test_shuffled_weights_keeps_wiring():
    c = _circuit()
    s = c.shuffled_weights(seed=2)
    assert np.array_equal(s.pre, c.pre) and np.array_equal(s.post, c.post)
    for k in np.unique(_pairs(c)):
        m = _pairs(c) == k
        assert np.array_equal(np.sort(s.weight[m]), np.sort(c.weight[m]))
    assert not np.array_equal(s.weight, c.weight)


def test_sign_flip_p():
    assert fd.sign_flip_p(np.ones(6)) == pytest.approx(2 / 64)                     # 정확한 최소 p
    assert fd.sign_flip_p(np.zeros(5)) == 1.0
    assert fd.sign_flip_p(np.array([1.0, -1.0, 1.0, -1.0])) == 1.0
    assert fd.sign_flip_p(np.ones(20) + np.random.default_rng(0).normal(0, 0.1, 20)) < 0.001


def _run_factory(effect):
    def run(c, seed):
        rng = np.random.default_rng(seed)
        bonus = effect if c.name.endswith(")") or "[" not in c.name else 0.0    # 실제 회로만 보너스
        return 0.5 + bonus + rng.normal(0, 0.02) + 0.01 * rng.random()
    return run


def test_compare_detects_effect_and_reports():
    c = _circuit()
    rep = fd.compare(_run_factory(0.1), c, controls=["shuffled", "randomized"], seeds=6, verbose=False)
    t = rep.table()
    assert set(t.index) == {"real", "shuffled", "randomized"}
    assert t.loc["shuffled", "p"] < 0.05 and t.loc["shuffled", "real_wins"] == "6/6"
    assert rep.verdict("shuffled") == "실제 배선이 더 좋음"
    assert any("Local" in w for w in rep.warnings)                                # 큰 구조 경고
    assert "real (실제 배선)" in str(rep) and rep.to_dict()["controls"] == ["shuffled", "randomized"]


def test_compare_no_effect_and_warnings():
    c = _circuit()
    rep = fd.compare(_run_factory(0.0), c, controls=["shuffled"], seeds=2, chance=0.6, verbose=False)
    assert rep.verdict("shuffled") == "차이를 확인하지 못함"
    text = " ".join(rep.warnings)
    assert "seed 2개로는 p가" in text and "찍기 수준" in text
    easy = fd.compare(lambda c, s: 1.0, c, controls=["shuffled"], seeds=3, verbose=False)
    assert any("너무 쉬워" in w for w in easy.warnings) and any("seed마다 똑같음" in w for w in easy.warnings)
    calls = iter(range(100))
    flaky = fd.compare(lambda c, s: next(calls) * 0.001, c, controls=["shuffled"], seeds=3, verbose=False)
    assert any("재현되지 않음" in w for w in flaky.warnings)


def test_compare_controls_spec():
    c = _circuit()
    xy = np.random.default_rng(0).random((c.N, 2)) * 10
    rep = fd.compare(_run_factory(0.0), c, seeds=3, verbose=False,
                     controls=[fd.controls.Local(xy, radius=5), lambda circ, s: circ.shuffled(seed=s + 100),
                               fd.controls.Shuffled(pairs=["A>B"], name="shuffled A>B")])
    assert list(rep.scores) == ["real", "local(r=5)", "<lambda>", "shuffled A>B"]
    with pytest.raises(ValueError):
        fd.compare(_run_factory(0), c, controls=["local"], seeds=2, verbose=False)
    with pytest.raises(ValueError):
        fd.compare(_run_factory(0), c, controls=["shuffled", "shuffled"], seeds=2, verbose=False)


def test_near_ceiling_warning():
    c = _circuit()
    rep = fd.compare(lambda circ, s: 0.97 + 0.01 * np.random.default_rng(s).random(), c, controls=["shuffled"],
                     seeds=5, verbose=False)
    assert any("상한" in w and "가려졌을" in w for w in rep.warnings)


def test_interpret_ladder():
    """randomized만 지고 shuffled는 같으면 → 차수 / shuffled도 지고 local은 같으면 → 위치 / local도 지면 → 세부 배선"""
    c = _circuit()
    xy = np.random.default_rng(0).random((c.N, 2)) * 10
    controls = ["randomized", "shuffled", fd.controls.Local(xy, radius=5)]

    def make(bonus):                                                            # 조건별 보너스 (실제 배선 기준 손실)
        def run(circ, seed):
            noise = np.random.default_rng(seed).normal(0, 0.003)
            for key, loss in bonus.items():
                if key in circ.name:
                    return 0.7 - loss + noise
            return 0.7 + noise
        return run
    cases = [({"randomized": 0.1}, "뉴런별 연결 수 분포"),
             ({"randomized": 0.1, "shuffled]": 0.1}, "시야 위치 대응"),
             ({"randomized": 0.1, "shuffled": 0.1}, "세부 배선")]                # "shuffled"는 local 이름에도 들어감
    for bonus, expect in cases:
        rep = fd.compare(make(bonus), c, controls=controls, seeds=6, verbose=False)
        assert any(expect in x for x in rep.interpret()), (bonus, rep.interpret())
