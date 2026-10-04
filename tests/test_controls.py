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


def test_interpret_control_wins():
    """대조군이 실제 배선보다 유의하게 좋으면 해석이 그것을 말해야 함 ("차이 없음"이라고 하면 안 됨)"""
    c = _circuit()

    def run(circ, seed):
        noise = np.random.default_rng(seed).normal(0, 0.003)
        return 0.8 + noise if "shuffled weights" in circ.name else 0.7 + noise
    rep = fd.compare(run, c, controls=["shuffled", "shuffled_weights"], seeds=6, verbose=False)
    text = rep.interpret()
    assert any("shuffled_weights가 실제 배선보다 좋음" in x for x in text), text
    assert not any("차이를 확인하지 못함" in x for x in text)


# ─────────────── 오류 처리 회귀 테스트 ───────────────
def test_invalid_labels_raise_clear_errors():
    import flydnet.ganglion as G
    X = np.random.rand(4, 3).astype(np.float32)
    for bad in ([0, 1, 2, 5], [0, 1, -1, 1]):                                  # 범위 밖, 음수 (음수는 조용히 틀렸음)
        with pytest.raises(ValueError, match="범위 밖"):
            G.surprise(X, bad)
        with pytest.raises(ValueError, match="범위 밖"):
            G.MushroomBodyOutput(3, 2, device="cpu").learn(X, bad)
        with pytest.raises(ValueError, match="범위 밖"):
            fd.AssocReadout(3, 2, device="cpu").fit(X, bad)
        with pytest.raises(ValueError, match="범위 밖"):
            fd.DopamineReadout(3, 2, device="cpu").fit(X, bad)
    with pytest.raises(ValueError, match="범위 밖"):
        fd.train_linear(X, [0, 1, 2, 3], X, [0, 1, 0, 1], n_classes=2, epochs=1, device="cpu")


def test_connectome_layer_rejects_nan_and_zero_steps():
    c = _circuit()
    layer = fd.ConnectomeLayer(c, "A", "B", t_ms=5, dt=0.5, device="cpu")
    x = np.ones((1, 20), np.float32); x[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        layer(x)
    with pytest.raises(ValueError, match="한 스텝도"):
        fd.ConnectomeLayer(c, "A", "B", t_ms=0.05, dt=0.1, device="cpu")


def test_compare_errors_have_context():
    c = _circuit()
    with pytest.raises(RuntimeError, match="'shuffled', seed 0"):
        fd.compare(lambda circ, s: 1 / 0 if "shuffled" in circ.name else 0.5, c, seeds=2, verbose=False)
    with pytest.raises(ValueError, match="nan"):
        fd.compare(lambda circ, s: float("nan"), c, seeds=2, verbose=False)
    with pytest.raises(RuntimeError, match="xy 모양"):
        fd.compare(lambda circ, s: 0.5, c, controls=[fd.controls.Local(np.zeros((3, 2)), 2)], seeds=2, verbose=False)


def test_loss_mode_has_no_accuracy_warnings():
    c = _circuit()
    rep = fd.compare(lambda circ, s: 1.5 + 0.01 * np.random.default_rng(s).random(), c, seeds=6,
                     higher_is_better=False, chance=2.0, verbose=False)
    assert not any("너무 쉬워" in w or "찍기" in w or "상한" in w for w in rep.warnings)


def test_extract_does_not_swallow_layer_errors():
    def layer(x):                                                              # seed를 받지 않는 층 안에서 TypeError
        raise TypeError("층 안의 진짜 버그")
    with pytest.raises(TypeError, match="진짜 버그"):
        fd.extract(layer, None, np.ones((2, 3)), batch=2)
    calls = []
    fd.extract(lambda x, seed: calls.append(seed) or x, None, np.ones((4, 3)), batch=2, seed=10)
    assert calls == [10, 12]


def test_console_output_survives_cp949_pipe():
    import subprocess, sys, os
    code = "import flydnet as fd; fd.data.CITATIONS['door'] = 'Münch — test'; from flydnet._console import say; say(fd.data.CITATIONS['door'])"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, env=dict(os.environ, PYTHONIOENCODING="cp949"))
    assert r.returncode == 0, r.stderr.decode("cp949", "replace")


def test_doctor_flags_colab_style_cupy_conflict(monkeypatch, capsys):
    """Colab 실제 사례: 드라이버 CUDA 13 + cupy-cuda12x와 cupy-cuda13x가 함께 설치 → 문제로 잡아야 함"""
    import flydnet.__main__ as cli
    from importlib import metadata

    class D:
        def __init__(self, name):
            self.metadata = {"Name": name}
    try:                                                           # torch는 처음 불릴 때 패키지 목록을 읽음 → 가짜 목록 전에
        import torch  # noqa: F401
    except ImportError:
        pass
    monkeypatch.setattr(cli, "_kernel_check", lambda: 0)           # 가짜 패키지 목록으로는 실제 컴파일 불가
    monkeypatch.setattr(cli, "_driver_cuda", lambda: "13.0")
    monkeypatch.setattr(metadata, "distributions", lambda: [D("cupy-cuda12x"), D("cupy-cuda13x"), D("numpy")])
    assert cli.doctor() == 1
    out = capsys.readouterr().out
    assert "CuPy가 2개" in out and "cupy-cuda12x" in out
    monkeypatch.setattr(metadata, "distributions", lambda: [D("cupy-cuda13x")])
    cli.doctor()
    assert "CuPy가" not in capsys.readouterr().out.split("설치된 CuPy")[1].split("\n", 1)[1]
    # 드라이버 업데이트 뒤 (CUDA 13 드라이버 + 예전 cupy-cuda12x): 하위 호환이라 문제 아님
    monkeypatch.setattr(metadata, "distributions", lambda: [D("cupy-cuda12x")])
    cli.doctor()
    assert "드라이버는 CUDA" not in capsys.readouterr().out
    # CuPy가 드라이버보다 새것 (CUDA 12 드라이버 + cupy-cuda13x): 안 돎
    monkeypatch.setattr(cli, "_driver_cuda", lambda: "12.4")
    monkeypatch.setattr(metadata, "distributions", lambda: [D("cupy-cuda13x")])
    cli.doctor()
    out = capsys.readouterr().out
    assert "드라이버는 CUDA 12" in out and "gpu-cuda12" in out
