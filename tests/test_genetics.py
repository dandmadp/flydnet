"""가상 유전학 (fd.genetics): 드라이버 고르기·조합, silence / block / activate / ablate, 스크린"""
import re

import numpy as np
import pandas as pd
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from flydnet.genetics import activate, ablate, block, driver, lines, screen, silence


def _chain(n_side=6):
    """A(입력) → H(중간) → O(출력) 흥분성 사슬 + 주석"""
    N = 3 * n_side
    A, H, O = np.arange(n_side), np.arange(n_side, 2 * n_side), np.arange(2 * n_side, N)
    pre = np.concatenate([np.repeat(A, n_side), np.repeat(H, n_side)])
    post = np.concatenate([np.tile(H, n_side), np.tile(O, n_side)])
    w = np.full(len(pre), 30.0, np.float32)
    meta = pd.DataFrame({"root_id": np.arange(N) + 1000,
                         "cell_type": ["a"] * n_side + ["h1"] * (n_side // 2) + ["h2"] * (n_side - n_side // 2) + ["o"] * n_side,
                         "side": (["left", "right"] * N)[:N]})
    return fd.Circuit(np.arange(N) + 1000, {"A": A, "H": H, "O": O}, pre, post, w, meta=meta)


def _layer(c, **kw):
    kw = dict(dict(t_ms=200, device="cpu"), **kw)
    return fd.ConnectomeLayer(c, kw.pop("inputs", "A"), kw.pop("outputs", ("H", "O")), **kw)


# ─────────────── 드라이버 ───────────────
def test_driver_selects_and_combines():
    c = _chain()
    h = driver(c, group="H")
    assert h.idx.tolist() == list(range(6, 12))
    h1 = driver(c, cell_type="h1")
    assert h1.idx.tolist() == [6, 7, 8]
    assert driver(c, cell_type=["h1", "h2"]).idx.tolist() == h.idx.tolist()
    assert driver(c, cell_type=re.compile("h.")).idx.tolist() == h.idx.tolist()
    left = driver(c, side="left")
    assert (h & left).idx.tolist() == [6, 8, 10]                          # split-GAL4
    assert len(h | driver(c, group="O")) == 12 and (h - h1).idx.tolist() == [9, 10, 11]
    assert driver(c, cell_type="h1", side="left").idx.tolist() == [6, 8]     # 조건 여러 개 = 교집합
    assert driver(c, root_ids=[1006, 1007]).idx.tolist() == [6, 7]
    np.testing.assert_array_equal(h1.root_ids, [1006, 1007, 1008])


def test_driver_errors():
    c = _chain()
    with pytest.raises(KeyError, match="회로에 없는 뉴런 ID 1개"):
        driver(c, root_ids=[1006, 99])
    with pytest.warns(UserWarning, match="회로에 없는 뉴런 ID"):
        assert driver(c, root_ids=[1006, 99], missing="warn").idx.tolist() == [6]
    with pytest.raises(ValueError, match="맞는 뉴런이 없음"):
        driver(c, cell_type="zzz")
    with pytest.raises(KeyError, match="주석 열이 없음"):
        driver(c, nerve="PhN")
    with pytest.raises(KeyError, match="그룹"):
        driver(c, group="Q")
    other = _chain(4)
    with pytest.raises(ValueError, match="다른 회로"):
        driver(c, group="H") & driver(other, group="H")
    with pytest.raises(ValueError, match="다른 회로"):
        silence(_layer(c), driver(other, group="H"))


def test_lines_collection():
    c = _chain()
    ls = lines(c, "cell_type")
    assert set(ls) == {"a", "h1", "h2", "o"} and len(ls["h2"]) == 3
    assert set(lines(c, "cell_type", within=driver(c, group="H"))) == {"h1", "h2"}


# ─────────────── 효과기 ───────────────
def test_silence_and_block():
    c = _chain()
    L = _layer(c, outputs=("H", "O"))
    x = np.full((2, 6), 150.0, np.float32)
    base = L(x, seed=0)
    assert base[:, :6].numpy().mean() > 5 and base[:, 6:].numpy().mean() > 5
    h = driver(c, group="H")
    with silence(L, h):
        s = L(x, seed=0).numpy()
    assert (s == 0).all()                                                # H 끄면 H도, 그 아래 O도 조용
    with block(L, h):
        b = L(x, seed=0).numpy()
    np.testing.assert_allclose(b[:, :6], base[:, :6].numpy())            # H는 그대로 발화
    assert (b[:, 6:] == 0).all()                                         # 전달만 막힘
    np.testing.assert_allclose(L(x, seed=0).numpy(), base.numpy())       # with 밖에서는 원래대로
    e = silence(L, driver(c, cell_type="h1"))                            # remove()로 끄는 방식
    part = L(x, seed=0).numpy()
    e.remove()
    assert (part[:, :3] == 0).all() and part[:, 3:6].mean() > 5


def test_activate_drives_at_rate_without_inputs():
    """입력 그룹 없이 활성화만: 활성화한 뉴런이 hz로 발화 (불응기 없음, 자극 하나 = 스파이크 하나), 나머지는 회로를 따라"""
    c = _chain()
    L = _layer(c, inputs=None, outputs=("A", "H", "O"), t_ms=1000)
    assert L.n_in == 0 and (L(None, seed=0, batch=3).numpy() == 0).all()
    with activate(L, driver(c, group="A"), hz=100):
        r = L(None, seed=0, batch=3).numpy()
    assert 90 < r[:, :6].mean() < 110
    assert r[:, 6:].mean() > 5
    with activate(L, driver(c, group="A"), hz=40):
        r40 = L(None, seed=0, batch=3).numpy()
    assert 30 < r40[:, :6].mean() < 45                                   # 자극 빈도를 따름


def test_activate_rules():
    c = _chain()
    L = _layer(c)
    with pytest.raises(ValueError, match="입력 그룹 뉴런"):
        activate(L, driver(c, group="A"))
    with activate(L, driver(c, group="H")):
        with pytest.raises(ValueError, match="이미 활성화"):
            activate(L, driver(c, cell_type="h1"))
    with pytest.raises(ValueError, match="hz"):
        activate(L, driver(c, group="H"), hz=0)


def test_activate_graded_clamps_level():
    c = _chain()
    L = _layer(c, inputs=None, outputs=("H",), neuron="graded", t_ms=50)
    with activate(L, driver(c, cell_type="h1"), level=0.7):
        r = L(None, batch=2).numpy()
    np.testing.assert_allclose(r[:, :3], 0.7, rtol=1e-5)


def test_effects_with_checkpointing_and_gradients():
    """효과기를 켜도 체크포인팅 결과가 같고 역전파가 됨"""
    c = _chain()
    x = fd.Signal(np.full((2, 6), 120.0, np.float32), plastic=True)
    outs = []
    for ce in (None, 7):
        L = _layer(c, inputs="A", outputs=("O",), trainable=True, checkpoint_every=ce, t_ms=60)
        with activate(L, driver(c, cell_type="h2"), hz=300), silence(L, driver(c, cell_type="h1")):
            out = L(x, seed=3)
            out.sum().retrograde()
        outs.append((out.numpy(), L.log_scale.retro.copy()))
        x.retro = None
    np.testing.assert_allclose(outs[0][0], outs[1][0])
    np.testing.assert_allclose(outs[0][1], outs[1][1], rtol=1e-4, atol=1e-6)
    assert np.abs(outs[0][1]).sum() > 0


def test_effects_cpu_gpu_equal():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    c = _chain()
    res = []
    for dev in ("cpu", "gpu"):
        L = _layer(c, inputs=None, outputs=("A", "H", "O"), device=dev)
        with activate(L, driver(c, group="A"), hz=150), block(L, driver(c, cell_type="h1")):
            res.append(L(None, seed=1, batch=2).numpy())
    np.testing.assert_allclose(res[0], res[1])


def test_ablate_removes_edges():
    c = _chain()
    h1 = driver(c, cell_type="h1")
    a = ablate(c, h1)
    assert a.N == c.N and not np.isin(a.pre, h1.idx).any() and not np.isin(a.post, h1.idx).any()
    assert a.n_edges == c.n_edges - 2 * 3 * 6
    L = _layer(a, outputs=("H",))
    r = L(np.full((1, 6), 150.0, np.float32), seed=0).numpy()
    assert (r[:, :3] == 0).all() and r[:, 3:].mean() > 5


def test_screen_table():
    c = _chain()
    L = _layer(c, outputs=("O",), t_ms=100)
    x = np.full((2, 6), 150.0, np.float32)
    df = screen(lambda layer, s: float(layer(x, seed=s).numpy().mean()), L,
                lines(c, "cell_type", within=driver(c, group="H")), seeds=4, verbose=False)
    assert list(df.line) and set(df.columns) >= {"line", "n", "baseline", "manipulated", "change", "p"}
    assert (df.change < 0).all()                                          # 중간 뉴런을 끄면 출력 감소
    assert df.change.abs().is_monotonic_decreasing


def test_screen_warns_and_corrects():
    c = _chain()
    L = _layer(c, outputs=("O",), t_ms=60)
    x = np.full((1, 6), 150.0, np.float32)
    m = lambda layer, s: float(layer(x, seed=s).numpy().mean())
    with pytest.warns(UserWarning, match="seed 3개로는 p가 0.25"):
        df = screen(m, L, lines(c, "cell_type", within=driver(c, group="H")), seeds=3, verbose=False)
    assert df.attrs["min_p"] == 0.25 and (df.p_holm >= df.p).all()
    with silence(L, driver(c, cell_type="h1")):
        with pytest.warns(UserWarning, match="이미 켜진 효과기"):
            screen(m, L, {"h2": driver(c, cell_type="h2")}, seeds=6, verbose=False)


def test_active_effects_visible_and_clearable():
    c = _chain()
    L = _layer(c)
    silence(L, driver(c, cell_type="h1"))                                  # with 없이 켜 둠
    block(L, driver(c, cell_type="h2"))
    assert "켜진 효과기" in repr(L) and "silence" in repr(L) and len(fd.genetics.active(L)) == 2
    fd.genetics.clear(L)
    assert not fd.genetics.active(L) and "켜진 효과기" not in repr(L)
