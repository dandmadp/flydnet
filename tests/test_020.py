"""0.1.20: 발화하지 않은 뉴런 건너뛰기 (GPU spmm skip_silent) - 결과가 비트 단위로 같은지"""
import numpy as np
import pytest

import flydnet as fd
from flydnet.ganglion import backend as B
from flydnet.ganglion import kernels as K

gpu = pytest.mark.skipif(not B.gpu_available(), reason="GPU 없음")


@gpu
@pytest.mark.parametrize("nb", [1, 3, 7, 16, 30, 33, 64, 200])
def test_spmm_skip_silent_same_bits(nb):
    """건너뛰기 커널(작은 배치: 스레드별, 큰 배치: 투표)이 spmm_seg와 같은 비트 - 활동 비율·실수 값·긴 행 (조각 여럿)"""
    import cupy as cp
    from flydnet.ganglion.csr import CSR
    rng = np.random.default_rng(nb)
    n, E = 700, 30000
    post = np.sort(rng.integers(0, n, E))
    post[:300] = 5                                                       # 긴 행 하나 (SEG 64개짜리 조각 여럿)
    post = np.sort(post)
    pre = rng.integers(0, n, E)
    M = CSR.from_coo(rng.standard_normal(E).astype(np.float32), post, pre, (n, n)).to_cupy()
    for frac in (0.0, 0.001, 0.05, 0.5, 1.0):
        on = rng.random(n) < frac
        x = (rng.random((n, nb)) < 0.5) * on[:, None]
        for vals in (x.astype(np.float32), (x * rng.standard_normal((n, nb))).astype(np.float32)):
            xx = cp.asarray(vals)
            a, b = K.spmm(M, xx), K.spmm(M, xx, skip_silent=True)
            assert bool((a.view(cp.uint32) == b.view(cp.uint32)).all()), (nb, frac)


@gpu
def test_layer_skip_silent_same_output():
    """ConnectomeLayer: 직전 활동이 드물면 건너뛰기를 쓰는데, 출력·기울기가 쓰지 않을 때와 같은 비트"""
    import cupy as cp
    from test_threefactor import _rec
    c = _rec(feedback_edges=True)
    c = fd.Circuit(c.root_ids, c.groups, c.pre, c.post, c.weight * 4)    # 출력까지 신호가 가게 (test_learner와 같음)
    X = np.full((5, 6), 120.0, np.float32)
    outs = []
    for force in (1.0, 0.0):                                             # 1.0: 건너뛰지 않음, 0.0: 건너뜀
        L = fd.ConnectomeLayer(c, "IN", "O", t_ms=40, trainable=True, device="gpu")
        object.__setattr__(L, "_activity", cp.float32(force))
        M, _ = K.matrices(L.wiring, L.values().data)
        assert L._skip_silent(M) == (force == 0.0)
        object.__setattr__(L, "_activity", cp.float32(force))
        x = fd.Signal(X, device="gpu", plastic=True)
        y = L(x, seed=3)
        (y * y).sum().retrograde()
        assert float(y.data.max()) > 0                                     # 비교가 의미 있게 (출력이 0이 아님)
        outs.append([B.numpy(y.data), B.numpy(L.log_scale.retro), B.numpy(x.retro)])
    for a, b in zip(*outs):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@gpu
def test_skip_silent_off_for_nonfinite_weights():
    """연결 세기에 무한대가 있으면 건너뛰지 않음 (무한대 x 0 = NaN을 건너뛰면 결과가 달라짐)"""
    import cupy as cp
    from test_threefactor import _rec
    L = fd.ConnectomeLayer(_rec(), "IN", "O", t_ms=20, device="gpu")
    object.__setattr__(L, "_activity", cp.float32(0.0))
    M, _ = K.matrices(L.wiring, L.values().data)
    assert L._skip_silent(M)
    M.data[0] = cp.inf
    assert not L._skip_silent(M)


def test_doctor_survives_unreadable_nvidia_smi(monkeypatch):
    """nvidia-smi 출력을 못 읽어도 (한국어 윈도우에서 GPU 프로세스 경로의 한글 - 예전엔 cp949로 읽다 실패하면 출력이 None이
    되어 re.search에서 멈췄음) 드라이버 버전을 모름(None)으로 넘어감. 읽을 때 UTF-8로 지정되는지도"""
    import shutil
    import subprocess
    from flydnet import __main__ as M
    seen = {}

    class R:
        stdout = None

    def fake_run(*a, **k):
        seen.update(k)
        return R()
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert M._driver_cuda() is None
    assert seen.get("encoding") == "utf-8" and seen.get("errors") == "replace"


def test_old_torch_names_say_where_they_went():
    """0.1.17까지의 fd.torch.ConnectomeLayer 등은 옮긴 곳을 알려 줌 (예전: 그냥 '속성 없음')"""
    pytest.importorskip("torch")
    with pytest.raises(AttributeError, match=r"fd\.ConnectomeLayer"):
        fd.torch.ConnectomeLayer
    with pytest.raises(AttributeError, match=r"fd\.RateEncoder"):
        fd.torch.RateEncoder
    with pytest.raises(AttributeError, match="bridge"):
        fd.torch.SpikeFn
    with pytest.raises(AttributeError, match="no attribute"):
        fd.torch.nothing_here
    assert callable(fd.torch.bridge)


def test_interpret_flags_ladder_contradiction():
    """더 많은 구조를 유지한 대조군(shuffled)은 이기고 더 적은 대조군(randomized)은 못 이기면 (사다리와 모순 - 보통 잡음)
    해석이 그 사실을 먼저 알림. 예전: 모순을 말하지 않고 '중요한 구조: ...'로 단정"""
    from flydnet import controls as C
    real = np.array([0.80, 0.82, 0.81, 0.83, 0.80, 0.82])
    scores = {"real": real, "randomized": real + np.array([0.01, -0.02, 0.0, 0.01, -0.01, 0.0]), "shuffled": real - 0.05}
    rep = C.CompareReport(scores=scores, seeds=list(range(6)),
                          controls=[C._as_control("randomized"), C._as_control("shuffled")])
    lines = rep.interpret()
    assert "맞지 않음" in lines[0] and "randomized" in lines[0] and "seed" in lines[0]
    ok = C.CompareReport(scores={"real": real, "randomized": real - 0.05, "shuffled": real - 0.04}, seeds=list(range(6)),
                         controls=[C._as_control("randomized"), C._as_control("shuffled")])
    assert not any("맞지 않음" in s for s in ok.interpret())                # 순서대로면 경고 없음
