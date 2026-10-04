"""안정성: 저장 파일 호환 (예전 파일 읽기, 새 형식·손상 파일 오류, 안전한 쓰기), 학습 중 NaN·발산·GPU 메모리 부족"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

import flydnet as fd
from flydnet import _archive
from flydnet.ganglion import backend as B

LEGACY = Path(__file__).parent / "data"


def _circuit():
    """tests/data/make_legacy.py와 같은 작은 회로"""
    rng = np.random.default_rng(0)
    N, n_in = 17, 5
    key = rng.choice(N * N, 120, replace=False)
    pre, post = key // N, key % N
    keep = pre != post
    w = (rng.integers(1, 6, keep.sum()) * rng.choice([-1, 1], keep.sum())).astype(np.float32)
    return fd.Circuit(np.arange(N), {"IN": np.arange(n_in), "OUT": np.arange(n_in, N)}, pre[keep], post[keep], w)


# ─────────────── 예전 버전이 저장한 파일 ───────────────
@pytest.mark.filterwarnings("ignore:.*모두 0")                    # 0.1.15 파일의 작은 입력 - 불러오기만 확인
@pytest.mark.filterwarnings("ignore:입력 최댓값이")                    # 0~1 입력 안내도
@pytest.mark.parametrize("ver", sorted(p.name for p in LEGACY.glob("v*")))
def test_legacy_files_still_load(ver):
    """예전 flydnet이 저장한 파일을 지금 버전이 읽고 같은 결과를 냄 (새 버전 파일은 make_legacy.py로 추가)"""
    d = LEGACY / ver
    inp = np.load(d / "inputs.npz")
    layer = fd.ConnectomeLayer.load(d / "layer.npz", device="cpu")
    np.testing.assert_allclose(layer(inp["x"], seed=0).numpy(), np.load(d / "layer_out.npy"), atol=1e-5)
    for cls, name in [(fd.DopamineReadout, "dopamine"), (fd.AssocReadout, "assoc")]:
        r = cls.load(d / f"{name}.npz", device="cpu")
        np.testing.assert_array_equal(r.predict(inp["X"]), np.load(d / f"{name}_pred.npy"))
    c = _circuit()
    t = fd.Pathway(fd.Projection(20, 5, seed=99, device="cpu"), fd.Neuropil(c, "IN", "OUT", device="cpu"))
    t.load(d / "tissue.npz")
    with np.load(d / "tissue.npz") as f:
        for k, s in t.named_synapses():
            np.testing.assert_array_equal(s.numpy(), f[k])


def test_save_records_format_and_roundtrips(tmp_path):
    t = fd.Pathway(fd.Projection(6, 4, seed=0, device="cpu"))
    path = t.save(tmp_path / "t")                                     # 확장자 없으면 .npz (np.savez와 같음)
    assert path.name == "t.npz"
    _, meta = _archive.read(path)
    assert meta["kind"] == "Pathway" and meta["format"] == _archive.FORMAT and meta["version"] == fd.__version__
    u = fd.Pathway(fd.Projection(6, 4, seed=5, device="cpu")).load(tmp_path / "t")
    np.testing.assert_array_equal(u.synapses()[0].numpy(), t.synapses()[0].numpy())


def test_newer_format_says_update(tmp_path):
    t = fd.Pathway(fd.Projection(6, 4, device="cpu"))
    arrays = dict(t.state())
    arrays["__flydnet__"] = np.array(json.dumps({"kind": "Pathway", "format": _archive.FORMAT + 1, "version": "9.9.9"}))
    np.savez(tmp_path / "new.npz", **arrays)
    with pytest.raises(ValueError, match=r"더 새 flydnet\(9\.9\.9\).*pip install -U flydnet"):
        t.load(tmp_path / "new.npz")


def test_wrong_kind_and_corrupt_files(tmp_path):
    r = fd.AssocReadout(5, 2, device="cpu")
    r.save(tmp_path / "a.npz")
    with pytest.raises(ValueError, match="DopamineReadout 파일이 아님.*AssocReadout"):
        fd.DopamineReadout.load(tmp_path / "a.npz")
    with pytest.raises(ValueError, match="ConnectomeLayer 파일이 아님"):
        fd.ConnectomeLayer.load(tmp_path / "a.npz")
    (tmp_path / "bad.npz").write_bytes(b"not a zip at all")
    with pytest.raises(ValueError, match="손상"):
        fd.AssocReadout.load(tmp_path / "bad.npz")
    full = (tmp_path / "a.npz").read_bytes()
    (tmp_path / "cut.npz").write_bytes(full[: len(full) // 2])          # 저장 도중 끊긴 파일
    with pytest.raises(ValueError, match="손상"):
        fd.AssocReadout.load(tmp_path / "cut.npz")
    with pytest.raises(FileNotFoundError, match="저장 파일이 없음"):
        fd.AssocReadout.load(tmp_path / "nope.npz")


def test_interrupted_save_keeps_old_file(tmp_path, monkeypatch):
    """저장 도중 오류가 나도 예전 파일은 그대로, 임시 파일은 남지 않음"""
    t = fd.Pathway(fd.Projection(6, 4, seed=0, device="cpu"))
    t.save(tmp_path / "m.npz")
    before = (tmp_path / "m.npz").read_bytes()

    def boom(f, **kw):
        f.write(b"partial")
        raise KeyboardInterrupt
    monkeypatch.setattr(_archive.np, "savez", boom)
    with pytest.raises(KeyboardInterrupt):
        t.save(tmp_path / "m.npz")
    assert (tmp_path / "m.npz").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m.npz"]


# ─────────────── 학습 중 NaN·발산 ───────────────
def _model():
    return fd.Pathway(fd.Projection(4, 8, seed=0, device="cpu"), fd.Activation("relu"),
                      fd.Projection(8, 3, seed=1, device="cpu"))


def test_nan_loss_stops_before_retrograde():
    m = _model()
    x = np.ones((2, 4), np.float32)
    x[0, 0] = np.nan
    loss = fd.surprise(m(fd.Signal(x)), np.array([0, 1]))
    with pytest.raises(FloatingPointError, match="손실이 nan.*clip"):
        loss.retrograde()


def test_nonfinite_retro_names_synapse_and_leaves_weights():
    m = _model()
    rule = fd.AdaptivePlasticity(m.named_synapses(), rate=0.1)
    loss = fd.surprise(m(fd.Signal(np.ones((2, 4), np.float32))), np.array([0, 1]))
    loss.retrograde()
    before = [s.numpy().copy() for s in m.synapses()]
    name, s = list(m.named_synapses())[2]
    s.retro[0, 0] = np.inf
    with pytest.raises(FloatingPointError, match=re.escape(f"{name} {tuple(s.shape)}")):
        rule.step()
    for b, s in zip(before, m.synapses()):
        np.testing.assert_array_equal(b, s.numpy())


def test_clip_limits_total_norm():
    """clip: 전체 L2 크기가 넘으면 줄여 적용 (SGD라 변화량 = rate × 줄인 기울기)"""
    m = _model()
    syn = m.synapses()
    rule = fd.Plasticity(syn, rate=1.0, clip=0.5)
    for s in syn:
        s.retro = np.full(s.shape, 10.0, np.float32)
    before = [s.numpy().copy() for s in syn]
    rule.step()
    delta = np.sqrt(sum(((b - s.numpy()) ** 2).sum() for b, s in zip(before, syn)))
    assert rule.last_norm == pytest.approx(10.0 * np.sqrt(sum(s.data.size for s in syn)), rel=1e-6)
    assert delta == pytest.approx(0.5, rel=1e-4)
    for s in syn:                                                      # 작으면 그대로
        s.retro = np.full(s.shape, 1e-3, np.float32)
    before = [s.numpy().copy() for s in syn]
    rule.step()
    np.testing.assert_allclose(before[0] - syn[0].numpy(), 1e-3, rtol=1e-4)
    with pytest.raises(ValueError, match="clip"):
        fd.Plasticity(syn, clip=0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")                    # 일부러 발산시킴
def test_clip_rescues_divergent_training():
    """학습률이 너무 크면 발산해 guard가 멈추고, clip을 쓰면 끝까지 유한"""
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((64, 4)) * 1e3).astype(np.float32)
    y = rng.integers(0, 3, 64)

    def train(clip):
        m = _model()
        rule = fd.Plasticity(m.synapses(), rate=50.0, clip=clip)
        for _ in range(30):
            loss = fd.surprise(m(fd.Signal(X)), y)
            rule.clear(); loss.retrograde(); rule.step()
        return m
    with pytest.raises(FloatingPointError):
        train(None)
    m = train(1.0)
    assert all(np.isfinite(s.numpy()).all() for s in m.synapses())


# ─────────────── GPU 메모리 부족 ───────────────
def test_gpu_oom_gets_hint():
    if not B.gpu_available():
        pytest.skip("GPU 없음")
    import cupy as cp
    pool = cp.get_default_memory_pool()
    old = pool.get_limit()
    m = fd.Pathway(fd.Projection(64, 64, device="gpu"))
    x = fd.Signal(cp.ones((4096, 64), cp.float32))
    try:
        pool.set_limit(size=pool.used_bytes() + 1024)
        with pytest.raises(B.GPUMemoryError, match="GPU 메모리 부족.*배치 줄이기") as e:
            m(x)
        assert isinstance(e.value.__cause__, cp.cuda.memory.OutOfMemoryError)
    finally:
        pool.set_limit(size=old)
    assert m(fd.Signal(cp.ones((4, 64), cp.float32))).shape == (4, 64)   # 상한을 풀면 다시 정상


def test_save_paths_home_and_new_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("USERPROFILE", str(tmp_path))
    p = fd.Projection(3, 2, device="cpu")
    out = p.save("~/models/sub/m")                                         # ~ 풀기 + 없는 폴더 만들기
    assert out == tmp_path / "models" / "sub" / "m.npz" and out.exists()
    fd.Projection(3, 2, seed=5, device="cpu").load("~/models/sub/m")
    assert not (tmp_path / "~").exists()
