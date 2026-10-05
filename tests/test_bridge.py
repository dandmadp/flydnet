"""fd.torch.bridge: 자체 엔진 구조물을 torch 모델 안에서 - 같은 출력·기울기, 메모리 공유, no_grad, GPU 복사 없음"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import flydnet as fd
from flydnet.ganglion import backend as B
from test_genetics import _chain


def _layer(device="cpu", neuron="graded"):
    c = _chain()
    kw = dict(t_ms=30, trainable=True, device=device, neuron=neuron)
    return c, fd.ConnectomeLayer(c, "A", ("O",), **kw)


def _engine_grads(layer, x, w):
    """자체 엔진만으로: 출력, 입력 기울기, 학습 값 기울기"""
    xs = fd.Signal(x, plastic=True, device=layer.device)
    out = layer(xs, seed=0)
    (out * fd.Signal(w, device=layer.device)).sum().retrograde()
    g = B.numpy(layer.log_scale.retro).copy()
    layer.log_scale.retro = None
    return out.numpy(), B.numpy(xs.retro), g


@pytest.mark.parametrize("neuron", ["graded", "lif"])
def test_bridge_matches_engine(neuron):
    _, layer = _layer(neuron=neuron)
    x = np.random.default_rng(0).uniform(0.2, 1, (3, 6)).astype(np.float32) * (150 if neuron == "lif" else 1)
    w = np.random.default_rng(1).standard_normal((3, 6)).astype(np.float32)
    out_e, gx_e, gp_e = _engine_grads(layer, x, w)
    m = fd.torch.bridge(layer, seed=0)
    xt = torch.tensor(x, requires_grad=True)
    out_t = m(xt)
    (out_t * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(out_t.detach().numpy(), out_e, rtol=1e-6)
    np.testing.assert_allclose(xt.grad.numpy(), gx_e, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(m.log_scale.grad.numpy(), gp_e, rtol=1e-5, atol=1e-7)
    assert layer.log_scale.retro is None                                   # 자체 엔진 쪽에 남기지 않음


def test_bridge_shares_memory_with_torch_optimizer():
    _, layer = _layer()
    m = fd.torch.bridge(layer, seed=0)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    x = torch.rand(4, 6)
    before = layer.log_scale.numpy().copy()
    loss = m(x).pow(2).sum()
    loss.backward(); opt.step()
    assert not np.allclose(layer.log_scale.numpy(), before)               # torch가 바꾼 값이 자체 엔진에도
    np.testing.assert_array_equal(layer.log_scale.numpy(), m.log_scale.detach().numpy())
    assert [n for n, _ in m.named_parameters()] == ["log_scale"]


def test_bridge_inside_sequential_and_no_grad():
    _, layer = _layer()
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Sigmoid(), fd.torch.bridge(layer, seed=0),
                                torch.nn.Linear(6, 3))
    x, y = torch.rand(8, 4), torch.randint(0, 3, (8,))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert model[0].weight.grad is not None                                # 앞쪽 torch 층까지 기울기가 이어짐
    with torch.no_grad():
        out = model(x)
    assert not out.requires_grad
    out[:] = 0                                                              # 출력은 복사본 → 자체 엔진은 안전
    model(x).sum().backward()


def test_bridge_resyncs_after_engine_replaces_arrays(tmp_path):
    _, layer = _layer()
    m = fd.torch.bridge(layer, seed=0)
    layer.log_scale.data = layer.log_scale.data + 1.0                       # 자체 엔진에서 배열을 바꿔 끼움
    with pytest.warns(UserWarning, match="옵티마이저를"):                   # 옛 옵티마이저는 쓰이지 않는 값을 갱신
        m(torch.rand(2, 6))
    np.testing.assert_array_equal(m.log_scale.detach().numpy(), layer.log_scale.numpy())


def test_bridge_errors():
    _, layer = _layer()
    m = fd.torch.bridge(layer, seed=0, record=[0])
    with pytest.raises(TypeError, match="여러 값"):
        m(torch.rand(2, 6, requires_grad=True))
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="cpu"):
            fd.torch.bridge(layer, seed=0)(torch.rand(2, 6, device="cuda"))


def test_bridge_gpu_zero_copy():
    if not (B.gpu_available() and torch.cuda.is_available()):
        pytest.skip("GPU 없음")
    _, layer = _layer(device="gpu", neuron="lif")
    m = fd.torch.bridge(layer, seed=0)
    assert m.log_scale.data_ptr() == int(layer.log_scale.data.data.ptr)    # 같은 GPU 메모리
    x = (torch.rand(3, 6, device="cuda") * 150).requires_grad_()
    out = m(x)
    assert out.is_cuda
    out.sum().backward()
    assert x.grad.is_cuda and m.log_scale.grad.is_cuda
    _, cpu_layer = _layer(device="cpu", neuron="lif")
    o2 = fd.torch.bridge(cpu_layer, seed=0)(x.detach().cpu())
    np.testing.assert_allclose(out.detach().cpu().numpy(), o2.detach().numpy())
