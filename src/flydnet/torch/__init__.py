"""flydnet.torch - torch 연동 (선택). torch가 설치되어 있어야 함 (pip install "flydnet[torch]")

flydnet의 계산은 자체 엔진 (flydnet.ganglion, NumPy·CuPy) 하나. torch 모델 안에서 flydnet 구조물을 쓸 때는 연결 장치:
    model = torch.nn.Sequential(..., fd.torch.bridge(fd.ConnectomeLayer(...), seed=0), ...)
계산은 자체 엔진 (원본 Brian2와 같은 계산, 전용 GPU 커널), 복사 없음, torch 역전파·옵티마이저와 이어짐.
0.1.17까지 있던 0.1의 torch판 복사본 (fd.torch.ConnectomeLayer 등)은 0.1.18에서 뺐음 - 같은 이름이 flydnet 최상위에 있음
"""
from .bridge import bridge, Bridge, to_engine, to_torch


def __getattr__(name):
    """0.1.17까지 있던 torch판 이름을 쓰면 어디로 옮겼는지 알림 (예전: 그냥 '속성 없음')"""
    import flydnet
    if name in ("SpikeFn", "SparsePropagate", "anatomy", "physiology", "layers"):
        raise AttributeError(f"flydnet.torch.{name}은 0.1.18에서 뺐음 (0.1의 torch판 복사본) - 자체 엔진 층을 "
                             "fd.torch.bridge(...)로 torch 모델 안에서 쓸 것")
    if not name.startswith("_") and hasattr(flydnet, name):
        raise AttributeError(f"flydnet.torch.{name}은 0.1.18에서 뺐음 - fd.{name}을 쓰고, torch 모델 안에서는 "
                             f"fd.torch.bridge(fd.{name}(...))")
    raise AttributeError(f"module 'flydnet.torch' has no attribute {name!r}")
