"""torch 연결 장치: flydnet 자체 엔진의 구조물(Tissue)을 torch 모델 안에서 nn.Module처럼

  import flydnet as fd
  layer = fd.ConnectomeLayer(mb, "PN", "MBON", trainable=True)            # 자체 엔진 (원본 모델과 같은 계산)
  model = torch.nn.Sequential(torch.nn.Linear(20, 344), torch.nn.ReLU(),
                              fd.torch.bridge(layer, seed=0),              # ← 계산은 자체 엔진, 겉은 nn.Module
                              torch.nn.Linear(48, 10))
  opt = torch.optim.Adam(model.parameters())                              # 커넥톰의 학습 값도 torch가 갱신
  loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()

  - 계산은 전부 자체 엔진 (timing="brian", 직접 작성한 CUDA 커널, genetics·explain 그대로)
  - 복사 없음: GPU는 DLPack, CPU는 numpy와 메모리 공유. 구조물의 학습 값(Synapse)이 torch Parameter와 같은 메모리라
    torch 옵티마이저가 바꾸면 자체 엔진도 바뀐 값을 씀
  - 역전파: torch backward() → 자체 엔진 retrograde() → 입력·학습 값의 기울기를 torch로
  - torch.no_grad()면 자체 엔진도 quiescent (경로를 만들지 않음)
  - 장치: 구조물이 gpu면 cuda 텐서, cpu면 cpu 텐서를 넣을 것

"""
from __future__ import annotations

import numpy as np
import torch

from ..ganglion import backend as B
from ..ganglion.signal import Signal, quiescent


def to_engine(t: torch.Tensor, device: str):
    """torch 텐서 → 자체 엔진 배열 (복사 없이). device = 구조물의 장치"""
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16) or not (t.is_floating_point() or t.dtype == torch.float64):
        t = t.float()                                          # bfloat16·float16·정수·불리언 → float32 (CuPy·numpy가 받는 형태)
    if device == "gpu":
        if not t.is_cuda:
            raise ValueError("구조물이 gpu에 있음 - 입력도 cuda 텐서로 (x.cuda())")
        import cupy as cp
        return cp.from_dlpack(t.contiguous())
    if t.is_cuda:
        raise ValueError("구조물이 cpu에 있음 - 입력도 cpu 텐서로 (또는 구조물을 .to('gpu'))")
    return t.contiguous().numpy()


def to_torch(a) -> torch.Tensor:
    """자체 엔진 배열(numpy·cupy) 또는 Signal → torch 텐서 (복사 없이)"""
    if isinstance(a, Signal):
        a = a.data
    if B.device_of(a) == "gpu":
        return torch.from_dlpack(a)
    return torch.from_numpy(np.ascontiguousarray(a))


class _Run(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, n_in, *tensors):
        inputs, params = tensors[:n_in], tensors[n_in:]
        dev = owner.tissue.device
        sig = [None if t is None else Signal(to_engine(t, dev), plastic=t.requires_grad) for t in inputs]
        for s in owner.synapses:
            s.retro = None
        out = owner.tissue(*sig, **owner.kwargs)
        if isinstance(out, tuple):
            raise TypeError("구조물이 여러 값을 돌려줌 - 연결 장치는 Signal 하나만 (record= 등은 빼고)")
        if not isinstance(out, Signal):
            raise TypeError(f"구조물의 출력이 Signal이 아님: {type(out).__name__}")
        ctx.state = (out, sig, owner)
        return to_torch(out.data).clone()            # 복사본: torch의 제자리 연산이 자체 엔진의 역전파 값을 건드리지 않게

    @staticmethod
    def backward(ctx, grad):
        out, sig, owner = ctx.state
        ctx.state = None
        dev = owner.tissue.device
        if out.plastic:
            out.retrograde(to_engine(grad, dev).astype(out.data.dtype, copy=False))
        g_in = [to_torch(s.retro).clone() if (s is not None and s.plastic and s.retro is not None) else None for s in sig]
        g_par = []
        for syn in owner.synapses:
            g_par.append(to_torch(syn.retro).clone() if syn.retro is not None else None)
            syn.retro = None
        return (None, None, *g_in, *g_par)


class Bridge(torch.nn.Module):
    """자체 엔진 구조물을 감싼 torch 모듈. 학습 값은 torch Parameter (자체 엔진과 같은 메모리)"""

    def __init__(self, tissue, **kwargs):
        super().__init__()
        self.tissue, self.kwargs = tissue, kwargs
        self._wrap()

    @property
    def synapses(self):
        return [s for _, s in self.tissue.named_synapses()]

    def _wrap(self):
        """Synapse 배열 → 같은 메모리의 torch Parameter (이름의 '.'은 '__'로)"""
        for n in getattr(self, "_names", []):                           # 예전 이름의 Parameter는 지움 (학습 값이 없어졌을 수도)
            self._parameters.pop(n.replace(".", "__"), None)
        self._names = [n for n, _ in self.tissue.named_synapses()]       # 감싼 뒤 학습 값이 생기거나 없어져도 (확장을 붙이는 등)
        self._ptrs = []
        for n, s in zip(self._names, self.synapses):
            if not s.data.flags.c_contiguous:
                s.data = B.xp(s.device).ascontiguousarray(s.data)
            self.register_parameter(n.replace(".", "__"), torch.nn.Parameter(to_torch(s.data), requires_grad=True))
            self._ptrs.append(_ptr(s.data))

    def _sync(self):
        """자체 엔진 쪽에서 배열을 바꿔 끼웠으면 (.to로 장치를 옮김 등) Parameter를 다시 묶음 - 이미 만든 torch 옵티마이저는
        옛 Parameter를 갱신하므로 다시 만들어야 함 (안 그러면 학습이 모델에 반영되지 않음)"""
        names = [n for n, _ in self.tissue.named_synapses()]
        if names != self._names or [_ptr(s.data) for s in self.synapses] != self._ptrs:
            import warnings
            warnings.warn("자체 엔진 구조물의 학습 값 배열이 바뀌어(.to 등) torch Parameter를 새로 만듦 - 옵티마이저를 "
                          "model.parameters()로 다시 만들 것 (옛 옵티마이저는 쓰이지 않는 값을 갱신함)", stacklevel=4)
            self._wrap()

    def forward(self, *inputs):
        self._sync()                                                     # 학습 값 목록을 먼저 맞춤 (autograd 함수에 넘길 개수가
        params = [self._parameters[n.replace(".", "__")] for n in self._names]   # 기울기 개수와 같아야 함)
        if not torch.is_grad_enabled():
            dev = self.tissue.device
            with quiescent():
                out = self.tissue(*[None if t is None else Signal(to_engine(t, dev)) for t in inputs], **self.kwargs)
            return to_torch(out.data).clone()
        return _Run.apply(self, len(inputs), *inputs, *params)

    def extra_repr(self):
        return f"{type(self.tissue).__name__} (자체 엔진, 장치 {self.tissue.device}), 인자 {self.kwargs}"


def _ptr(a) -> int:
    return int(a.data.ptr) if B.device_of(a) == "gpu" else int(a.ctypes.data)


def bridge(tissue, **kwargs) -> Bridge:
    """자체 엔진 구조물(ConnectomeLayer, Pathway, Neuropil 등)을 torch 모델 안에서 쓰는 nn.Module로.
    kwargs는 매번 구조물에 넘김 (예: seed=0, return_all=True)"""
    return Bridge(tissue, **kwargs)
