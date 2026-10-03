"""가소성 규칙: 역행성 신호(기울기)로 시냅스를 바꾸는 방법 (torch.optim에 해당)

  Plasticity(synapses, rate, momentum=0, decay=0)          ≈ optim.SGD
  AdaptivePlasticity(synapses, rate, betas, eps, decay)     ≈ optim.Adam / AdamW (decay는 AdamW처럼 따로)

  rule.step()     쌓인 .retro로 시냅스 갱신
  rule.clear()    .retro 지우기 (zero_grad)
"""
from __future__ import annotations

from . import backend as B
from .signal import quiescent


def _to_device(state, device):
    if isinstance(state, tuple):
        return tuple(B.to(a, device) for a in state)
    return B.to(state, device)


class _Rule:
    def __init__(self, synapses, rate: float):
        self.synapses = list(synapses)
        if not self.synapses:
            raise ValueError("바꿀 시냅스가 없음")
        self.rate = rate
        self.state = [None] * len(self.synapses)

    def clear(self):
        for s in self.synapses:
            s.retro = None

    def step(self):
        with quiescent():
            for i, s in enumerate(self.synapses):
                if s.retro is not None:
                    if self.state[i] is not None:                       # 시냅스가 다른 장치로 옮겨졌으면 상태도
                        self.state[i] = _to_device(self.state[i], s.device)
                    self._update(i, s, s.retro)

    def _update(self, i, s, g):
        raise NotImplementedError


class Plasticity(_Rule):
    """경사 하강 (SGD). momentum: 이전 변화의 관성, decay: 시냅스 크기를 줄이는 항상성 (weight decay)"""

    def __init__(self, synapses, rate: float = 0.01, momentum: float = 0.0, decay: float = 0.0):
        super().__init__(synapses, rate)
        self.momentum, self.decay = momentum, decay

    def _update(self, i, s, g):
        if self.decay:
            g = g + self.decay * s.data
        if self.momentum:
            v = g if self.state[i] is None else self.momentum * self.state[i] + g
            self.state[i] = v
            g = v
        s.data -= self.rate * g


class AdaptivePlasticity(_Rule):
    """적응형 가소성 (Adam): 시냅스마다 기울기 크기에 맞춰 변화량을 조절. decay는 AdamW처럼 따로 적용"""

    def __init__(self, synapses, rate: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, decay: float = 0.0):
        super().__init__(synapses, rate)
        self.b1, self.b2 = betas
        self.eps, self.decay = eps, decay
        self.t = [0] * len(self.synapses)

    def _update(self, i, s, g):
        xp = s.xp
        if self.state[i] is None:
            self.state[i] = (xp.zeros_like(s.data), xp.zeros_like(s.data))
        m, v = self.state[i]
        self.t[i] += 1
        m *= self.b1; m += (1 - self.b1) * g
        v *= self.b2; v += (1 - self.b2) * g * g
        mh = m / (1 - self.b1 ** self.t[i])
        vh = v / (1 - self.b2 ** self.t[i])
        if self.decay:
            s.data -= self.rate * self.decay * s.data
        s.data -= self.rate * mh / (xp.sqrt(vh) + self.eps)
