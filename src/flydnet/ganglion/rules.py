"""가소성 규칙: 역행성 신호(기울기)로 시냅스를 바꾸는 방법 (torch.optim에 해당)

  Plasticity(synapses, rate, momentum=0, decay=0)          ≈ optim.SGD
  AdaptivePlasticity(synapses, rate, betas, eps, decay)     ≈ optim.Adam / AdamW (decay는 AdamW처럼 따로)

  rule.step()     쌓인 .retro로 시냅스 갱신
  rule.clear()    .retro 지우기 (zero_grad)

공통 옵션:
  clip=값         역행성 신호 전체 크기(L2)가 값을 넘으면 줄여서 적용 (clip_grad_norm_). rule.last_norm = 줄이기 전 크기
  guard=True      역행성 신호에 NaN·무한대가 있으면 시냅스를 바꾸기 전에 멈추고 어느 시냅스인지 알려 줌
synapses에 tissue.named_synapses()를 주면 오류 메시지에 이름이 나옴.
"""
from __future__ import annotations

import numpy as np

from . import backend as B
from .signal import quiescent


def _to_device(state, device):
    if isinstance(state, tuple):
        return tuple(B.to(a, device) for a in state)
    return B.to(state, device)


class _Rule:
    def __init__(self, synapses, rate: float, clip: float | None = None, guard: bool = True):
        items = list(synapses)
        if not items:
            raise ValueError("바꿀 시냅스가 없음")
        named = all(isinstance(it, tuple) and len(it) == 2 for it in items)      # named_synapses()
        self.names = [n for n, _ in items] if named else [f"#{i}" for i in range(len(items))]
        self.synapses = [s for _, s in items] if named else items
        if clip is not None and not clip > 0:
            raise ValueError(f"clip은 양수: {clip}")
        self.rate, self.clip, self.guard = rate, clip, guard
        self.last_norm = None
        self.state = [None] * len(self.synapses)

    def clear(self):
        for s in self.synapses:
            s.retro = None

    def _norm(self, live) -> float:
        """역행성 신호 전체 L2 크기 (float64로 더해 넘침 방지). NaN·무한대면 guard가 어느 시냅스인지 알려 줌"""
        sq = [float(s.xp.sum(s.retro.astype(s.xp.float64) ** 2)) for _, s in live]
        total = float(np.sqrt(sum(sq)))
        if self.guard and not np.isfinite(total):
            bad = [f"{self.names[i]} {tuple(s.shape)}" for (i, s), q in zip(live, sq) if not np.isfinite(q)]
            raise FloatingPointError(
                f"역행성 신호(기울기)에 NaN 또는 무한대: {', '.join(bad) or '합이 넘침'} - 시냅스는 바꾸지 않음. "
                f"학습률(rate={self.rate})을 낮추거나 clip=1.0 같은 값을 쓰고, 입력에 NaN·아주 큰 값이 없는지 확인")
        return total

    def step(self):
        live = [(i, s) for i, s in enumerate(self.synapses) if s.retro is not None]
        if not live:
            return
        with quiescent():
            scale = 1.0
            if self.guard or self.clip is not None:
                self.last_norm = self._norm(live)
                if self.clip is not None and self.last_norm > self.clip:
                    scale = self.clip / (self.last_norm + 1e-12)
            for i, s in live:
                if self.state[i] is not None:                           # 시냅스가 다른 장치로 옮겨졌으면 상태도
                    self.state[i] = _to_device(self.state[i], s.device)
                self._update(i, s, s.retro * scale if scale != 1.0 else s.retro)

    def _update(self, i, s, g):
        raise NotImplementedError


class Plasticity(_Rule):
    """경사 하강 (SGD). momentum: 이전 변화의 관성, decay: 시냅스 크기를 줄이는 항상성 (weight decay)"""

    def __init__(self, synapses, rate: float = 0.01, momentum: float = 0.0, decay: float = 0.0,
                 clip: float | None = None, guard: bool = True):
        super().__init__(synapses, rate, clip, guard)
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

    def __init__(self, synapses, rate: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, decay: float = 0.0,
                 clip: float | None = None, guard: bool = True):
        super().__init__(synapses, rate, clip, guard)
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
