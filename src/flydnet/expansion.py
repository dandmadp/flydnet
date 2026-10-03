"""실제 PN→KC 배선을 한 번에 계산하는 빠른 확장 층 (스파이크 시뮬레이션 없음)

  kc = fd.KCExpansion(fd.Circuit.from_flywire(), n_in=512)    # 특징 512개 → KC 2,597개
  codes = kc(features)                                         # (B, 2597), KC 5%만 켜짐

버섯체가 하는 일을 앞먹임 계산으로 줄인 것 (FlyHash, Dasgupta et al. 2017과 같은 구조, 단 배선은 실제 FlyWire):
  특징 ─(고정 희소 투영)─▶ PN 활동 ─(평균 빼기: 촉각엽의 측억제)─▶ ─(실제 PN→KC 시냅스 수)─▶ KC 입력
       ─(상위 k_frac만 남김: APL 억제)─▶ KC 코드
ConnectomeLayer(LIF)보다 수천 배 빠르고 같은 입력에 항상 같은 출력 → 연속 학습 리드아웃(AssocReadout) 앞에 쓰기 좋음.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .circuit import Circuit


class KCExpansion(nn.Module):
    """특징 (B, n_in) → KC 코드 (B, n_kc)

    circuit:   PN·KC 그룹이 있는 회로 (Circuit.from_flywire(), 무작위 대조군은 circuit.shuffled())
    n_in:      입력 특징 수. None이면 입력이 이미 PN 활동 (B, n_pn)
    projection: 특징 → PN 고정 투영. "sparse" = PN마다 특징 k_in개 평균 (음수 없음, 사구체처럼) /
               "gaussian" = 부호 있는 밀집 무작위 (특징이 많을 때 정보 손실이 적음)
    k_in:      sparse 투영에서 PN 하나가 모으는 특징 수
    k_frac:    켜 둘 KC 비율 (실제 초파리는 약 5~10%)
    center:    PN 활동에서 샘플별 평균을 뺌 (없으면 모든 입력에 같은 KC가 켜지기 쉬움)
    binary:    True = 켜진 KC는 1 / False = 켜진 KC는 입력 세기 그대로
    """

    def __init__(self, circuit: Circuit, n_in: int | None = None, pre: str = "PN", post: str = "KC",
                 k_in: int = 20, k_frac: float = 0.05, center: bool = True, binary: bool = False,
                 projection: str = "sparse", seed: int = 0, device: str | None = None):
        super().__init__()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        P, K = circuit.groups[pre], circuit.groups[post]
        lp = np.full(circuit.N, -1, np.int64); lp[P] = np.arange(len(P))
        lk = np.full(circuit.N, -1, np.int64); lk[K] = np.arange(len(K))
        m = (lp[circuit.pre] >= 0) & (lk[circuit.post] >= 0) & (circuit.weight > 0)
        W = torch.zeros(len(K), len(P))
        W.index_put_((torch.tensor(lk[circuit.post[m]]), torch.tensor(lp[circuit.pre[m]])),
                     torch.tensor(circuit.weight[m], dtype=torch.float32), accumulate=True)
        self.register_buffer("W", W.to(dev))                       # (n_kc, n_pn) 시냅스 수
        self.n_pn, self.n_out = len(P), len(K)
        self.n_in = n_in if n_in is not None else len(P)
        self.k = max(1, int(round(k_frac * len(K))))
        self.center, self.binary = center, binary
        if n_in is not None:
            g = torch.Generator().manual_seed(seed)
            if projection == "sparse":
                k_in = min(k_in, n_in)
                cols = torch.stack([torch.randperm(n_in, generator=g)[:k_in] for _ in range(len(P))])
                proj = torch.zeros(len(P), n_in).scatter_(1, cols, 1.0 / k_in)
            elif projection == "gaussian":                          # 부호 있는 밀집 투영: 정보 손실이 가장 적음
                proj = torch.randn(len(P), n_in, generator=g) / n_in ** 0.5
            else:
                raise ValueError(projection)
            self.register_buffer("proj", proj.to(dev))
        else:
            self.proj = None
        self.name = circuit.name

    def pn(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.W.device).float().flatten(1)
        a = x @ self.proj.T if self.proj is not None else x
        return a - a.mean(1, keepdim=True) if self.center else a

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """KC 입력 (상위 k만 남기기 전)"""
        return self.pn(x) @ self.W.T

    @torch.no_grad()
    def forward(self, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
        out = []
        for i in range(0, len(x), batch):
            d = self.drive(x[i:i + batch])
            top = d.topk(self.k, dim=1)
            code = torch.zeros_like(d)
            code.scatter_(1, top.indices, 1.0 if self.binary else top.values.clamp_min(0))
            out.append(code)
        return torch.cat(out)

    def extra_repr(self):
        return f"{self.name}: in {self.n_in} → PN {self.n_pn} → KC {self.n_out}, 켜짐 {self.k}개 ({self.k / self.n_out:.1%})"
