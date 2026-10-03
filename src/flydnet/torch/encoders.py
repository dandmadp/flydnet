"""텐서 → 입력 뉴런 발화율(Hz) 변환. '설탕 뉴런 150Hz 자극'을 일반 데이터로 확장한 것"""
import numpy as np
import torch
import torch.nn as nn


def _to_rates(x: torch.Tensor, max_rate: float) -> torch.Tensor:
    """음수는 0, 샘플마다 최댓값 = max_rate (전체 세기가 달라도 같은 패턴이면 같은 발화율)"""
    x = x.clamp_min(0)
    return x / x.amax(1, keepdim=True).clamp_min(1e-8) * max_rate


class RateEncoder(nn.Module):
    """입력 특징 n_in개를 입력 뉴런 n_out개의 발화율로 바꿈

    - n_in == n_out 이고 projection=None: 특징 하나 = 뉴런 하나
    - 아니면 고정 무작위 희소 투영: 뉴런마다 특징 k개를 모아 받음 (사구체가 여러 수용체 입력을 모으듯)
    출력은 샘플마다 최댓값이 max_rate가 되도록 정규화 (음수는 0)
    """

    def __init__(self, n_in: int, n_out: int, max_rate: float = 100.0, k: int = 20, seed: int = 0,
                 projection: str | None = "random"):
        super().__init__()
        self.n_in, self.n_out, self.max_rate = n_in, n_out, max_rate
        if projection is None:
            if n_in != n_out:
                raise ValueError("projection=None이면 n_in == n_out 이어야 함")
            self.P = None
        else:
            g = torch.Generator().manual_seed(seed)
            cols = torch.stack([torch.randperm(n_in, generator=g)[:k] for _ in range(n_out)])
            P = torch.zeros(n_out, n_in)
            P.scatter_(1, cols, 1.0 / k)
            self.register_buffer("P", P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1).float()
        if self.P is not None:
            x = x @ self.P.T.to(x.device)
        return _to_rates(x, self.max_rate)


class GlomerularEncoder(nn.Module):
    """냄새 = 사구체별 활성 벡터 (B, n_glomeruli) → 투사 뉴런(PN) 발화율 (B, n_PN)

    실제 더듬이엽 구조를 따름: 같은 사구체의 단일 사구체형 PN들은 같은 발화율을 받음.
    다중 사구체형 PN은 입력 0 (오른쪽 버섯체에서 PN→KC 시냅스의 96%가 단일 사구체형).
    사구체 이름은 cell_type의 '_' 앞부분 (예: DM1_lPN → DM1)
    """

    def __init__(self, circuit, group: str = "PN", max_rate: float = 100.0):
        super().__init__()
        if circuit.meta is None:
            raise ValueError("circuit.meta(세포 주석)가 필요함 — Circuit.from_flywire()로 만든 회로를 쓸 것")
        m = circuit.meta.iloc[circuit.groups[group]]
        uni = m.cell_sub_class.astype(str).eq("uniglomerular").values
        glom = m.cell_type.astype(str).str.split("_").str[0].values
        self.glomeruli = sorted(set(glom[uni]))
        col = {gname: j for j, gname in enumerate(self.glomeruli)}
        P = torch.zeros(len(m), len(self.glomeruli))
        for i in np.nonzero(uni)[0]:
            P[i, col[glom[i]]] = 1.0
        self.register_buffer("P", P)
        self.max_rate = max_rate

    @property
    def n_glomeruli(self) -> int:
        return len(self.glomeruli)

    def forward(self, odor: torch.Tensor) -> torch.Tensor:
        return _to_rates(odor.float() @ self.P.T.to(odor.device), self.max_rate)
