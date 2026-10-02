"""텐서 → 입력 뉴런 발화율(Hz) 변환. '설탕 뉴런 150Hz 자극'을 일반 데이터로 확장한 것"""
import torch
import torch.nn as nn


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
        x = x.clamp_min(0)
        return x / x.amax(1, keepdim=True).clamp_min(1e-8) * self.max_rate
