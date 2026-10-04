"""값 → 입력 뉴런 발화율(Hz) 변환 (자체 엔진판, torch 없음). '설탕 뉴런 150Hz 자극'을 일반 데이터로 확장한 것"""
from __future__ import annotations

from . import _check as _C
import numpy as np

from .ganglion import backend as B
from .ganglion.signal import Signal, as_signal, as_input
from .ganglion.tissue import Tissue


def _samples(x, n: int, what: str, device=None):
    """입력을 (B, n) Signal로. 1차원 (n,)은 시료 하나 → (신호, True). 특징 수가 다르면 오류
    (예전: 1차원을 특징 1개짜리 시료 n개로 봐서 모두 max_rate가 나옴)"""
    x = as_input(x, device)
    one = x.ndim == 1
    if one:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.flatten(1)
    if x.shape[-1] != n:
        raise ValueError(f"{what}: 입력 특징 {x.shape[-1]}개 ≠ {n}개")
    if x.data.dtype.kind != "f":
        x = Signal(x.data.astype(np.float32))
    return x, one


def to_rates(x, max_rate: float) -> Signal:
    """음수는 0, 샘플마다 최댓값 = max_rate (전체 세기가 달라도 같은 패턴이면 같은 발화율). 미분 가능"""
    x = as_input(x).flatten(1).relu()
    return x / x.max(axis=1, keepdims=True).clip(lo=1e-8) * max_rate


class RateEncoder(Tissue):
    """입력 특징 n_in개를 입력 뉴런 n_out개의 발화율로 바꿈

    - n_in == n_out 이고 projection=None: 특징 하나 = 뉴런 하나
    - 아니면 고정 무작위 희소 투영: 뉴런마다 특징 k개를 모아 받음 (사구체가 여러 수용체 입력을 모으듯)
    출력은 샘플마다 최댓값이 max_rate가 되도록 정규화 (음수는 0)
    torch판(flydnet.torch.RateEncoder)과 투영의 무작위 선택은 다름 (난수 생성기가 다름)
    """

    def __init__(self, n_in: int, n_out: int, max_rate: float = 100.0, k: int = 20, seed: int = 0,
                 projection: str | None = "random", device: str | None = None):
        _C.integer('n_in', n_in)
        _C.integer('n_out', n_out)
        _C.nonneg('max_rate', max_rate)
        _C.integer('k', k)
        super().__init__()
        dev = B.check(device) if device is not None else B.default_device()
        self.n_in, self.n_out, self.max_rate = n_in, n_out, max_rate
        if projection is None:
            if n_in != n_out:
                raise ValueError("projection=None이면 n_in == n_out 이어야 함")
            self.buffer("P", None, persistent=False)
        else:
            rng = np.random.default_rng(seed)
            k = min(k, n_in)
            P = np.zeros((n_out, n_in), np.float32)
            for i in range(n_out):
                P[i, rng.choice(n_in, k, replace=False)] = 1.0 / k
            self.buffer("P", B.to(P, dev))
        self._dev = dev

    def forward(self, x) -> Signal:
        dev = B.device_of(self.P) if self.P is not None else self._dev
        x, one = _samples(x, self.n_in, "RateEncoder", dev)
        if self.P is not None:
            x = x @ Signal(self.P.T)
        r = to_rates(x, self.max_rate)
        return r.reshape(-1) if one else r

    def extra_repr(self):
        return f"{self.n_in} → {self.n_out}, 최대 {self.max_rate} Hz"


class GlomerularEncoder(Tissue):
    """냄새 = 사구체별 활성 벡터 (B, n_glomeruli) → 투사 뉴런(PN) 발화율 (B, n_PN)

    실제 더듬이엽 구조를 따름: 같은 사구체의 단일 사구체형 PN들은 같은 발화율을 받음.
    다중 사구체형 PN은 입력 0 (오른쪽 버섯체에서 PN→KC 시냅스의 96%가 단일 사구체형).
    사구체 이름은 cell_type의 '_' 앞부분 (예: DM1_lPN → DM1)
    """

    def __init__(self, circuit, group: str = "PN", max_rate: float = 100.0, device: str | None = None):
        _C.nonneg('max_rate', max_rate)
        super().__init__()
        if circuit.meta is None:
            raise ValueError("circuit.meta(세포 주석)가 필요함 - Circuit.from_flywire()로 만든 회로를 쓸 것")
        dev = B.check(device) if device is not None else B.default_device()
        m = circuit.meta.iloc[circuit.groups[group]]
        uni = m.cell_sub_class.astype(str).eq("uniglomerular").values
        glom = m.cell_type.astype(str).str.split("_").str[0].values
        self.glomeruli = sorted(set(glom[uni]))
        col = {g: j for j, g in enumerate(self.glomeruli)}
        P = np.zeros((len(m), len(self.glomeruli)), np.float32)
        for i in np.nonzero(uni)[0]:
            P[i, col[glom[i]]] = 1.0
        self.buffer("P", B.to(P, dev))
        self.max_rate = max_rate

    @property
    def n_glomeruli(self) -> int:
        return len(self.glomeruli)

    def forward(self, odor) -> Signal:
        x, one = _samples(odor, self.n_glomeruli, "GlomerularEncoder (사구체 수)", B.device_of(self.P))
        if x.data.dtype != np.float32 and not x.plastic:
            x = Signal(x.data.astype(np.float32))
        r = to_rates(x @ Signal(self.P.T), self.max_rate)
        return r.reshape(-1) if one else r

    def extra_repr(self):
        return f"사구체 {self.n_glomeruli} → PN {self.P.shape[0]}, 최대 {self.max_rate} Hz"
