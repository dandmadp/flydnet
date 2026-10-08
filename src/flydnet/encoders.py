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


NEG_FRAC = 0.05                                                       # 입력 칸 중 음수가 이 비율 이상이면 알림


def _warn_negative(tissue, x, what: str):
    """음수가 많은 입력을 한 번 알림 (구조물마다 한 번). 발화율은 0 이상이라 음수는 0으로 잘려 그 정보가 사라짐 -
    예전엔 조용히 잘려 StandardScaler 입력에서 정확도가 떨어졌음 (보고된 예: Iris 0.867 → 0.733)"""
    if getattr(tissue, "_neg_warned", False) or not x.data.size:
        return
    frac = float((x.data < 0).mean())
    if frac >= NEG_FRAC:
        import warnings
        object.__setattr__(tissue, "_neg_warned", True)
        hint = " 부호에 뜻이 있으면 negative='onoff' (ON·OFF 두 채널)" if what == "RateEncoder" else ""
        warnings.warn(f"{what} 입력의 {frac:.0%}가 음수 - 음수는 발화율 0으로 잘림 (그 정보가 사라짐). "
                      f"MinMaxScaler나 X / 255처럼 0 이상으로 바꿀 것.{hint}", stacklevel=4)


def to_rates(x, max_rate: float) -> Signal:
    """음수는 0, 샘플마다 최댓값 = max_rate (전체 세기가 달라도 같은 패턴이면 같은 발화율). 미분 가능"""
    x = as_input(x).flatten(1).relu()
    return x / x.max(axis=1, keepdims=True).clip(lo=1e-8) * max_rate


class RateEncoder(Tissue):
    """입력 특징 n_in개를 입력 뉴런 n_out개의 발화율로 바꿈

    - n_in == n_out 이고 projection=None: 특징 하나 = 뉴런 하나
    - 아니면 고정 무작위 희소 투영: 뉴런마다 특징 k개를 모아 받음 (사구체가 여러 수용체 입력을 모으듯)
    출력은 샘플마다 최댓값이 max_rate가 되도록 정규화
    negative: 음수 특징을 어떻게
      "clip"  (기본) 0으로 자름 - 발화율은 0 이상. 음수가 많으면 (5% 이상) 한 번 알림
      "onoff" ON(max(x, 0))·OFF(max(-x, 0)) 두 채널로 나눠 넣음 (시각계 ON/OFF 경로처럼, 부호 정보가 남음).
              특징 2 x n_in개가 입력 뉴런 n_out개로 (projection=None이면 n_out = 2 x n_in)
    """

    def __init__(self, n_in: int, n_out: int, max_rate: float = 100.0, k: int = 20, seed: int = 0,
                 projection: str | None = "random", device: str | None = None, negative: str = "clip"):
        _C.integer('n_in', n_in)
        _C.integer('n_out', n_out)
        _C.pos('max_rate', max_rate)                                   # 0이면 입력이 모두 0 Hz (예전: 허용)
        _C.integer('k', k)
        if negative not in ("clip", "onoff"):
            raise ValueError(f"negative는 'clip'(음수를 0으로) 또는 'onoff'(ON·OFF 두 채널): {negative!r}")
        super().__init__()
        dev = B.check(device) if device is not None else B.default_device()
        self.n_in, self.n_out, self.max_rate, self.negative = n_in, n_out, max_rate, negative
        ch = 2 * n_in if negative == "onoff" else n_in                 # 실제로 모으는 채널 수
        if projection is None:
            if ch != n_out:
                raise ValueError("projection=None이면 n_in == n_out 이어야 함" if negative == "clip" else
                                 f"projection=None, negative='onoff'이면 n_out = 2 x n_in ({2 * n_in}): {n_out}")
            self.buffer("P", None, persistent=False)
        else:
            rng = np.random.default_rng(seed)
            # 특징 수의 절반 이하: k >= n_in이면 모든 뉴런이 같은 평균을 받아, 시료마다 최댓값 정규화 뒤 모두 max_rate가
            # 되어 입력 정보가 전부 사라졌음 (특징 8개, k 20 → 20개 뉴런 모두 100 Hz). n_in >= 40이면 예전과 같음
            k = min(k, max(1, ch // 2))
            P = np.zeros((n_out, ch), np.float32)
            for i in range(n_out):
                P[i, rng.choice(ch, k, replace=False)] = 1.0 / k
            self.buffer("P", B.to(P, dev))
        self._dev = dev

    def forward(self, x) -> Signal:
        dev = B.device_of(self.P) if self.P is not None else self._dev
        x, one = _samples(x, self.n_in, "RateEncoder", dev)
        if self.negative == "onoff":
            from .ganglion.signal import concat
            x = concat([x.relu(), (-x).relu()], axis=1)                  # ON·OFF 두 채널
        else:
            _warn_negative(self, x, "RateEncoder")
        if self.P is not None:
            x = x @ Signal(self.P.T)
        r = to_rates(x, self.max_rate)
        return r.reshape(-1) if one else r

    def _moved(self, device):
        self._dev = device                                               # 투영이 없으면 장치를 따로 기억 (예전: .to 뒤에도 CPU에서 계산)

    @property
    def device(self) -> str:
        return B.device_of(self.P) if self.P is not None else self._dev

    def extra_repr(self):
        return f"{self.n_in} → {self.n_out}, 최대 {self.max_rate} Hz" + (", ON·OFF 두 채널" if self.negative == "onoff" else "")


class GlomerularEncoder(Tissue):
    """냄새 = 사구체별 활성 벡터 (B, n_glomeruli) → 투사 뉴런(PN) 발화율 (B, n_PN)

    실제 더듬이엽 구조를 따름: 같은 사구체의 단일 사구체형 PN들은 같은 발화율을 받음.
    다중 사구체형 PN은 입력 0 (오른쪽 버섯체에서 PN→KC 시냅스의 96%가 단일 사구체형).
    사구체 이름은 cell_type의 '_' 앞부분 (예: DM1_lPN → DM1)
    """

    def __init__(self, circuit, group: str = "PN", max_rate: float = 100.0, device: str | None = None):
        _C.pos('max_rate', max_rate)                                   # 0이면 입력이 모두 0 Hz (예전: 허용)
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
        _warn_negative(self, x, "GlomerularEncoder")
        r = to_rates(x @ Signal(self.P.T), self.max_rate)
        return r.reshape(-1) if one else r

    def extra_repr(self):
        return f"사구체 {self.n_glomeruli} → PN {self.P.shape[0]}, 최대 {self.max_rate} Hz"
