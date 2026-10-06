"""실제 PN→KC 배선을 한 번에 계산하는 빠른 확장 층 (자체 엔진판, torch 없음, 스파이크 시뮬레이션 없음)

  kc = fd.KCExpansion(fd.Circuit.from_flywire(), n_in=512)    # 특징 512개 → KC 2,597개
  codes = kc(features)                                         # Signal (B, 2597), KC 5%만 켜짐

버섯체가 하는 일을 앞먹임 계산으로 줄인 것 (FlyHash, Dasgupta et al. 2017과 같은 구조, 단 배선은 실제 FlyWire):
  특징 ─(고정 투영)─▶ PN 활동 ─(평균 빼기: 촉각엽의 측억제)─▶ ─(실제 PN→KC 시냅스 수)─▶ KC 입력
       ─(상위 k_frac만 남김: APL 억제)─▶ KC 코드
"""
from __future__ import annotations

from . import _check as _C
import numpy as np

from .circuit import Circuit
from .ganglion import backend as B
from .ganglion.physiology import kenyon_code
from .ganglion.signal import Signal
from .ganglion.tissue import Tissue


class KCExpansion(Tissue):
    """특징 (B, n_in) → KC 코드 (B, n_kc)

    circuit:    PN·KC 그룹이 있는 회로 (무작위 대조군은 circuit.shuffled())
    n_in:       입력 특징 수. None이면 입력이 이미 PN 활동 (B, n_pn)
    projection: "sparse" = PN마다 특징 k_in개 평균 (음수 없음, 사구체처럼) / "gaussian" = 부호 있는 밀집 무작위
    k_frac:     켜 둘 KC 비율 (실제 초파리는 약 5~10%)
    center:     PN 활동에서 샘플별 평균을 뺌
    binary:     True = 켜진 KC는 1 / False = 켜진 KC는 입력 세기 그대로
    """

    def __init__(self, circuit: Circuit, n_in: int | None = None, pre: str = "PN", post: str = "KC",
                 k_in: int = 20, k_frac: float = 0.05, center: bool = True, binary: bool = False,
                 projection: str = "sparse", seed: int = 0, device: str | None = None):
        _C.unit('k_frac', k_frac, lo_open=True)
        super().__init__()
        dev = B.check(device) if device is not None else B.default_device()
        P, K = circuit.groups[pre], circuit.groups[post]
        lp = np.full(circuit.N, -1, np.int64); lp[P] = np.arange(len(P))
        lk = np.full(circuit.N, -1, np.int64); lk[K] = np.arange(len(K))
        m = (lp[circuit.pre] >= 0) & (lk[circuit.post] >= 0) & (circuit.weight > 0)
        W = np.zeros((len(K), len(P)), np.float32)
        np.add.at(W, (lk[circuit.post[m]], lp[circuit.pre[m]]), circuit.weight[m])
        self.buffer("W", B.to(W, dev))                                    # (n_kc, n_pn) 시냅스 수
        self.n_pn, self.n_out = len(P), len(K)
        self.n_in = n_in if n_in is not None else len(P)
        self.k = max(1, int(round(k_frac * len(K))))
        self.center, self.binary = center, binary
        if n_in is not None:
            rng = np.random.default_rng(seed)
            if projection == "sparse":
                k_in = min(k_in, max(1, n_in // 2))                   # 특징 수의 절반 이하: k_in >= n_in이면 모든 PN이 같은 평균 →
                                                                     # 평균 빼기 뒤 0이 되어 입력과 상관없이 같은 KC 코드였음 (n_in <= 20)
                proj = np.zeros((len(P), n_in), np.float32)
                for i in range(len(P)):
                    proj[i, rng.choice(n_in, k_in, replace=False)] = 1.0 / k_in
            elif projection == "gaussian":
                proj = (rng.normal(size=(len(P), n_in)) / np.sqrt(n_in)).astype(np.float32)
            else:
                raise ValueError(f"projection은 'sparse' 또는 'gaussian': {projection!r}")
            self.buffer("proj", B.to(proj, dev))
        else:
            self.buffer("proj", None, persistent=False)
        self.name = circuit.name

    def forward(self, x, batch: int = 4096) -> Signal:
        _C.integer("batch", batch)
        xs = x.data if isinstance(x, Signal) else x
        if hasattr(xs, "detach"):
            xs = xs.detach().cpu().numpy()
        elif hasattr(xs, "to_numpy"):                                     # pandas
            xs = xs.to_numpy()
        elif not hasattr(xs, "shape"):                                    # 리스트
            xs = np.asarray(xs, np.float32)
        one = xs.ndim == 1                                                # 시료 하나 (n_in,)
        if one:
            xs = xs.reshape(1, -1)
        if xs.ndim != 2 or xs.shape[1] != self.n_in:
            raise ValueError(f"KCExpansion: 입력은 (시료, {self.n_in}): {tuple(xs.shape)}")
        xp = B.xp(B.device_of(self.W))
        if len(xs) == 0:                                                  # 빈 입력 → 빈 출력 (다른 층과 같게, 예전: concatenate 오류)
            return Signal(xp.zeros((0, self.n_out), dtype=self.W.dtype))
        out = [kenyon_code(xs[i:i + batch], self.W, self.k, self.proj, self.center, self.binary)
               for i in range(0, len(xs), batch)]
        out = xp.concatenate(out)
        return Signal(out[0] if one else out)

    def extra_repr(self):
        return f"{self.name}: in {self.n_in} → PN {self.n_pn} → KC {self.n_out}, 켜짐 {self.k}개 ({self.k / self.n_out:.1%})"
