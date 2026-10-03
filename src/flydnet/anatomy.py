"""해부학: 매개변수를 가진 신경 구조물 (torch.nn에 해당)

    import torch.nn as nn
    import flydnet as fd
    from flydnet.anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput

    mb = fd.Circuit.from_flywire()
    model = nn.Sequential(
        nn.Linear(784, 344),                     # 픽셀 → PN 344개
        Neuropil(mb, "PN", "KC"),                # 실제 PN→KC 배선 (nn.Linear처럼, 연결은 커넥톰에 있는 것만)
        LateralInhibition(frac=0.05),            # APL: KC 5%만 남김
        nn.Linear(2597, 10),
    )

| 구조물              | 하는 일                                   | torch에서 비슷한 것 |
|---------------------|-------------------------------------------|---------------------|
| Neuropil            | 실제 커넥톰 배선으로 신호 전달 (학습 가능) | nn.Linear (희소)    |
| LateralInhibition   | 가장 강한 k개만 남김 (APL 억제)            | 활성화 함수         |
| AxonHillock         | 문턱을 넘으면 스파이크 (대리 기울기)        | 활성화 함수         |
| MushroomBodyOutput  | 도파민 연합 학습 분류기 (역전파 없음)       | 분류기 헤드         |

상태 없는 함수는 flydnet.physiology. 시간에 따른 스파이크 시뮬레이션은 ConnectomeLayer.
"""
from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn

from . import physiology as P
from .circuit import Circuit


class Neuropil(nn.Module):
    """신경망 영역: 회로의 pre 그룹 → post 그룹 시냅스로 신호 전달. (..., n_pre) → (..., n_post)

    nn.Linear와 같은 자리에 쓰되, 연결은 커넥톰에 실제로 있는 시냅스만 (희소). 시간 시뮬레이션 없이 한 번에 계산.

    train:  "edge" = 연결마다 배율 학습, 부호 유지 (Dale의 법칙, 기본)
            "pair" = 연결 종류(그룹 쌍)마다 배율 하나 학습, 부호 유지 (매개변수 적음)
            "free" = 연결마다 값을 자유롭게 학습 (부호도 바뀔 수 있음, 배선 모양만 커넥톰)
            None   = 고정
    init:   "fan_in" = 받는 뉴런마다 √(Σ w²)로 나눔 → 입력 분산 ≈ 출력 분산 (깊게 쌓아도 신호 크기 유지)
            "counts" = 시냅스 수 그대로 (±)
    bias:   True면 받는 뉴런마다 편향 (nn.Linear처럼)
    """

    def __init__(self, circuit: Circuit, pre, post, train: str | None = "edge", init: str = "fan_in",
                 bias: bool = False, device: str | None = None):
        super().__init__()
        if train not in ("pair", "edge", "free", None):
            raise ValueError(f"train은 'pair', 'edge', 'free', None 중 하나: {train}")
        if init not in ("fan_in", "counts"):
            raise ValueError(init)
        pre = [pre] if isinstance(pre, str) else list(pre)
        post = [post] if isinstance(post, str) else list(post)
        for g in pre + post:
            if g not in circuit.groups:
                raise ValueError(f"회로에 없는 그룹: {g}")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pre_idx = np.concatenate([circuit.groups[g] for g in pre])
        post_idx = np.concatenate([circuit.groups[g] for g in post])
        lp = np.full(circuit.N, -1, np.int64); lp[pre_idx] = np.arange(len(pre_idx))
        lq = np.full(circuit.N, -1, np.int64); lq[post_idx] = np.arange(len(post_idx))
        m = (lp[circuit.pre] >= 0) & (lq[circuit.post] >= 0)
        q, p, w = lq[circuit.post[m]], lp[circuit.pre[m]], circuit.weight[m].astype(np.float64)
        key, inv = np.unique(q * len(pre_idx) + p, return_inverse=True)   # 같은 연결은 시냅스 수 합치기
        w = np.bincount(inv, weights=w, minlength=len(key))
        q, p = key // len(pre_idx), key % len(pre_idx)
        nz = w != 0                                                       # 합이 0인 연결(±가 상쇄)은 빼기
        q, p, w, key = q[nz], p[nz], w[nz], key[nz]
        if len(key) == 0:
            raise ValueError(f"{pre} → {post} 연결이 없음")
        wr, order = P.wiring(q, p, len(post_idx), len(pre_idx), device=dev)
        q, p, w = q[order], p[order], w[order]
        if init == "fan_in":
            norm = np.sqrt(np.bincount(q, weights=w ** 2, minlength=len(post_idx)))
            w = w / norm[q]
        for name, t in zip(wr._fields, wr):
            self.register_buffer(name, t, persistent=False)
        self.register_buffer("base", torch.tensor(w, dtype=torch.float32, device=dev))
        self.register_buffer("edge_post", torch.tensor(q, device=dev), persistent=False)
        self.register_buffer("edge_pre", torch.tensor(p, device=dev), persistent=False)

        # 연결 종류 (예: "PN>KC") — 정수 번호로 계산 (전체 뇌 1,500만 연결에서 문자열을 만들면 느림)
        names = list(circuit.groups)
        gid = np.full(circuit.N, -1, np.int64)
        for i, nm in enumerate(names):
            gid[circuit.groups[nm]] = i
        code = gid[pre_idx][p] * len(names) + gid[post_idx][q]
        uniq, which = np.unique(code, return_inverse=True)
        self.pairs = np.array([f"{names[c // len(names)]}>{names[c % len(names)]}" for c in uniq], dtype=object)
        self.register_buffer("pair_of", torch.tensor(which, device=dev), persistent=False)
        # 배선 지문: 다른 회로(예: 무작위 배선)의 state_dict를 불러오면 조용히 섞이지 않고 오류
        fp = np.array([len(pre_idx), len(post_idx), len(q), int(q.sum()), int(p.sum()),
                       int((q * 31 + p * 17).sum() % (2 ** 61 - 1))], dtype=np.int64)
        self.register_buffer("wiring_id", torch.tensor(fp))
        self.train_mode = train
        if train == "pair":
            self.log_scale = nn.Parameter(torch.zeros(len(self.pairs), device=dev))
        elif train == "edge":
            self.log_scale = nn.Parameter(torch.zeros(len(w), device=dev))
        elif train == "free":
            self.weight_values = nn.Parameter(self.base.clone())
        self.bias = nn.Parameter(torch.zeros(len(post_idx), device=dev)) if bias else None
        self.in_features, self.out_features = len(pre_idx), len(post_idx)
        self.pre_names, self.post_names = pre, post
        self._name = circuit.name

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "wiring_id"
        if key in state_dict and not torch.equal(state_dict[key].cpu(), self.wiring_id.cpu()):
            raise RuntimeError("Neuropil 배선이 다름: 이 state_dict는 다른 회로(또는 다른 그룹)로 만든 Neuropil의 것")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def n_synapses(self) -> int:
        """연결(시냅스 쌍) 수"""
        return len(self.base)

    def values(self) -> torch.Tensor:
        """현재 연결 값 (wiring 순서: post 순, 그 안에서 pre 순)"""
        if self.train_mode == "pair":
            return self.base * self.log_scale.exp()[self.pair_of]
        if self.train_mode == "edge":
            return self.base * self.log_scale.exp()
        if self.train_mode == "free":
            return self.weight_values
        return self.base

    @property
    def wiring(self) -> P.Wiring:
        return P.Wiring(self.crow, self.col, self.crow_t, self.col_t, self.perm_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = P.transmit(x.to(self.base.device).float(), self.values(), self.wiring)
        return out + self.bias if self.bias is not None else out

    def dense(self) -> torch.Tensor:
        """밀집 가중치 (n_post, n_pre) — 확인용. 큰 회로에서는 메모리 주의"""
        if self.out_features * self.in_features > 2e8:
            raise MemoryError(f"밀집 행렬 {self.out_features}×{self.in_features}는 너무 큼")
        W = torch.zeros(self.out_features, self.in_features, device=self.base.device)
        return W.index_put((self.edge_post, self.edge_pre), self.values().detach())

    def scale_of(self) -> dict:
        """train="pair"일 때 연결 종류별 학습된 배율"""
        if self.train_mode != "pair":
            raise ValueError("train='pair'에서만")
        return dict(zip(self.pairs, self.log_scale.detach().exp().tolist()))

    def extra_repr(self):
        dens = self.n_synapses / (self.in_features * self.out_features)
        return (f"{'+'.join(self.pre_names)} {self.in_features} → {'+'.join(self.post_names)} {self.out_features}, "
                f"시냅스 연결 {self.n_synapses:,}개 (밀도 {dens:.2%}), 학습 {self.train_mode}, "
                f"bias={self.bias is not None}")


class LateralInhibition(nn.Module):
    """측억제 (APL 뉴런처럼): 마지막 차원에서 가장 강한 k개(또는 비율 frac)만 남기고 0. 기울기는 남은 것에만"""

    def __init__(self, k: int | None = None, frac: float | None = None):
        super().__init__()
        if (k is None) == (frac is None):
            raise ValueError("k와 frac 중 하나만")
        self.k, self.frac = k, frac

    def forward(self, x):
        return P.inhibit(x, k=self.k, frac=self.frac)

    def extra_repr(self):
        return f"k={self.k}" if self.k is not None else f"frac={self.frac}"


class AxonHillock(nn.Module):
    """축삭 둔덕: 문턱을 넘으면 스파이크 1, 아니면 0. 역전파는 대리 기울기 (slope가 클수록 날카로움)"""

    def __init__(self, threshold: float = 0.0, slope: float = 10.0):
        super().__init__()
        self.threshold, self.slope = threshold, slope

    def forward(self, v):
        return P.fire(v, self.threshold, self.slope)

    def extra_repr(self):
        return f"threshold={self.threshold}, slope={self.slope}"


class MushroomBodyOutput(nn.Module):
    """버섯체 출력 구역: 도파민 연합 학습 분류기 (역전파 없음, 연속 학습에 강함)

    클래스마다 출력(원형) per_class개. learn(x, y): 정답 클래스의 가장 잘 맞는 원형 하나만 보상 → 그 원형은
    받은 샘플의 평균이 됨. 다른 클래스는 절대 안 바뀜 → 클래스를 차례로 배워도 잊지 않음.
    forward(x) = 클래스 점수 (B, n_classes): 그 클래스 원형 중 최대 코사인 유사도 (배우지 않은 클래스는 −inf).
    nn.Module이라 .to(device), state_dict 저장, nn.Sequential 안에 넣기가 됨. AssocReadout과 같은 계산.
    """

    def __init__(self, n_in: int, n_classes: int, per_class: int = 1, binary: bool = False):
        super().__init__()
        self.n_in, self.n_classes, self.per_class, self.binary = n_in, n_classes, per_class, binary
        self.register_buffer("prototypes", torch.zeros(n_classes * per_class, n_in))
        self.register_buffer("count", torch.zeros(n_classes * per_class))

    def activity(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.prototypes.device).float().flatten(1)
        if self.binary:
            return (x > 0).float()
        return x / x.amax(1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = P.recall(self.activity(x), self.prototypes, self.count)
        return s.view(len(x), self.n_classes, self.per_class).amax(2)

    @torch.no_grad()
    def learn(self, x: torch.Tensor, y: torch.Tensor, batch: int = 256):
        """도파민 강화로 배우기 (순서와 묶음 크기에 거의 무관한 누적 평균)"""
        for i in range(0, len(x), batch):
            P.reinforce_(self.prototypes, self.count, self.activity(x[i:i + batch]), y[i:i + batch], self.per_class)
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor, classes=None) -> torch.Tensor:
        """classes: 후보로 삼을 클래스 (예: 지금까지 배운 것)"""
        s = self(x)
        if classes is not None:
            mask = torch.full((self.n_classes,), float("-inf"), device=s.device)
            mask[list(classes)] = 0
            s = s + mask
        return s.argmax(1)

    @property
    def learned_classes(self) -> list[int]:
        return torch.nonzero(self.count.view(self.n_classes, self.per_class).sum(1) > 0)[:, 0].tolist()

    def extra_repr(self):
        return f"in {self.n_in} → 클래스 {self.n_classes} × 원형 {self.per_class}, 배운 클래스 {len(self.learned_classes)}개"


__all__ = ["Neuropil", "LateralInhibition", "AxonHillock", "MushroomBodyOutput"]
