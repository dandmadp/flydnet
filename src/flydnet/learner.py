"""한 줄 학습기: 커넥톰은 고정, 데이터는 한 번 훑어 배우고, 새 데이터는 이어서 배움 (예전 것 유지)

  learner = fd.Learner()                          # 오른쪽 버섯체 PN → KC (FlyWire 데이터 필요: fd.download())
  learner.learn(X_odor, y_odor)                   # 처음 데이터로 입력 변환·연결 세기 보정까지 자동
  learner.learn(X_new, y_new)                     # 새 클래스: 이어서 추가, 앞에서 배운 것은 그대로
  learner.learn(X_img, y_img, source="image")     # 다른 종류(특징 수가 다른) 데이터: 입력 변환만 새로, 커넥톰·기억은 공유
  learner.predict(X), learner.score(X, y), learner.save("learner.npz"), fd.Learner.load("learner.npz")

구조: 데이터 → 입력 변환(source마다) → 커넥톰 층 (스파이킹, 학습 안 함) → KC 발화율 → 기억(출력)
rule
  "lda"       흐름 선형 판별 (streaming LDA, Hayes & Kanan 2020): 클래스마다 평균 KC 반응 + 모든 클래스가 함께 쓰는 공분산을
              누적 (한 번 훑기, 학습률·에폭 없음, 들어온 순서와 무관하게 같은 결과). 공분산으로 KC끼리 겹치는 반응을 걸러
              (백색화 - 측면 억제가 하는 탈상관과 같은 효과) 평균보다 훨씬 잘 구별. 앞 클래스의 평균은 바뀌지 않음 (기본)
              공분산 축소 shrink "auto" = max(OAS 추정, 특징 수 / 본 시료 수) - 시료가 특징보다 적을수록 세게
              class-incremental (차례로 배우기, 재생 없음):
                냄새 24개 KC 특징 0.863 (assoc 0.744), MNIST KC 특징 0.896 (assoc 0.728), CIFAR-100 resnet18 특징 0.640
                (assoc 0.539, 한 번에 전부 학습한 선형 상한 0.658)
  "assoc"     도파민 연합 학습 (AssocReadout과 같은 계산): 정답 클래스의 원형만 강화 → 한 번 훑기로 끝, 학습률·에폭 없음.
              다른 클래스는 절대 바뀌지 않아 차례로 배워도 잊지 않음 (생물에 가장 가까운 국소 규칙 - 희소 부호 + 연합 학습의
              망각 방지는 Shen, Dasgupta & Navlakha 2023 PNAS)
  "backprop"  역전파, 망각을 줄이는 세 가지를 함께:
              ① 새 클래스의 가중치를 그 클래스 평균 특징으로 시작 (연합 규칙으로 초기화한 뒤 역전파로 다듬기)
              ② 이번 learn()에 나온 클래스끼리만 경쟁 (도파민이 그 출력들에만 오는 것과 같음 - 다른 클래스 가중치는 안 바뀜)
              ③ 코사인 점수 (특징·가중치 크기 정규화 - 나중에 배운 클래스 쪽으로 쏠리지 않음)
              특징 중심은 처음 learn() 데이터로 정하고 고정. 세 가지 모두 연속 학습 분야에 이미 있는 기법을 묶은 것:
              ① 클래스 평균 원형 (iCaRL, Rebuffi et al. 2017), ② 나온 클래스만 경쟁 ("labels trick", Zeno et al. 2018),
              ③ 코사인 정규화 분류 층 (LUCIR, Hou et al. 2019)
              class-incremental 실험 (과제당 1에폭, 재생 없음, 본 클래스 전체 정확도):
                MNIST KC 특징 41.9% (역전파 그대로) → 65.6% (이 방법), assoc 72.8%
                CIFAR-100 resnet18 특징 24.7% → 58.5%, assoc 53.9%
"""
from __future__ import annotations

import json

import numpy as np

from . import _check as _C
from .ganglion import backend as B
from .ganglion.circuitry import ConnectomeLayer
from .ganglion.physiology import surprise
from .ganglion.rules import AdaptivePlasticity
from .ganglion.signal import Signal, quiescent
from .ganglion.tissue import Synapse, Tissue

RULES = ("lda", "assoc", "backprop")


class _StreamingLDA:
    """클래스별 평균·개수 + 클래스 안 흩어짐 행렬 S (정확한 병합 - 묶음 크기·순서와 무관). 예측 행렬은 필요할 때 다시 계산"""

    def __init__(self, d: int, shrink="auto"):
        self.d, self.shrink = d, shrink
        self.mu = np.zeros((0, d))
        self.n = np.zeros(0)
        self.S = np.zeros((d, d))
        self._W = None

    def learn(self, F, yi, n_classes):
        F = np.asarray(F, np.float64)
        if len(self.n) < n_classes:
            add = n_classes - len(self.n)
            self.mu = np.concatenate([self.mu, np.zeros((add, self.d))])
            self.n = np.concatenate([self.n, np.zeros(add)])
        for c in np.unique(yi):
            x = F[yi == c]
            nb, mb = len(x), x.mean(0)
            dx = x - mb
            self.S += dx.T @ dx
            n, m = self.n[c], self.mu[c]
            if n:
                self.S += (n * nb / (n + nb)) * np.outer(mb - m, mb - m)
            self.mu[c] = (n * m + nb * mb) / (n + nb)
            self.n[c] = n + nb
        self._W = None

    def shrinkage(self) -> float:
        """공분산 Σ에 더할 단위 행렬의 배율 (Σ의 평균 분산 기준)"""
        if self.shrink != "auto":
            return float(self.shrink)
        N, p = float(self.n.sum()), self.d
        dof = max(N - np.count_nonzero(self.n), 1.0)
        Sig = self.S / dof
        tr, tr2 = np.trace(Sig), float((Sig * Sig).sum())
        den = (dof + 1 - 2 / p) * (tr2 - tr * tr / p)
        rho = 1.0 if den <= 0 else min(1.0, max(0.0, ((1 - 2 / p) * tr2 + tr * tr) / den))   # OAS (Chen et al. 2010)
        oas = rho / (1 - rho) if rho < 1 else 1e6
        return max(oas, p / max(N, 1.0))

    def scores(self, F):
        if self._W is None:
            dof = max(float(self.n.sum()) - np.count_nonzero(self.n), 1.0)
            Sig = self.S / dof
            t = max(np.trace(Sig) / self.d, 1e-12)
            A = Sig + (self.shrinkage() + 1e-9) * t * np.eye(self.d)
            seen = self.n > 0
            W = np.zeros_like(self.mu)
            W[seen] = np.linalg.solve(A, self.mu[seen].T).T
            b = np.where(seen, -0.5 * (W * self.mu).sum(1), -np.inf)
            self._W = (W, b)
        W, b = self._W
        return np.asarray(F, np.float64) @ W.T + b


def _rows(X):
    """입력을 (n, 특징) float32 numpy로. 시료 하나 (d,)면 (1, d)"""
    if isinstance(X, Signal):
        X = X.numpy()
    elif hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    elif hasattr(X, "to_numpy"):                                         # pandas
        X = X.to_numpy()
    X = B.numpy(X) if B.device_of(X) == "gpu" else np.asarray(X)
    X = X.astype(np.float32, copy=False)
    if X.ndim == 1:
        X = X[None]
    if X.ndim != 2:
        X = X.reshape(len(X), -1)                                        # 이미지 (n, h, w) 등 → 펼침
    if len(X) == 0:
        raise ValueError("데이터가 비어 있음 (시료 0개)")
    if not np.isfinite(X).all():
        raise ValueError("입력에 NaN 또는 무한대가 있음")
    return X


def _label(v):
    """저장(JSON)할 수 있는 라벨: numpy 정수·실수·문자열 → 파이썬 값"""
    if isinstance(v, np.generic):
        v = v.item()
    if not isinstance(v, (int, str, float, bool)):
        raise TypeError(f"라벨은 정수·문자열 등 단순한 값: {v!r} ({type(v).__name__})")
    return v


class _CosineHead(Tissue):
    """코사인 점수 분류 층: s · (x / |x|) · (w / |w|). 클래스가 늘면 행을 덧붙임"""

    def __init__(self, n_in: int, scale: float = 16.0, device=None):
        super().__init__()
        self.scale = scale
        self.W = Synapse(np.zeros((0, n_in), np.float32), device=device)

    def grow(self, rows):
        xp = B.xp(self.W.device)
        self.W.data = xp.concatenate([self.W.data, B.to(np.asarray(rows, np.float32), self.W.device)])
        self.W.retro = None

    def forward(self, x):
        w = self.W
        return (x @ (w / ((w * w).sum(axis=1, keepdims=True) + 1e-12) ** 0.5).T) * self.scale


class Learner:
    """커넥톰 고정 + 기억(흐름 LDA·연합 학습·망각을 줄인 역전파). 위 모듈 설명 참고

    circuit:   회로 (None이면 오른쪽 버섯체 fd.flywire())
    inputs, outputs: 입력·출력 그룹 (기본 PN → KC)
    rule:      "lda" (기본) / "assoc" / "backprop"
    shrink:    lda의 공분산 축소 ("auto" 또는 숫자 - Σ + shrink x 평균 분산 x I)
    per_class: assoc의 클래스당 원형 수 (클래스 안에 모양이 여러 가지면 늘림, 예: 10)
    epochs, rate: backprop의 learn() 한 번당 에폭 수·학습률 (적응형 가소성 = Adam)
    t_ms, target_hz, max_rate: 커넥톰 층 시뮬레이션 길이 (ms), 보정할 출력 평균 발화율 ("auto": ConnectomeModel과 같음),
               입력 변환의 최대 발화율 (Hz)
    나머지 인자는 ConnectomeLayer로 (dt, input_mode 등)
    """

    def __init__(self, circuit=None, inputs="PN", outputs="KC", rule: str = "lda", per_class: int = 1, shrink="auto",
                 epochs: int = 5, rate: float = 1e-3, t_ms: float = 50.0, target_hz="auto", max_rate: float = 100.0,
                 batch: int = 256, seed: int = 0, device: str | None = None, **layer_kw):
        if rule not in RULES:
            raise ValueError(f"rule은 {RULES} 중 하나: {rule!r}")
        _C.integer("per_class", per_class)
        _C.integer("epochs", epochs)
        _C.pos("rate", rate)
        _C.pos("t_ms", t_ms)
        _C.pos("max_rate", max_rate)
        _C.integer("batch", batch)
        if target_hz != "auto":
            _C.pos("target_hz", target_hz)
        if shrink != "auto":
            _C.nonneg("shrink", shrink)
        self.shrink = shrink
        self.lda = None                       # lda
        self.circuit, self.inputs, self.outputs = circuit, inputs, outputs
        self.rule, self.per_class, self.epochs, self.rate = rule, per_class, epochs, float(rate)
        self.t_ms, self.target_hz, self.max_rate, self.batch, self.seed = float(t_ms), target_hz, float(max_rate), batch, seed
        self.device = B.check(device) if device is not None else B.default_device()
        self.layer_kw = dict(layer_kw)
        self.layer = None                     # 처음 learn()에서 만듦
        self.encoders = {}                    # source 이름 → 입력 변환
        self.n_in = {}                        # source 이름 → 특징 수
        self.classes = []                     # 배운 라벨 (순서 = 기억 번호)
        self._index = {}
        self.W = self.count = None            # assoc 원형
        self.head = self.opt = None           # backprop
        self.mu = self.sd = None

    # ─────────────── 만들기 ───────────────
    def _build_layer(self):
        if self.circuit is None:
            from .circuit import Circuit
            self.circuit = Circuit.from_flywire()
        self.layer = ConnectomeLayer(self.circuit, self.inputs, self.outputs, t_ms=self.t_ms, trainable=False,
                                     device=self.device, **{"dt": 0.5, "input_mode": "regular", **self.layer_kw})
        if self.target_hz == "auto":                                     # ConnectomeModel과 같은 규칙
            span = self.layer.t_ms - self.layer.count_from_ms
            self.target_hz = float(np.clip(100.0 / (self.layer.n_out * span / 1000.0), 5.0, 30.0))
        self.calibrated = False

    def _encoder(self, source, n_in):
        from .models import ConnectomeModel
        ins = [self.inputs] if isinstance(self.inputs, str) else list(self.inputs)
        enc = ConnectomeModel._encoder("auto", self.circuit, ins, n_in, self.layer.n_in, self.max_rate,
                                       self.seed + len(self.encoders), self.device)
        self.encoders[source], self.n_in[source] = enc, n_in
        return enc

    def _ready(self, X, source):
        if self.layer is None:
            self._build_layer()
        if source not in self.encoders:
            self._encoder(source, X.shape[1])
        elif X.shape[1] != self.n_in[source]:
            raise ValueError(f"source {source!r}의 특징 수는 {self.n_in[source]}: 받은 것 {X.shape[1]} "
                             f"(다른 종류의 데이터면 source='이름'을 따로)")
        if not self.calibrated:                                          # 처음 데이터로 연결 세기 보정 (한 번만 -
            pick = np.unique(np.linspace(0, len(X) - 1, min(len(X), 256)).round().astype(np.int64))   # 나중에 바꾸면
            with quiescent():                                            # 앞에서 배운 기억과 특징이 어긋남)
                rates = self.encoders[source](X[pick])
            self.layer.calibrate(rates.data if isinstance(rates, Signal) else rates,
                                 {g: self.target_hz for g in self.layer.out_names}, verbose=False)
            self.calibrated = True

    def features(self, X, source: str = "default") -> np.ndarray:
        """커넥톰 층 출력 발화율 (n, 출력 뉴런 수) numpy - 같은 X면 늘 같은 값 (seed 고정)"""
        from .readout import extract
        X = _rows(X)
        if self.layer is None or source not in self.encoders:
            raise RuntimeError(f"source {source!r}로 아직 learn()하지 않음")
        if X.shape[1] != self.n_in[source]:
            raise ValueError(f"source {source!r}의 특징 수는 {self.n_in[source]}: 받은 것 {X.shape[1]}")
        return extract(self.layer, X, self.encoders[source], batch=self.batch, seed=self.seed).astype(np.float32)

    def _labels(self, y, grow: bool):
        out = np.empty(len(y), np.int64)
        for i, v in enumerate(y):
            v = _label(v)
            if v not in self._index:
                if not grow:
                    raise ValueError(f"배운 적 없는 라벨: {v!r}")
                self._index[v] = len(self.classes)
                self.classes.append(v)
            out[i] = self._index[v]
        return out

    # ─────────────── 배우기 ───────────────
    def learn(self, X, y, source: str = "default"):
        """X, y를 배움 (assoc: 한 번 훑기, backprop: epochs번). 새 라벨은 새 클래스로, 앞에서 배운 것은 유지. 반환: self"""
        X = _rows(X)
        y = np.asarray(y.tolist() if hasattr(y, "tolist") else list(y), dtype=object)
        if y.ndim == 2 and y.shape[1] == 1:                               # (n, 1) 열 벡터 (scikit-learn 습관) → 폄
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"y는 시료마다 라벨 하나 (1차원): 모양 {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X와 y의 개수가 다름: {len(X)} 대 {len(y)}")
        self._ready(X, source)
        F = self.features(X, source)
        n0 = len(self.classes)
        yi = self._labels(y, grow=True)
        try:
            {"lda": self._learn_lda, "assoc": self._learn_assoc, "backprop": self._learn_backprop}[self.rule](F, yi)
        except BaseException:                                            # 배우다 실패하면 새 라벨도 없던 일로 (예전: 라벨만
            for v in self.classes[n0:]:                                   # 남아, 나중에 그중 일부만 배우면 backprop의 빈 클래스
                del self._index[v]                                        # 가중치가 빈 평균 = NaN)
            del self.classes[n0:]
            raise
        return self

    def _learn_lda(self, F, yi):
        if self.lda is None:
            d = F.shape[1]
            if d > 12000:                                                 # 공분산 d x d (float64): 12,000이면 1.2 GB
                raise ValueError(f"rule='lda'는 출력 뉴런 수의 제곱만큼 메모리를 씀 ({d:,}개 → {d * d * 8 / 1e9:.1f} GB) - "
                                 f"출력 그룹을 줄이거나 rule='assoc'")
            self.lda = _StreamingLDA(d, self.shrink)
        self.lda.learn(F, yi, len(self.classes))

    def _learn_assoc(self, F, yi):
        from .ganglion.physiology import reinforce_
        xp, k, C = B.xp(self.device), self.per_class, len(self.classes)
        if self.W is None:
            self.W = xp.zeros((0, F.shape[1]), xp.float32)
            self.count = xp.zeros(0, xp.float32)
        if len(self.count) < C * k:                                      # 새 클래스: 빈 원형을 덧붙임
            add = C * k - len(self.count)
            self.W = xp.concatenate([self.W, xp.zeros((add, F.shape[1]), xp.float32)])
            self.count = xp.concatenate([self.count, xp.zeros(add, xp.float32)])
        a = B.to(F / np.maximum(F.max(axis=1, keepdims=True), 1e-8), self.device)   # AssocReadout.activity와 같음
        for i in range(0, len(a), 32):                                    # 32개씩 (AssocReadout.fit과 같은 묶음, 순서 고정)
            reinforce_(self.W, self.count, a[i:i + 32], B.to(yi[i:i + 32], self.device), k)

    def _center(self, F):
        x = (F - self.mu) / self.sd
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def _learn_backprop(self, F, yi):
        if self.mu is None:                                               # 특징 중심: 처음 데이터로 정하고 고정
            self.mu = F.mean(0).astype(np.float32)
            self.sd = np.float32(max(float((F - self.mu).std()), 1e-6))
        x = self._center(F).astype(np.float32)
        C = len(self.classes)
        if self.head is None:
            self.head = _CosineHead(F.shape[1], device=self.device)
            self.opt = AdaptivePlasticity(self.head.named_synapses(), rate=self.rate)
        have = self.head.W.shape[0]
        if have < C:                                                      # ① 새 클래스 = 그 클래스 평균 특징으로 시작
            init = np.zeros((C - have, F.shape[1]), np.float32)
            for c in range(have, C):
                init[c - have] = x[yi == c].mean(0)
            self.head.grow(init)                                          # (모양이 바뀌면 가소성 규칙이 관성을 처음부터)
        mask = np.full(C, -1e4, np.float32)                               # ② 이번에 나온 클래스끼리만 경쟁
        mask[np.unique(yi)] = 0
        mask = B.to(mask, self.device)
        rng = np.random.default_rng(self.seed + len(self.classes))
        for _ in range(self.epochs):
            perm = rng.permutation(len(x))
            for i in range(0, len(x), 32):
                j = perm[i:i + 32]
                self.opt.clear()
                surprise(self.head(B.to(x[j], self.device)) + mask, yi[j]).retrograde()
                self.opt.step()

    # ─────────────── 쓰기 ───────────────
    def scores(self, X, source: str = "default") -> np.ndarray:
        """(n, 클래스 수) 점수 numpy (열 순서 = self.classes)"""
        if not self.classes:
            raise RuntimeError("아직 아무것도 배우지 않음: learn(X, y)부터")
        F = self.features(X, source)
        if self.rule == "lda":
            return self.lda.scores(F)
        if self.rule == "assoc":
            from .ganglion.physiology import recall
            a = B.to(F / np.maximum(F.max(axis=1, keepdims=True), 1e-8), self.device)
            s = recall(a, self.W, self.count).reshape(len(a), len(self.classes), self.per_class).max(axis=2)
            return B.numpy(s)
        with quiescent():
            return B.numpy(self.head(B.to(self._center(F).astype(np.float32), self.device)).data)

    def predict(self, X, source: str = "default", classes=None):
        """예측 라벨 (배운 라벨 그대로의 값). classes: 이 라벨들 중에서만 고름. 시료 하나 (d,)면 라벨 하나"""
        one = (X.ndim if hasattr(X, "ndim") else np.ndim(X)) == 1
        s = self.scores(X, source)
        if classes is not None:
            m = np.full(len(self.classes), -np.inf)
            m[self._labels(list(classes), grow=False)] = 0
            s = s + m
        out = np.asarray(self.classes, dtype=object)[s.argmax(1)]
        if all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in self.classes):
            out = out.astype(np.int64)
        return out[0] if one else out

    def score(self, X, y, source: str = "default", classes=None) -> float:
        """정확도"""
        y = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X, source, classes)
        return float(np.mean([_label(a) == _label(b) for a, b in zip(p, y)]))

    def __repr__(self):
        where = "만들기 전" if self.layer is None else \
            f"{self.layer.circuit.name} {'+'.join(self.layer.in_names)} → {'+'.join(self.layer.out_names)} {self.layer.n_out}"
        return f"Learner({where}, rule={self.rule}, 클래스 {len(self.classes)}개, source {list(self.encoders) or '-'})"

    # ─────────────── 저장 ───────────────
    def _config(self):
        return dict(inputs=self.inputs, outputs=self.outputs, rule=self.rule, per_class=self.per_class, shrink=self.shrink,
                    epochs=self.epochs, rate=self.rate, t_ms=self.t_ms, target_hz=self.target_hz,
                    max_rate=self.max_rate, batch=self.batch, seed=self.seed, layer_kw=self.layer_kw)

    def save(self, path):
        """파일 하나에 (설정, 라벨, 입력 변환·커넥톰 보정, 기억). 회로 자체는 저장하지 않음 - 불러올 때 같은 회로로"""
        from ._archive import write
        if self.layer is None:
            raise RuntimeError("아직 아무것도 배우지 않음: 저장할 것이 없음")
        arr = {"config": np.array(json.dumps(self._config())), "classes": np.array(json.dumps(self.classes)),
               "sources": np.array(json.dumps(self.n_in)), "circuit_name": np.array(str(self.circuit.name))}
        arr.update({f"layer.{k}": v for k, v in self.layer.state().items()})
        for i, (name, enc) in enumerate(self.encoders.items()):
            arr.update({f"enc{i}.{k}": v for k, v in enc.state().items()})
        if self.rule == "lda":
            arr["lda_mu"], arr["lda_n"], arr["lda_S"] = self.lda.mu, self.lda.n, self.lda.S
        elif self.rule == "assoc":
            arr["W"], arr["count"] = B.numpy(self.W), B.numpy(self.count)
        else:
            arr["head"], arr["mu"], arr["sd"] = self.head.W.numpy(), self.mu, np.asarray(self.sd)
            st = self.opt.state[0]                                       # 학습 관성 (Adam m, v, 단계 수) - 불러온 뒤 이어 배우면
            if st is not None:                                           # 저장 안 한 것과 같은 결과 (예전: 관성을 잃어 달라짐)
                arr["opt_m"], arr["opt_v"] = B.numpy(st[0]), B.numpy(st[1])
                arr["opt_t"] = np.asarray(B.numpy(self.opt.t[0]) if np.ndim(self.opt.t[0]) else self.opt.t[0])
        return write(path, "Learner", arr)

    @classmethod
    def load(cls, path, circuit=None, device: str | None = None):
        """save()한 파일. circuit: 만들 때 회로를 직접 줬으면 같은 회로 (None이면 fd.flywire())"""
        from ._archive import read
        d, _ = read(path, "Learner")
        cfg = json.loads(str(d["config"]))
        kw = cfg.pop("layer_kw")
        obj = cls(circuit=circuit, device=device, **cfg, **kw)
        obj._build_layer()
        obj.target_hz = cfg["target_hz"]
        obj.layer.load_state({k[6:]: d[k] for k in d if k.startswith("layer.")})
        obj.calibrated = True
        for i, (name, n_in) in enumerate(json.loads(str(d["sources"])).items()):
            enc = obj._encoder(name, int(n_in))
            p = f"enc{i}."
            enc.load_state({k[len(p):]: d[k] for k in d if k.startswith(p)})
        obj.classes = json.loads(str(d["classes"]))
        obj._index = {v: i for i, v in enumerate(obj.classes)}
        if obj.rule == "lda":
            obj.lda = _StreamingLDA(d["lda_S"].shape[0], obj.shrink)
            obj.lda.mu, obj.lda.n, obj.lda.S = (np.asarray(d[k], np.float64) for k in ("lda_mu", "lda_n", "lda_S"))
        elif obj.rule == "assoc":
            obj.W, obj.count = B.to(d["W"], obj.device), B.to(d["count"], obj.device)
        else:
            obj.mu, obj.sd = np.asarray(d["mu"], np.float32), np.float32(d["sd"])
            obj.head = _CosineHead(d["head"].shape[1], device=obj.device)
            obj.head.W.data = B.to(np.asarray(d["head"], np.float32), obj.device)
            obj.opt = AdaptivePlasticity(obj.head.named_synapses(), rate=obj.rate)
            if "opt_m" in d:
                obj.opt.state[0] = (B.to(np.asarray(d["opt_m"], np.float32), obj.device),
                                    B.to(np.asarray(d["opt_v"], np.float32), obj.device))
                t = np.asarray(d["opt_t"])
                obj.opt.t[0] = int(t) if t.ndim == 0 else B.to(t, obj.device)
        return obj
