"""한 줄로 만드는 커넥톰 모델: 입력 → 발화율(Hz) 변환 → 커넥톰 층 → 정규화 → 분류 층, 세기는 처음 데이터로 자동 보정

  model = fd.MushroomBody(n_in=X.shape[1], n_classes=12)       # 오른쪽 버섯체 PN → KC
  model.fit(Xtr, ytr, val=(Xte, yte))                           # fd.train (보정은 학습 전에 자동으로)
  model.score(Xte, yte), model.predict(X)

  model = fd.ConnectomeModel(circuit, "in", "out", n_in=20, n_classes=4)   # 어떤 회로·그래프든

PyTorch의 nn.Sequential처럼 짧게 쓰도록, 커넥톰 층을 쓸 때 매번 손으로 하던 세 가지를 대신 함:
  - 입력을 발화율(Hz)로: 0~1 값을 그대로 넣으면 입력이 거의 없는 것과 같음. 버섯체에 사구체 반응(사구체 수와 같은 특징 수)이
    들어오면 실제 사구체 → PN 구조(GlomerularEncoder), 아니면 RateEncoder (특징 수 = 입력 뉴런 수면 하나씩, 아니면 무작위 투영)
  - 연결 세기 보정: 기본 세기로는 출력이 거의 조용함 (버섯체 MBON 96%가 0 Hz) → 처음 본 데이터로 calibrate
    (fit 전에, 또는 처음 순전파 때). 보정 값은 저장됨
  - 출력 뉴런 수(생물학이 정함, KC 2,597개 등)를 클래스 수로: Homeostasis + Projection
세부 조정이 필요하면 .encoder·.layer·.head로 각 부분에 접근하거나, 지금처럼 fd.Pathway로 직접 조립.
"""
from __future__ import annotations

import numpy as np

from . import _check as _C
from .ganglion import backend as B
from .ganglion.circuitry import ConnectomeLayer
from .ganglion.signal import Signal, quiescent
from .ganglion.tissue import Homeostasis, Pathway, Projection


class ConnectomeModel(Pathway):
    """encoder → ConnectomeLayer → Homeostasis → Projection(n_out, n_classes). fd.train·evaluate·save·load 그대로

    circuit, inputs, outputs: ConnectomeLayer와 같음
    n_in:       입력 특징 수
    n_classes:  분류할 클래스 수
    encoder:    "auto" / "rate" / "glomeruli" / 구조물(Tissue, 출력이 입력 뉴런 수의 Hz) / None (입력이 이미 Hz)
    max_rate:   변환한 입력의 최대 발화율 (Hz)
    target_hz:  보정할 출력 그룹 평균 발화율 (Hz). "auto" = 시행당 출력 스파이크가 약 100개가 되게 5~30 Hz
                (KC 2,597개면 5 Hz로 희소하게, 출력 뉴런이 적으면 높게 - 10개에 5 Hz면 50 ms에 스파이크 몇 개뿐이라
                정보가 남지 않음)
    trainable:  "path" (입력 → 출력 흥분성 경로 위의 연결 종류, 기본: 출력이 KC면 PN>KC, MBON이면 PN>KC·KC>MBON 등) /
                "input" (입력 그룹에서 나가는 연결 종류) / True (모든 연결) / False (커넥톰 고정, 분류 층만) / 목록
    calibrate:  True면 처음 본 데이터로 자동 보정 (calibrate_samples개까지)
    나머지 (t_ms, dt, input_mode, seed, device, 그 밖의 ConnectomeLayer 인자)는 커넥톰 층으로
    """

    def __init__(self, circuit, inputs, outputs, n_in: int, n_classes: int, encoder="auto", max_rate: float = 100.0,
                 target_hz="auto", trainable="path", calibrate: bool = True, calibrate_samples: int = 256,
                 t_ms: float = 50.0, dt: float = 0.5, input_mode: str = "regular", seed: int = 0,
                 device: str | None = None, **layer_kw):
        _C.integer("n_in", n_in)
        _C.integer("n_classes", n_classes, lo=2)
        _C.pos("max_rate", max_rate)
        if target_hz != "auto":
            _C.pos("target_hz", target_hz)
        _C.integer("calibrate_samples", calibrate_samples)
        dev = B.check(device) if device is not None else B.default_device()
        ins = [inputs] if isinstance(inputs, str) else list(inputs)
        outs = [outputs] if isinstance(outputs, str) else list(outputs)
        if trainable == "input":
            trainable = self._input_pairs(circuit, ins) or False
        elif trainable == "path":
            trainable = self._path_pairs(circuit, ins, outs) or False
        layer = ConnectomeLayer(circuit, inputs, outputs, t_ms=t_ms, dt=dt, input_mode=input_mode, trainable=trainable,
                                device=dev, **layer_kw)
        enc = self._encoder(encoder, circuit, ins, n_in, layer.n_in, max_rate, seed, dev)
        super().__init__(*([enc] if enc is not None else []), layer, Homeostasis(),
                         Projection(layer.n_out, n_classes, seed=seed, device=dev))
        if target_hz == "auto":
            target_hz = float(np.clip(100.0 / (layer.n_out * layer.t_ms / 1000.0), 5.0, 30.0))
        self.n_in, self.n_classes, self.target_hz = n_in, n_classes, float(target_hz)
        self.calibrate_samples, self.auto_calibrate = calibrate_samples, calibrate
        self.buffer("calibrated", np.array(False), optional=True)          # 저장됨 → 불러온 뒤 다시 보정하지 않음

    # ─────────────── 부분 ───────────────
    @property
    def encoder(self):
        return self._order[0] if len(self._order) == 4 else None

    @property
    def layer(self) -> ConnectomeLayer:
        return self._order[-3]

    @property
    def head(self):
        return Pathway(*self._order[-2:])

    @staticmethod
    def _path_pairs(circuit, ins, outs) -> list:
        """입력 → 출력 흥분성 경로 위의 연결 종류: 보내는 그룹은 입력에서 (흥분성으로) 닿고, 받는 그룹은 출력에 닿는 쌍.
        억제를 내보내는 그룹(APL 등)의 연결은 빠짐"""
        g = circuit.group_of()
        pre, post = g[circuit.pre], g[circuit.post]
        import pandas as pd
        net = pd.Series(circuit.weight.astype(np.float64)).groupby([pre, post]).sum()
        exc = [(a, b) for (a, b), w in net.items() if w > 0 and a != "?" and b != "?"]
        fwd, frontier = set(ins), list(ins)
        while frontier:
            frontier = [b for a, b in exc if a in frontier and b not in fwd]
            fwd.update(frontier)
        bwd, frontier = set(outs), list(outs)
        while frontier:
            frontier = [a for a, b in exc if b in frontier and a not in bwd]
            bwd.update(frontier)
        return sorted(f"{a}>{b}" for a, b in exc if a in fwd and b in bwd and b not in ins)

    @staticmethod
    def _input_pairs(circuit, ins) -> list:
        """입력 그룹에서 나가는 연결 종류 이름 (예: ["PN>KC", "PN>APL"])"""
        g = circuit.group_of()
        pre, post = g[circuit.pre], g[circuit.post]
        keys = {f"{a}>{b}" for a, b in zip(pre, post) if a in ins}
        return sorted(keys)

    @staticmethod
    def _encoder(kind, circuit, ins, n_in, n_neurons, max_rate, seed, dev):
        from .encoders import GlomerularEncoder, RateEncoder
        from .ganglion.tissue import Tissue
        if kind is None:
            if n_in != n_neurons:
                raise ValueError(f"encoder=None이면 입력 특징 수 = 입력 뉴런 수 ({n_neurons}): n_in={n_in}")
            return None
        if isinstance(kind, Tissue) or callable(kind) and not isinstance(kind, str):
            return kind
        if kind not in ("auto", "rate", "glomeruli"):
            raise ValueError(f"encoder는 'auto', 'rate', 'glomeruli', 구조물, None: {kind!r}")
        glom = None
        if kind in ("auto", "glomeruli") and len(ins) == 1 and circuit.meta is not None and \
                {"cell_sub_class", "cell_type"} <= set(circuit.meta.columns):
            try:
                glom = GlomerularEncoder(circuit, ins[0], max_rate=max_rate, device=dev)
            except Exception:                                                # noqa: BLE001 - 사구체 주석이 없는 그룹
                glom = None
        if kind == "glomeruli":
            if glom is None or glom.n_glomeruli != n_in:
                raise ValueError(f"encoder='glomeruli'는 입력 특징 수 = 사구체 수 ({None if glom is None else glom.n_glomeruli})")
            return glom
        if glom is not None and glom.n_glomeruli == n_in:                   # 사구체 반응 데이터 (fd.door_task 등)
            return glom
        return RateEncoder(n_in, n_neurons, max_rate=max_rate, seed=seed,
                           projection=None if n_in == n_neurons else "random", device=dev)

    # ─────────────── 보정 ───────────────
    @property
    def is_calibrated(self) -> bool:
        return bool(B.numpy(self.calibrated))

    def prepare(self, X, verbose: bool = False):
        """처음 데이터로 연결 세기 보정 (출력 그룹 평균 target_hz). 반환: calibrate 표. fit이 학습 전에 부름"""
        n = len(X)
        if n > self.calibrate_samples:                                     # 앞에서 자르면 클래스 순으로 정렬된 데이터
            pick = np.unique(np.linspace(0, n - 1, self.calibrate_samples).round().astype(np.int64))   # (door_task 등)의
            X = X.iloc[pick] if hasattr(X, "iloc") else X[pick] if hasattr(X, "shape") else [X[i] for i in pick]   # 뒤 클래스가 빠짐
        with quiescent():
            rates = self.encoder(X) if self.encoder is not None else X
        rates = rates.data if isinstance(rates, Signal) else rates
        tab = self.layer.calibrate(rates, {g: self.target_hz for g in self.layer.out_names}, verbose=verbose)
        self.buffer("calibrated", np.array(True), optional=True)
        return tab

    def forward(self, x, seed: int | None = None):
        if self.auto_calibrate and not self.is_calibrated:
            import warnings
            warnings.warn("보정 전에 순전파 - 이 묶음으로 자동 보정함 (더 많은 데이터로: model.prepare(X) 또는 model.fit)",
                          stacklevel=2)
            self.prepare(x.data if isinstance(x, Signal) else x)
        return super().forward(x, seed=seed)

    # ─────────────── sklearn처럼 ───────────────
    def fit(self, X, y, **train_kw) -> dict:
        """fd.train(self, X, y, ...) - 보정이 안 되어 있으면 먼저 X로 보정. 반환: 학습 기록"""
        from .training import train
        if self.auto_calibrate and not self.is_calibrated:
            self.prepare(X, verbose=False)
        return train(self, X, y, **train_kw)

    def predict(self, X, batch: int = 256) -> np.ndarray:
        """예측 클래스 (numpy)"""
        _C.integer("batch", batch)
        out = []
        with quiescent():
            for i in range(0, len(X), batch):
                o = self(X[i:i + batch], seed=10 ** 6 + i)
                out.append(B.numpy(o.data).argmax(-1))
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def score(self, X, y, batch: int = 256) -> float:
        """정확도"""
        from .training import evaluate
        return evaluate(self, X, y, batch=batch)

    def extra_repr(self):
        return (f"in {self.n_in} → {self.layer.circuit.name} ({'+'.join(self.layer.in_names)} → "
                f"{'+'.join(self.layer.out_names)} {self.layer.n_out}) → 클래스 {self.n_classes}, "
                f"보정 {'됨' if self.is_calibrated else '전'} (목표 {self.target_hz:g} Hz)")


def MushroomBody(n_in: int, n_classes: int, output: str = "KC", side: str = "right", data_dir=None, **kw) -> ConnectomeModel:
    """오른쪽(side) 버섯체 PN → output("KC" 또는 "MBON") 모델. FlyWire 데이터 필요 (fd.download()).
    KC 출력 (기본)이 정확도가 높음: MBON은 48개뿐이고 KC → MBON 배선을 거쳐 정보가 줄어듦 (README 표 참고)"""
    from .circuit import Circuit
    if output not in ("KC", "MBON"):
        raise ValueError(f"output은 'KC' 또는 'MBON': {output!r}")
    c = Circuit.from_flywire(side=side, data_dir=data_dir)
    return ConnectomeModel(c, "PN", output, n_in, n_classes, **kw)
