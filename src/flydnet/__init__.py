"""flydnet — 초파리 커넥톰을 배선으로 쓰는 스파이킹 신경망 층

    import flydnet as fd
    mb = fd.Circuit.from_flywire()                       # 버섯체: PN → KC → MBON (+APL)
    enc = fd.RateEncoder(784, len(mb.groups["PN"]))      # 텐서 → PN 발화율
    layer = fd.ConnectomeLayer(mb, inputs="PN", outputs="KC")
    feats = layer(enc(images))                           # (B, n_KC) 발화율 → 리드아웃 학습

    # 역전파 학습: 배선 고정, 연결 세기 학습 (대리 기울기)
    layer = fd.ConnectomeLayer(mb, "PN", "KC", dt=0.5, t_ms=50, input_mode="regular", trainable=True)
    model = torch.nn.Sequential(enc, layer, torch.nn.Linear(layer.n_out, 10))
"""
from .circuit import Circuit, MUSHROOM_BODY, DEFAULT_DATA
from .encoders import RateEncoder, GlomerularEncoder
from .layers import ConnectomeLayer, SpikeFn, DEFAULT_PARAMS
from .readout import extract, train_linear
from .plasticity import DopamineReadout, AssocReadout
from .datasets import synthetic_odors, door_odors, biconditional_mixtures

__version__ = "0.1.0"
