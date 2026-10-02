"""flydnet — 초파리 커넥톰을 배선으로 쓰는 스파이킹 신경망 층

    import flydnet as fd
    mb = fd.Circuit.from_flywire()                       # 버섯체: PN → KC → MBON (+APL)
    enc = fd.RateEncoder(784, len(mb.groups["PN"]))      # 텐서 → PN 발화율
    layer = fd.ConnectomeLayer(mb, inputs="PN", outputs="KC")
    feats = layer(enc(images))                           # (B, n_KC) 발화율 → 리드아웃 학습
"""
from .circuit import Circuit, MUSHROOM_BODY, DEFAULT_DATA
from .encoders import RateEncoder, GlomerularEncoder
from .layers import ConnectomeLayer, DEFAULT_PARAMS
from .readout import extract, train_linear
from .plasticity import DopamineReadout, AssocReadout
from .datasets import synthetic_odors, door_odors

__version__ = "0.1.0"
