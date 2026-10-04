"""flydnet.torch - torch 연동 (선택). torch가 설치되어 있어야 함

flydnet의 기준 엔진은 flydnet.ganglion (NumPy·CuPy).
torch 모델 안에서 flydnet을 쓸 때는 연결 장치 (계산은 자체 엔진, 복사 없음, 역전파 이어짐):
    model = torch.nn.Sequential(..., fd.torch.bridge(fd.ConnectomeLayer(...), seed=0), ...)
아래는 0.1의 torch판 복사본 (예전 결과 재현용, ConnectomeLayer는 timing="legacy"와 같음):
    from flydnet.torch.anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput   # torch.nn
    import flydnet.torch.physiology as P                                                            # torch.nn.functional
"""
from . import anatomy, physiology
from .bridge import bridge, Bridge, to_engine, to_torch   # 자체 엔진 구조물을 torch 모델 안에서 (권장)
from .anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput
from .layers import ConnectomeLayer, SpikeFn, SparsePropagate, DEFAULT_PARAMS   # 0.1의 torch판 시간 시뮬레이션
# 0.1의 torch판 기능 (자체 엔진판은 flydnet 최상위에 같은 이름으로)
from .encoders import RateEncoder, GlomerularEncoder
from .readout import extract, train_linear
from .plasticity import DopamineReadout, AssocReadout
from .expansion import KCExpansion
from .datasets import synthetic_odors, door_odors, biconditional_mixtures
from .visual import (visual_circuit, column_map, drifting_grating, direction_offsets,
                     VISUAL_SYSTEM, PHOTORECEPTORS, COLUMNAR, LPTC, MOTION_PATHWAY)
