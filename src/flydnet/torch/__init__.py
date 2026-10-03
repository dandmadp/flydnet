"""flydnet.torch — torch 연동 (선택). torch가 설치되어 있어야 함

flydnet의 기준 엔진은 flydnet.ganglion (NumPy·CuPy). 이 패키지는 같은 구조물·작용을 torch 텐서로 제공:
    from flydnet.torch.anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput   # torch.nn
    import flydnet.torch.physiology as P                                                            # torch.nn.functional
"""
from . import anatomy, physiology
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
