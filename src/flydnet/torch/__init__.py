"""flydnet.torch — torch 연동 (선택). torch가 설치되어 있어야 함

flydnet의 기준 엔진은 flydnet.ganglion (NumPy·CuPy). 이 패키지는 같은 구조물·작용을 torch 텐서로 제공:
    from flydnet.torch.anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput   # torch.nn
    import flydnet.torch.physiology as P                                                            # torch.nn.functional
"""
from . import anatomy, physiology
from .anatomy import Neuropil, LateralInhibition, AxonHillock, MushroomBodyOutput
