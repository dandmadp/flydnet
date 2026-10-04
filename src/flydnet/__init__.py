"""flydnet - 초파리 커넥톰(FlyWire)을 배선으로 쓰는 신경망

기준 엔진은 flydnet.ganglion (자체 자동 미분, CPU = NumPy, GPU = CuPy). torch 없이 동작:

    import flydnet as fd
    mb = fd.Circuit.from_flywire()                       # 버섯체: PN → KC → MBON (+APL)
    model = fd.Pathway(fd.Projection(784, 344), fd.Neuropil(mb, "PN", "KC"),
                       fd.LateralInhibition(frac=0.05), fd.Projection(2597, 10))
    loss = fd.surprise(model(x), y); loss.retrograde()   # 역행성 신호 (자동 미분)

모든 기능이 자체 엔진 (torch 없음): 시간 시뮬레이션 fd.ConnectomeLayer, 인코더, 리드아웃, 도파민 학습,
KC 확장, 데이터셋, 시각계 도구.
torch 연동 (pip install flydnet[torch]): flydnet.torch - 0.1의 torch판 전부 (같은 이름), torch.nn용 구조물.
"""
import importlib as _importlib

from . import data
from .data import data_dir, set_data_dir, download, data_status
from .circuit import Circuit, MUSHROOM_BODY
from . import ganglion
from .ganglion import (Signal, Synapse, Tissue, Pathway, Projection, Neuropil, LateralInhibition, AxonHillock,
                       Activation, Homeostasis, MushroomBodyOutput, Plasticity, AdaptivePlasticity, quiescent, surprise,
                       transmit, fire, inhibit, ConnectomeLayer, DEFAULT_PARAMS, checkpoint)

__version__ = "0.1.17"

# 0.1 기능의 자체 엔진판 (torch 없음). torch판은 flydnet.torch에 같은 이름으로
from .encoders import RateEncoder, GlomerularEncoder, to_rates
from .readout import extract, train_linear
from .plasticity import DopamineReadout, AssocReadout
from .expansion import KCExpansion
from .datasets import synthetic_odors, door_odors, biconditional_mixtures
from . import graphs                       # fd.graphs.erdos_renyi / watts_strogatz / barabasi_albert / stochastic_block / layered
from . import genetics                     # fd.genetics.driver / silence / block / activate / ablate / screen
from .training import train, evaluate, door_task  # 학습 루프 한 줄, 정확도, 실제 냄새 분류 과제
from .models import ConnectomeModel, MushroomBody  # 한 줄 모델: 입력 Hz 변환·자동 보정·분류 층까지
from . import neurons                      # 사용자 정의 뉴런 모델: fd.neurons.NeuronModel, register, LIF, Izhikevich
from .stdp import STDP, Monitor                 # 관찰자 규격과 STDP (사용자 정의 학습 규칙의 틀)
from .gradcheck import gradcheck, tune_surrogate, GradCheck   # 스파이킹 역전파 기울기 확인·감쇠 고르기
from .threefactor import ThreeFactor             # 3요소 학습 규칙 (e-prop): 역전파 없이 커넥톰 연결 학습
from .attribution import explain, Explanation   # 회로 기여도: 어떤 세포 유형·경로에 기대는지 + 실제로 꺼서 확인
from . import controls                     # fd.controls.Shuffled / Randomized / ShuffledWeights / Local / Custom
from .controls import compare, CompareReport, sign_flip_p
from .visual import (visual_circuit, column_map, drifting_grating, direction_offsets,
                     VISUAL_SYSTEM, PHOTORECEPTORS, COLUMNAR, LPTC, MOTION_PATHWAY)


# 짧은 이름 (긴 이름도 그대로 동작)
Connectome = ConnectomeLayer
Adaptive = AdaptivePlasticity
Glomeruli = GlomerularEncoder
Inhibition = LateralInhibition
MBON = MushroomBodyOutput
tune = tune_surrogate
flywire = Circuit.from_flywire          # fd.flywire() = 오른쪽 버섯체, fd.flywire(groups, side=...)
brain = Circuit.whole_brain             # 전체 뇌 138,639개 뉴런
worm = Circuit.celegans                 # 예쁜꼬마선충


def __getattr__(name):
    """flydnet.torch (torch 연동)는 처음 쓸 때 불러옴. torch가 없으면 설치 안내"""
    if name != "torch":
        from .ganglion.hints import MODULE, missing
        err = missing("flydnet", name, MODULE, list(globals()))
        raise AttributeError(str(err).replace("'flydnet' object", "module 'flydnet'"))
    try:
        m = _importlib.import_module(".torch", __name__)
    except ImportError as e:
        raise ImportError("flydnet.torch에는 PyTorch가 필요함: pip install flydnet[torch]  "
                          "(torch 없이 쓰려면 flydnet 최상위 이름들)") from e
    globals()["torch"] = m
    return m
