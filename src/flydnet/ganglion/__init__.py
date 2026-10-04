"""flydnet.ganglion - flydnet 자체 엔진 (신경절): NumPy(CPU)·CuPy(GPU) 위의 신경 신호와 역행성 신호(자동 미분)

    import flydnet.ganglion as G

    model = G.Pathway(
        G.Projection(784, 344),                  # 축삭 투사 (모두 연결)
        G.Neuropil(mb, "PN", "KC"),              # 실제 커넥톰 배선
        G.LateralInhibition(frac=0.05),          # APL 억제
        G.Projection(2597, 10),
    )
    rule = G.AdaptivePlasticity(model.synapses(), rate=1e-3)
    loss = G.surprise(model(x), y)               # 놀람 (교차 엔트로피)
    rule.clear(); loss.retrograde(); rule.step() # 역행성 신호 → 가소성

| torch                    | flydnet.ganglion                 |
|--------------------------|----------------------------------|
| Tensor                   | Signal                           |
| requires_grad            | plastic                          |
| backward() / .grad       | retrograde() / .retro            |
| no_grad()                | quiescent()                      |
| nn.Parameter / nn.Module | Synapse / Tissue                 |
| nn.Sequential / Linear   | Pathway / Projection             |
| optim.SGD / Adam         | Plasticity / AdaptivePlasticity  |
| F.cross_entropy          | surprise                         |
| zero_grad / state_dict   | clear / state                    |
| "cuda"                   | "gpu"                            |
"""
from . import backend
from .backend import gpu_available, default_device, limit_gpu_memory
from .signal import Signal, quiescent, learning_enabled, as_signal, concat, where
from .physiology import Wiring, wiring, transmit, fire, inhibit, log_softmax, surprise
from .tissue import (Synapse, Tissue, Pathway, Projection, Neuropil, LateralInhibition, AxonHillock, Activation, Homeostasis,
                     MushroomBodyOutput)
from .rules import Plasticity, AdaptivePlasticity
from .circuitry import ConnectomeLayer, DEFAULT_PARAMS
from .signal import checkpoint
