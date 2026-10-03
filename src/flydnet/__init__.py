"""flydnet — 초파리 커넥톰(FlyWire)을 배선으로 쓰는 신경망

기준 엔진은 flydnet.ganglion (자체 자동 미분, CPU = NumPy, GPU = CuPy). torch 없이 동작:

    import flydnet as fd
    mb = fd.Circuit.from_flywire()                       # 버섯체: PN → KC → MBON (+APL)
    model = fd.Pathway(fd.Projection(784, 344), fd.Neuropil(mb, "PN", "KC"),
                       fd.LateralInhibition(frac=0.05), fd.Projection(2597, 10))
    loss = fd.surprise(model(x), y); loss.retrograde()   # 역행성 신호 (자동 미분)

torch 연동 (pip install flydnet[torch]): flydnet.torch.anatomy / flydnet.torch.physiology,
그리고 0.1.0의 torch 기반 기능 (ConnectomeLayer, AssocReadout, KCExpansion, RateEncoder 등)은 처음 쓸 때 불러옴.
"""
import importlib as _importlib

from . import data
from .data import data_dir, set_data_dir, download, data_status
from .circuit import Circuit, MUSHROOM_BODY
from . import ganglion
from .ganglion import (Signal, Synapse, Tissue, Pathway, Projection, Neuropil, LateralInhibition, AxonHillock,
                       Activation, MushroomBodyOutput, Plasticity, AdaptivePlasticity, quiescent, surprise,
                       transmit, fire, inhibit)

__version__ = "0.2.0"

# torch가 필요한 기능: 처음 쓸 때 불러옴 (torch 없이도 import flydnet은 됨)
_TORCH_ATTRS = {
    "RateEncoder": "encoders", "GlomerularEncoder": "encoders",
    "ConnectomeLayer": "layers", "SpikeFn": "layers", "DEFAULT_PARAMS": "layers",
    "extract": "readout", "train_linear": "readout",
    "DopamineReadout": "plasticity", "AssocReadout": "plasticity",
    "synthetic_odors": "datasets", "door_odors": "datasets", "biconditional_mixtures": "datasets",
    "KCExpansion": "expansion",
    "visual_circuit": "visual", "column_map": "visual", "drifting_grating": "visual", "direction_offsets": "visual",
    "VISUAL_SYSTEM": "visual", "PHOTORECEPTORS": "visual", "COLUMNAR": "visual", "LPTC": "visual",
    "MOTION_PATHWAY": "visual",
}
_TORCH_MODULES = {"encoders", "layers", "readout", "plasticity", "datasets", "expansion", "visual", "torch"}


def __getattr__(name):
    mod = _TORCH_ATTRS.get(name, name if name in _TORCH_MODULES else None)
    if mod is None:
        raise AttributeError(f"module 'flydnet' has no attribute '{name}'")
    try:
        m = _importlib.import_module(f".{mod}", __name__)
    except ImportError as e:
        if "torch" in str(e):
            raise ImportError(f"flydnet.{name}에는 PyTorch가 필요함: pip install flydnet[torch]  "
                              f"(torch 없이 쓰려면 flydnet.ganglion)") from e
        raise
    val = m if name == mod else getattr(m, name)
    globals()[name] = val                                  # 다음부터는 바로
    return val


def __dir__():
    return sorted(set(globals()) | set(_TORCH_ATTRS) | _TORCH_MODULES)
