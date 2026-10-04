"""없는 이름을 쓰면 맞는 이름을 알려 주는 오류 (torch·numpy에서 쓰던 이름 → flydnet 이름)"""
from __future__ import annotations

import difflib

_DATA = "이 신호 자체가 값: .numpy() (numpy 배열) 또는 .data"

SIGNAL = {
    "grad": ".retro (역행성 신호 = 기울기)",
    "backward": ".retrograde()",
    "requires_grad": ".plastic",
    "requires_grad_": "Signal(x, plastic=True)",
    "w": _DATA, "weight": _DATA, "weights": _DATA, "value": _DATA, "values": _DATA, "tensor": _DATA, "array": _DATA,
    "cpu": ".to('cpu')", "cuda": ".to('gpu')", "clone": ".copy()",
    "view": ".reshape(...)", "unsqueeze": ".reshape(...) (예: x.reshape(1, -1))", "permute": ".transpose(...)",
    "dim": ".ndim", "numel": ".size", "zero_": "s.data[...] = 0 (값을 직접) 또는 rule.clear() (기울기)",
    "float": ".astype(np.float32)", "long": ".astype(np.int64)",
}

TISSUE = {
    "parameters": ".synapses()", "named_parameters": ".named_synapses()", "zero_grad": ".clear_retro()",
    "state_dict": ".state()", "load_state_dict": ".load_state(...)", "cuda": ".to('gpu')", "cpu": ".to('cpu')",
    "children": ".tissues()", "modules": ".tissues()", "eval": "with fd.quiescent(): (역전파 경로 없이)",
    "train": "fd.train(model, X, y) (학습 루프) - 층의 상태 전환은 없음",
    "weight": "Projection이면 .weight, Connectome이면 .log_scale (학습 배율) / .weights() (연결마다 실제 세기)",
    "w": "Projection이면 .weight, Connectome이면 .log_scale / .weights()",
    "num_parameters": ".n_synapses()",
}

RULE = {
    "zero_grad": ".clear()", "param_groups": ".synapses", "lr": ".rate", "params": ".synapses",
}

CIRCUIT = {
    "num_nodes": ".N", "num_neurons": ".N", "n_neurons": ".N", "nodes": ".root_ids", "edges": ".pre, .post, .weight",
    "edge_index": ".pre, .post", "weights": ".weight", "n": ".N", "num_edges": ".n_edges", "adjacency": ".to_scipy()",
}

MODULE = {
    "Linear": "fd.Projection", "Sequential": "fd.Pathway", "Module": "fd.Tissue", "Parameter": "fd.Synapse",
    "Tensor": "fd.Signal", "tensor": "fd.Signal", "Adam": "fd.Adaptive", "AdamW": "fd.Adaptive(..., decay=...)",
    "SGD": "fd.Plasticity", "no_grad": "fd.quiescent", "cross_entropy": "fd.surprise", "CrossEntropyLoss": "fd.surprise",
    "LayerNorm": "fd.Homeostasis", "ReLU": "fd.Activation('relu')", "relu": "Signal.relu() 또는 fd.Activation('relu')",
    "load": "fd.Connectome.load(path) 또는 tissue.load(path)", "ConnectomeLayer_": "fd.Connectome",
    "Brain": "fd.brain()", "FlyWire": "fd.flywire()", "from_flywire": "fd.flywire()", "whole_brain": "fd.brain()",
}


def missing(obj_name: str, name: str, table: dict, candidates) -> AttributeError:
    """AttributeError에 맞는 이름 안내를 붙임"""
    msg = f"'{obj_name}' object has no attribute '{name}'"
    if name in table:
        return AttributeError(f"{msg} - flydnet에서는 {table[name]}")
    close = difflib.get_close_matches(name, [c for c in candidates if not c.startswith("_")], n=3, cutoff=0.7)
    return AttributeError(f"{msg}" + (f" - 혹시 {', '.join(close)}?" if close else ""))
