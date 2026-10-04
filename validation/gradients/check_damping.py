"""스파이킹 역전파 기울기 확인: 회로·길이마다 surrogate_damp 후보를 fd.gradcheck로 비교 (README.md의 표)

  python validation/gradients/check_damping.py      # 약 15분 (GPU), worm 데이터 필요 (python -m flydnet download worm)
"""
import numpy as np, flydnet as fd
fd.ganglion.limit_gpu_memory(0.6)
rng = np.random.default_rng(0)
cases = []
mb = fd.Circuit.from_flywire(); enc = fd.GlomerularEncoder(mb); door = fd.door_odors(enc.glomeruli)
P = door["X"][np.argsort(door["X"].sum(1))[::-1][:16]]; P = P / P.max()
Xmb = enc(np.clip(np.repeat(P, 4, 0) * rng.lognormal(0, 0.5, (64, P.shape[1])), 0, 1).astype(np.float32))
cases.append(("버섯체 dt 0.5, 50 ms (100스텝)", lambda: fd.ConnectomeLayer(mb, "PN", "MBON", t_ms=50, dt=0.5, gains={"PN>KC": 3.0, "KC>MBON": 3.0}, input_mode="regular", trainable=True), Xmb, 48))
cases.append(("버섯체 dt 0.1, 100 ms (1000스텝)", lambda: fd.ConnectomeLayer(mb, "PN", "MBON", t_ms=100, gains={"PN>KC": 3.0, "KC>MBON": 3.0}, input_mode="regular", trainable=True), Xmb, 48))
worm = fd.Circuit.celegans()
amph = [f"{c}{s}" for c in ["ASE", "AWC", "AWA", "ASH", "ADL", "ASK", "ASI", "ASJ", "AWB", "ADF", "ASG", "AFD"] for s in "LR"]
sens = fd.genetics.driver(worm, node=amph)
worm = worm.regroup({"sensory": sens.idx, "muscle": worm.groups["body_muscle"]}, rest="neuron")
Xw = (rng.uniform(0, 1, (64, 24)) ** 2 * 150).astype(np.float32)
cases.append(("예쁜꼬마선충 100 ms (1000스텝)", lambda: fd.ConnectomeLayer(worm, "sensory", "muscle", t_ms=100, params={"w_syn": 0.825}, input_mode="regular", trainable=True), Xw, 95))
sb = fd.graphs.stochastic_block({"in": 20, "A": 150, "B": 150, "out": 30}, {("in", "A"): 0.2, ("A", "A"): 0.05, ("A", "B"): 0.05, ("B", "A"): 0.05, ("B", "B"): 0.05, ("B", "out"): 0.2}, weight=25.0, seed=0)
Xs = (rng.uniform(0, 1, (64, 20)) * 150).astype(np.float32)
cases.append(("합성 블록 (되먹임 A<->B) 100 ms", lambda: fd.ConnectomeLayer(sb, "in", "out", t_ms=100, input_mode="regular", trainable=True), Xs, 30))
for name, make, X, n_out in cases:
    L = make()
    w = fd.ganglion.backend.to(np.random.default_rng(1).standard_normal(n_out).astype(np.float32), L.device)
    score = lambda L_, s, X=X, w=w: (L_(X, seed=s) * fd.Signal(w)).sum() * (1.0 / len(X))
    print(f"== {name}", flush=True)
    best = fd.tune_surrogate(score, L, candidates=(1.0, 0.3, 0.1, 0.03))
    print(f"   → 가장 잘 맞는 surrogate_damp {best}", flush=True)
