"""0.1.15 저장 형식의 고정 파일 만들기 (한 번만 실행해 저장소에 넣음 - 이후 버전이 계속 읽는지 test_archive.py가 확인)"""
import sys
from pathlib import Path

import numpy as np
import flydnet as fd

out = Path(__file__).parent / f"v{fd.__version__ if len(sys.argv) < 2 else sys.argv[1]}"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
N, n_in = 17, 5
key = rng.choice(N * N, 120, replace=False)
pre, post = key // N, key % N
keep = pre != post
w = (rng.integers(1, 6, keep.sum()) * rng.choice([-1, 1], keep.sum())).astype(np.float32)
c = fd.Circuit(np.arange(N), {"IN": np.arange(n_in), "OUT": np.arange(n_in, N)}, pre[keep], post[keep], w)
x = rng.random((4, n_in)).astype(np.float32)
X = rng.random((40, 20)).astype(np.float32)
y = rng.integers(0, 3, 40)
np.savez(out / "inputs.npz", x=x, X=X, y=y)

layer = fd.ConnectomeLayer(c, "IN", "OUT", t_ms=20, trainable=True, device="cpu")
for s in layer.synapses():
    s.data += rng.normal(0, 0.1, s.shape).astype(np.float32)
layer.save(out / "layer.npz")
np.save(out / "layer_out.npy", layer(x, seed=0).numpy())

tissue = fd.Pathway(fd.Projection(20, 5, seed=1, device="cpu"), fd.Neuropil(c, "IN", "OUT", device="cpu"))
tissue.save(out / "tissue.npz")

dop = fd.DopamineReadout(20, 3, device="cpu"); dop.fit(X, y, epochs=2)
dop.save(out / "dopamine.npz"); np.save(out / "dopamine_pred.npy", dop.predict(X))
asc = fd.AssocReadout(20, 3, device="cpu"); asc.fit(X, y)
asc.save(out / "assoc.npz"); np.save(out / "assoc_pred.npy", asc.predict(X))
print("ok", out)
