"""엔진 비트 단위 비교: 여러 설정(효과기·ThreeFactor·체크포인트·플러그인 등)의 출력·기울기를 기준 파일과 비교
  python tests/snapshot.py save [경로]    # 기준 저장 (엔진을 일부러 바꾼 뒤에만)
  python tests/snapshot.py check [경로]   # 기준과 비교 (기본 .verify/snapshot_base.npz)
"""
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flydnet as fd
from flydnet.ganglion import backend as B
from test_threefactor import _rec

warnings.simplefilter("ignore")
G = fd.genetics
PATH = sys.argv[2] if len(sys.argv) > 2 else str(pathlib.Path(__file__).resolve().parents[1] / ".verify" / "snapshot_base.npz")
c = _rec(feedback_edges=True, strong=True)
X = np.random.default_rng(0).uniform(50, 200, (5, 6)).astype(np.float32)


def run(dev, kw, effect=None, tf=False):
    L = fd.Connectome(c, "IN", "O", **dict(dict(t_ms=40, device=dev, trainable=True), **kw))
    x = fd.Signal(X, device=dev, plastic=True)
    e = effect(L) if effect else None
    try:
        if tf:
            t = fd.ThreeFactor(L, seed=0)
            o = t(X, seed=2); (o * o).sum().retrograde()
            return [B.numpy(o.data), B.numpy(t.assign(o))]
        y = L(x, seed=2)
        (y * y).sum().retrograde()
        return [B.numpy(y.data), B.numpy(L.log_scale.retro), B.numpy(x.retro)]
    finally:
        if e is not None:
            e.remove()


CASES = {
    "기본": dict(kw={}), "legacy": dict(kw=dict(timing="legacy")), "regular": dict(kw=dict(input_mode="regular")),
    "ckpt·절단·잡음": dict(kw=dict(ckpt=50, truncate=60, noise=0.7)), "감쇠 0.3": dict(kw=dict(damp=0.3)),
    "count_from": dict(kw=dict(count_from_ms=10.0)),
    "activate": dict(kw={}, effect=lambda L: G.activate(L, G.driver(c, group="H"), hz=80)),
    "silence": dict(kw={}, effect=lambda L: G.silence(L, G.driver(c, group="H"))),
    "block": dict(kw={}, effect=lambda L: G.block(L, G.driver(c, group="H"))),
    "mosaic": dict(kw={}, effect=lambda L: G.mosaic(L, p=0.4)),
    "ThreeFactor": dict(kw={}, tf=True), "neuron_params": dict(kw=dict(t_mbr={"H": 12.0}, train_neurons=True)),
    "LIF 플러그인": dict(kw=dict(neuron=fd.neurons.LIF())), "graded": dict(kw=dict(neuron="graded")),
}
def seq_activate(dev):
    """크기가 같은 집단을 차례로 활성화 (캐시가 다른 집단 것을 쓰는지)"""
    H = c.groups["H"]
    out = []
    for i in (0, 5, 0, 5, 10, 0):
        L = fd.Connectome(c, "IN", "O", t_ms=30, device=dev)
        with G.activate(L, G.Line(c, H[i:i + 5], str(i)), hz=300), fd.quiescent():
            out.append(B.numpy(L(np.zeros((2, 6), np.float32), seed=1, return_all=True).data))
    return out


devs = ["cpu"] + (["gpu"] if B.gpu_available() else [])
res = {f"{d}|{n}|{i}": a for d in devs for n, case in CASES.items() for i, a in enumerate(run(d, **case))}
res.update({f"{d}|연속 활성화|{i}": a for d in devs for i, a in enumerate(seq_activate(d))})
if len(devs) == 2:                                                          # CPU = GPU (활성화 집단이 같은지)
    for i in range(6):
        if not np.array_equal(res[f"cpu|연속 활성화|{i}"], res[f"gpu|연속 활성화|{i}"]):
            print("  CPU ≠ GPU 연속 활성화", i)
if sys.argv[1] == "save":
    np.savez(PATH, **res)
    print("저장", len(res))
else:
    ref = np.load(PATH)
    bad = [k for k in ref.files if not np.array_equal(ref[k], res[k])]
    for k in bad:
        print("  다름:", k, float(np.abs(ref[k] - res[k]).max()))
    print(f"비교 {len(ref.files)}개 중 다름 {len(bad)}개")
