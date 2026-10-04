"""성능 측정: 전체 뇌 ConnectomeLayer 학습 1스텝 (순전파 + 역전파 + 가소성) 과 시뮬레이션 (역전파 없음)

  python scripts/bench.py                  # GPU, 약 2분

설정: 전체 뇌 138,639개 뉴런·연결 1,509만, 입력 = 감각 뉴런, 출력 = 하행 뉴런, 연결마다 학습, 20 ms (200스텝),
체크포인팅 20스텝마다
"""
import argparse
import time

import numpy as np

import flydnet as fd
from flydnet.ganglion import backend as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--t-ms", type=float, default=20.0)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    B.limit_gpu_memory(0.75)
    brain = fd.Circuit.whole_brain()
    print(brain)
    for timing in ("brian", "legacy"):
        layer = fd.ConnectomeLayer(brain, "sensory", "descending", t_ms=a.t_ms, trainable=True, checkpoint_every=20,
                                   timing=timing)
        rule = fd.AdaptivePlasticity(layer.synapses(), rate=1e-3)
        for bn in map(int, a.batches.split(",")):
            x = np.random.default_rng(0).uniform(0, 100, (bn, layer.n_in)).astype(np.float32)
            y = np.zeros(bn, np.int64)
            times = []
            try:
                for r in range(a.repeat + 1):                          # 첫 번은 준비 (커널 컴파일)
                    t = time.perf_counter()
                    out = layer(x, seed=r)
                    loss = fd.surprise(out * 0.01, y)
                    rule.clear(); loss.retrograde(); rule.step()
                    B.xp(layer.device).cuda.Device().synchronize() if layer.device == "gpu" else None
                    times.append(time.perf_counter() - t)
            except B.GPUMemoryError:
                print(f"timing {timing:<6} 배치 {bn:>3}: GPU 메모리 부족 (상한 75%)", flush=True)
                rule.clear(); B.gpu_memory_peak_reset()
                continue
            with fd.quiescent():
                t = time.perf_counter()
                layer(x, seed=0).numpy()
                sim = time.perf_counter() - t
            print(f"timing {timing:<6} 배치 {bn:>3}: 학습 1스텝 {np.median(times[1:]):.2f}초, 시뮬레이션만 {sim:.2f}초, "
                  f"GPU 메모리 {B.gpu_memory_used() / 2**30:.1f} GB", flush=True)
        del layer, rule
        B.gpu_memory_peak_reset()


if __name__ == "__main__":
    main()
