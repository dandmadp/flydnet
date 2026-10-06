"""성능 측정: 전체 뇌 ConnectomeLayer 학습 1스텝 (순전파 + 역전파 + 가소성) 과 시뮬레이션 (역전파 없음)

  python scripts/bench.py                  # GPU, 약 2분
  python scripts/bench.py --torch          # + torch 모델 안에서 연결 장치(fd.torch.bridge)로 학습 1스텝

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
    ap.add_argument("--torch", action="store_true", help="연결 장치 대 torch판 복사본 (배치 8)")
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
    if a.torch:
        bench_torch(brain, a)


def bench_torch(brain, a):
    import torch
    kw = dict(t_ms=a.t_ms, trainable=True, checkpoint_every=20)
    makers = [("연결 장치 (자체 엔진)", lambda: fd.torch.bridge(fd.ConnectomeLayer(brain, "sensory", "descending", **kw), seed=0))]
    for name, make in makers:
        m = make()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x = torch.rand(8, len(brain.groups["sensory"]), device="cuda") * 100
        ts = []
        for r in range(a.repeat + 1):
            t = time.perf_counter()
            opt.zero_grad(); (m(x) * 0.01).sum().backward(); opt.step(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        print(f"torch 모델 안 {name:<20} 배치 8: 학습 1스텝 {np.median(ts[1:]):.2f}초", flush=True)
        del m, opt
        torch.cuda.empty_cache(); B.gpu_memory_peak_reset()


if __name__ == "__main__":
    main()
