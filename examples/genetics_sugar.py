"""가상 유전학으로 Shiu et al. 2024 (Nature) 재현: 당 감지 미각 뉴런(GRN)을 켜면 섭식 운동 뉴런 MN9가 발화하는가

  1) 무자극 / 당 GRN 활성화 (빈도별) / 쓴맛 GRN 활성화 / 둘 다 → MN9 발화율
  2) 유전자 스크린: 당 자극 때 가장 활발한 중간 뉴런을 하나씩 끄고(Kir2.1) MN9가 얼마나 줄어드는지

  python examples/genetics_sugar.py                 # 약 13분 (RTX 5070: 조건 7개 약 3분 + 스크린 약 10분)
  python examples/genetics_sugar.py --quick         # 약 3분 (시행·스크린 축소)

뉴런 ID는 Shiu et al. 저장소(github.com/philshiu/Drosophila_brain_model, example.ipynb)와 같음.
모델 매개변수도 같음 (LIF, dt 0.1 ms, 자극 = 포아송 입력 스파이크, 1초 시행).
"""
import argparse
import time

import numpy as np

import flydnet as fd
from flydnet.ganglion import backend as B

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
         720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
         720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=30, help="조건마다 시행 수 (Shiu et al.: 30)")
    ap.add_argument("--t-ms", type=float, default=1000.0, help="시행 길이 ms (Shiu et al.: 1000)")
    ap.add_argument("--screen", type=int, default=10, help="스크린할 뉴런 수 (당 자극 때 활발한 순)")
    ap.add_argument("--seeds", type=int, default=3, help="스크린의 짝지은 seed 수")
    ap.add_argument("--quick", action="store_true", help="시행 10, 스크린 5개 x seed 2")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.quick:
        a.trials, a.screen, a.seeds = 10, 5, 2
    B.limit_gpu_memory(0.75)
    t0 = time.time()

    brain = fd.Circuit.whole_brain()
    print(brain)
    G = fd.genetics
    sugar = G.driver(brain, root_ids=SUGAR, missing="warn", name="당 GRN (Shiu et al.)")
    mn9 = G.driver(brain, root_ids=[MN9], name="MN9")
    side = brain.meta.side.iloc[sugar.idx].mode()[0]
    bitter = G.driver(brain, cell_sub_class="bitter", side=side, name=f"쓴맛 GRN ({side})")
    print(f"{sugar}, {bitter}, {mn9}")
    layer = fd.ConnectomeLayer(brain, inputs=None, outputs="motor", t_ms=a.t_ms, device=a.device)

    def rates(seed=0, trials=a.trials):
        with fd.quiescent():
            return layer(None, seed=seed, batch=trials, return_all=True).numpy()       # (시행, 뉴런) Hz

    def mn9_hz(r):
        return r[:, mn9.idx[0]]

    print(f"\n1) MN9 발화율 (시행 {a.trials}번 x {a.t_ms:.0f} ms, 평균 ± 표준오차)")
    results = {}
    conds = [("무자극", []), ("당 25 Hz", [(sugar, 25)]), ("당 50 Hz", [(sugar, 50)]), ("당 100 Hz", [(sugar, 100)]),
             ("당 200 Hz", [(sugar, 200)]), ("쓴맛 100 Hz", [(bitter, 100)]),
             ("당 100 + 쓴맛 100 Hz", [(sugar, 100), (bitter, 100)])]
    for name, acts in conds:
        exprs = [G.activate(layer, line, hz=hz) for line, hz in acts]
        try:
            r = rates()
        finally:
            for e in exprs:
                e.remove()
        m = mn9_hz(r)
        results[name] = r
        print(f"  {name:<20} MN9 {m.mean():6.1f} ± {m.std(ddof=1) / np.sqrt(len(m)):4.1f} Hz", flush=True)

    # 2) 스크린: 당 100 Hz 때 가장 활발한 뉴런 (당 GRN 제외)
    r = results["당 100 Hz"].mean(0) - results["무자극"].mean(0)
    r[sugar.idx] = -np.inf
    r[mn9.idx] = -np.inf
    top = np.argsort(r)[::-1][:a.screen]
    meta = brain.meta
    cands = {}
    for i in top:
        label = f"{meta.cell_type.iloc[i] if isinstance(meta.cell_type.iloc[i], str) else meta.super_class.iloc[i]}" \
                f" #{brain.root_ids[i]}"
        cands[label] = G.driver(brain, root_ids=[brain.root_ids[i]], name=label)
    print(f"\n2) 유전자 스크린: 당 100 Hz 때 가장 활발한 뉴런 {a.screen}개를 하나씩 끄기 (MN9 변화, seed {a.seeds}개)")

    def measure(L, seed):
        with G.activate(L, sugar, hz=100):
            return float(mn9_hz(rates(seed, max(5, a.trials // 3))).mean())
    df = G.screen(measure, layer, cands, effector="silence", seeds=a.seeds)
    on = results["당 100 Hz"].mean(0)                                 # r은 위에서 증가량으로 바뀜 - 발화율은 따로
    df["rate_100Hz"] = [float(on[cands[n].idx[0]]) for n in df.line]   # 당 100 Hz 때 그 뉴런의 발화율 (예전: 증가량이 들어갔음)
    df["rise_100Hz"] = [float(r[cands[n].idx[0]]) for n in df.line]   # 무자극 대비 증가량
    print(df[["line", "baseline", "manipulated", "change", "rel_change", "p", "rate_100Hz", "rise_100Hz"]].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
