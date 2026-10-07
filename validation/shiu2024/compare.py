"""flydnet(timing="brian") 대 원본 Brian2: 조건별 MN9 발화율 표. flydnet 쪽을 새로 돌리려면 --run (GPU 약 3분)"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
CONDS = ["none_0", "sugar_25", "sugar_50", "sugar_100", "sugar_200", "bitter_100", "sugarbitter_100"]


def run_flydnet(trials=30):
    import flydnet as fd
    from run_brian2 import MN9, SUGAR
    fd.ganglion.limit_gpu_memory(0.75)
    brain = fd.Circuit.whole_brain()
    G = fd.genetics
    sugar = G.driver(brain, root_ids=SUGAR, missing="ignore")
    bitter = G.driver(brain, root_ids=json.loads((HERE / "bitter_ids.json").read_text(encoding="utf-8")))
    mn9 = G.driver(brain, root_ids=[MN9]).idx[0]
    layer = fd.ConnectomeLayer(brain, inputs=None, outputs="motor", t_ms=1000)
    res = {}
    for name in CONDS:
        kind, hz = name.split("_")[0], float(name.split("_")[1])
        lines = {"none": [], "sugar": [sugar], "bitter": [bitter], "sugarbitter": [sugar, bitter]}[kind]
        exprs = [G.activate(layer, line, hz=hz) for line in lines]
        with fd.quiescent():
            res[name] = layer(None, seed=0, batch=trials, return_all=True).numpy()[:, mn9].tolist()
        for e in exprs:
            e.remove()
        print(name, np.mean(res[name]), flush=True)
    (HERE / "flydnet_results.json").write_text(json.dumps(res), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="flydnet 결과를 새로 계산")
    a = ap.parse_args()
    if a.run:
        run_flydnet()
    br = json.loads((HERE / "brian2_results.json").read_text(encoding="utf-8"))
    fl = json.loads((HERE / "flydnet_results.json").read_text(encoding="utf-8"))
    se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
    print(f"{'조건':<16}{'Brian2':>14}{'flydnet':>14}{'p (Welch)':>11}")
    for k in CONDS:
        b, f = np.array(br[k]), np.array(fl[k])
        p = stats.ttest_ind(b, f, equal_var=False).pvalue if b.std() + f.std() > 0 else float("nan")
        print(f"{k:<16}{b.mean():>8.1f} ±{se(b):4.1f}{f.mean():>8.1f} ±{se(f):4.1f}{p:>11.3f}")


if __name__ == "__main__":
    main()
