"""원본 Shiu et al. 2024 Brian2 모델로 MN9 발화율 (flydnet과 비교용). flydnet과 다른 가상환경에서 실행:

  python -m venv .venv-brian2 && .venv-brian2\\Scripts\\pip install brian2 "numpy<2.3" pandas pyarrow joblib
  .venv-brian2\\Scripts\\python run_brian2.py --data <FlyWire 데이터 폴더> sugar_100 30

model.py는 flydnet 데이터와 같은 고정 커밋에서 받는다. 조건: none_0, sugar_<Hz>, bitter_<Hz>, sugarbitter_<Hz>
1초 시행 하나에 약 15초 (NumPy 코드 생성, C++ 컴파일러 없이)
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
HERE = Path(__file__).parent
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
         720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
         720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Completeness_783.csv, Connectivity_783.parquet가 있는 폴더")
    ap.add_argument("cond")
    ap.add_argument("trials", type=int)
    a = ap.parse_args()
    if not (HERE / "model.py").exists():
        url = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{COMMIT}/model.py"
        urllib.request.urlretrieve(url, HERE / "model.py")
    import model
    from brian2 import Hz, ms

    data = Path(a.data)
    comp, con = data / "Completeness_783.csv", data / "Connectivity_783.parquet"
    ids = pd.read_csv(comp, index_col=0).index.values
    pos = {r: i for i, r in enumerate(ids)}
    sugar = [pos[r] for r in SUGAR if r in pos]
    bitter = [pos[r] for r in json.load(open(HERE / "bitter_ids.json"))]
    kind, hz = a.cond.split("_")[0], float(a.cond.split("_")[1])
    p = dict(model.default_params)
    p["r_poi"] = p["r_poi2"] = hz * Hz
    exc, exc2 = {"none": ([], []), "sugar": (sugar, []), "bitter": (bitter, []), "sugarbitter": (sugar, bitter)}[kind]
    out = []
    for t in range(a.trials):
        t0 = time.time()
        spk = model.run_trial(exc, exc2, [], comp, con, p)
        out.append(len(spk.get(pos[MN9], [])) / float(p["t_run"] / (1000 * ms)))
        print(f"{a.cond} trial {t}: MN9 {out[-1]:.1f} Hz ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dict(cond=a.cond, rates=out), open(HERE / f"brian2_{a.cond}.json", "w"))
    print(np.mean(out))


if __name__ == "__main__":
    main()
