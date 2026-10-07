"""정답 고정(골든) 파일 만들기: python tests/golden/make_golden.py [이름 ...]

엔진을 바꾸기 전의 결과를 저장 → tests/test_golden.py가 이후 모든 변경의 기준으로 비교.
이미 있는 파일을 덮어쓰려면 --force (골든 값을 바꾸는 것이므로 이유를 커밋에 남길 것)"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402

import flydnet as fd  # noqa: E402

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    force = "--force" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(cases.circuits())
    for name in names:
        path = HERE / f"{name}.npz"
        if path.exists() and not force:
            print(f"{path.name} 있음 - 건너뜀 (덮어쓰려면 --force)")
            continue
        if cases.circuits()[name]["data"] and fd.data.missing("flywire"):
            print(f"{name}: FlyWire 데이터 없음 - 건너뜀")
            continue
        res = cases.compute(name)
        again = cases.compute(name)                                   # 같은 계산을 두 번: 결정적이어야 골든이 됨
        for k in res:
            if not np.array_equal(res[k], again[k]):
                raise SystemExit(f"{name}.{k}: 같은 계산이 두 번 다름 - 골든으로 쓸 수 없음")
        np.savez_compressed(path, **res, _version=np.array(fd.__version__), _numpy=np.array(np.__version__))
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB, {len(res)}개 값")
