"""정답 고정(골든) 비교: 엔진 정리(희소 연산 교체 등) 전의 결과(tests/golden/*.npz)와 지금 엔진의 결과가 같은지

허용 오차: 배열마다 max|지금 - 골든| <= 1e-5 x max|골든| (상대 오차 1e-5). 스파이크 시각(lif_spikes)은 정확히 같아야 함.
골든 값을 바꾸거나 허용 오차를 넓히지 말 것 - 다르면 원인을 찾아 보고"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import flydnet as fd

HERE = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(HERE))
import cases  # noqa: E402

RTOL = 1e-5
EXACT = ("lif_spikes",)


def _golden(name):
    p = HERE / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"골든 파일 없음: {p.name} (python tests/golden/make_golden.py)")
    with np.load(p) as z:
        return {k: z[k] for k in z.files if not k.startswith("_")}


@pytest.fixture(scope="module", params=list(cases.circuits()))
def pair(request):
    name = request.param
    if cases.circuits()[name]["data"] and fd.data.missing("flywire"):
        pytest.skip("FlyWire 데이터 없음")
    gold = _golden(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return name, gold, cases.compute(name)


def test_golden_same_keys(pair):
    name, gold, now = pair
    assert set(gold) == set(now), f"{name}: 값 목록이 다름 {set(gold) ^ set(now)}"


def test_golden_values(pair):
    name, gold, now = pair
    bad = []
    for k, g in gold.items():
        a = now[k]
        if a.shape != g.shape:
            bad.append(f"{k}: 모양 {a.shape} 대 골든 {g.shape}")
            continue
        if k in EXACT:
            n = int((a != g).sum())
            if n:
                first = np.argwhere(a != g)[0].tolist()
                bad.append(f"{k}: 스파이크 {n}개 다름 (첫 위치 시료·스텝·뉴런 {first})")
            continue
        scale = float(np.abs(g).max()) if g.size else 0.0
        err = float(np.abs(a.astype(np.float64) - g).max()) if g.size else 0.0
        if err > RTOL * max(scale, 1e-30):
            bad.append(f"{k}: 최대 오차 {err:.3g} (골든 크기 {scale:.3g}, 상대 {err / max(scale, 1e-30):.3g} > {RTOL})")
    assert not bad, f"{name} 골든과 다름:\n  " + "\n  ".join(bad)
