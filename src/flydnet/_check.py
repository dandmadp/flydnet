"""인자 검증 (모든 공개 함수가 같은 규칙·같은 메시지로). 잘못된 값이 조용히 통과해 결과가 틀어지지 않게"""
from __future__ import annotations

import math
import numbers


def _num(name, v):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{name}는 숫자: {v!r}")
    if not math.isfinite(float(v)):
        raise ValueError(f"{name}는 유한한 숫자 (NaN·무한대 안 됨): {v!r}")
    return float(v)


def finite(name, v):
    return _num(name, v)


def pos(name, v):
    """0보다 큼"""
    if _num(name, v) <= 0:
        raise ValueError(f"{name}는 0보다 커야 함: {v!r}")
    return v


def nonneg(name, v):
    """0 이상"""
    if _num(name, v) < 0:
        raise ValueError(f"{name}는 0 이상: {v!r}")
    return v


def unit(name, v, lo_open: bool = False, hi_open: bool = False):
    """0 ~ 1 (lo_open이면 0 제외, hi_open이면 1 제외)"""
    x = _num(name, v)
    if x < 0 or x > 1 or (lo_open and x == 0) or (hi_open and x == 1):
        rng = f"{'(' if lo_open else '['}0, 1{')' if hi_open else ']'}"
        raise ValueError(f"{name}는 {rng} 범위: {v!r}")
    return v


def integer(name, v, lo: int = 1):
    """lo 이상의 정수 (2.5·'3' 같은 값은 안 됨)"""
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{name}는 정수: {v!r}")
    if v < lo:
        raise ValueError(f"{name}는 {lo} 이상의 정수: {v!r}")
    return int(v)


def optional(check, name, v, *a, **k):
    return v if v is None else check(name, v, *a, **k)


# LIF 매개변수 (ConnectomeLayer params): 이름 → 규칙
LIF_RULES = {"v_0": finite, "v_rst": finite, "v_th": finite, "t_mbr": pos, "tau": pos, "t_rfc": nonneg,
             "t_dly": nonneg, "w_syn": finite, "f_poi": nonneg, "dt": pos, "r_max": pos}


def lif_params(params: dict | None, defaults: dict):
    """알 수 없는 이름(오타)·잘못된 값·문턱 ≤ 리셋을 막음"""
    if not params:
        return
    import difflib
    unknown = [k for k in params if k not in LIF_RULES]
    if unknown:
        hints = {k: difflib.get_close_matches(k, list(LIF_RULES), n=1) for k in unknown}
        raise KeyError("알 수 없는 뉴런 매개변수 " + ", ".join(f"{k!r}" + (f" (혹시 {h[0]!r}?)" if h else "")
                                                       for k, h in hints.items())
                       + f" - 쓸 수 있는 것: {sorted(LIF_RULES)}")
    for k, v in params.items():
        LIF_RULES[k](f"params[{k!r}]", v)
    p = dict(defaults, **params)
    if p["v_th"] <= p["v_rst"]:
        raise ValueError(f"문턱 v_th({p['v_th']})는 리셋 전위 v_rst({p['v_rst']})보다 커야 함 - 아니면 발화 판정이 뒤집힘")
