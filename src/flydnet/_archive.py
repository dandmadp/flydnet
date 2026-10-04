"""저장 파일 (np.savez) 공통: 형식 버전 기록, 안전한 쓰기, 알기 쉬운 읽기 오류

파일마다 "__flydnet__" 항목에 {"kind": 무엇, "format": 형식 버전, "version": 저장한 flydnet}을 JSON으로 넣는다.
  - format은 패키지 버전과 따로 센다: 파일 구조가 바뀔 때만 올림 (FORMAT)
  - 더 새 형식의 파일은 "flydnet을 업데이트하라"는 오류, 이 항목이 없는 파일은 0.1.15 이전 형식(format 0)으로 읽음
  - 쓰기는 임시 파일에 쓴 뒤 바꿔치기 → 저장 도중 멈춰도 예전 파일이 깨지지 않음
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np

FORMAT = 1
_KEY = "__flydnet__"


def _final(path) -> Path:
    """np.savez처럼 확장자가 없으면 .npz를 붙인 경로"""
    p = Path(path).expanduser()                                     # "~/model" → 홈 폴더
    return p if p.suffix == ".npz" else p.with_name(p.name + ".npz")


def write(path, kind: str, arrays: dict) -> Path:
    """arrays를 path에 저장 (형식 정보 포함, 안전한 쓰기). 실제 저장한 경로 반환"""
    from . import __version__
    if _KEY in arrays:
        raise ValueError(f"{_KEY}는 예약된 이름")
    final = _final(path)
    final.parent.mkdir(parents=True, exist_ok=True)                 # 없는 폴더는 만듦
    meta = json.dumps({"kind": kind, "format": FORMAT, "version": __version__})
    tmp = final.with_name(final.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **{_KEY: np.array(meta)}, **arrays)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    return final


def read(path, kind: str | None = None) -> tuple[dict, dict]:
    """(배열 dict, 형식 정보) 읽기. kind를 주면 다른 종류의 파일일 때 오류"""
    p = Path(path).expanduser()
    if not p.exists() and _final(p).exists():
        p = _final(p)
    if not p.exists():
        raise FileNotFoundError(f"저장 파일이 없음: {p}")
    try:
        with np.load(p, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
        raise ValueError(f"{p}: flydnet 저장 파일(npz)이 아니거나 손상됨 ({type(e).__name__}: {e})") from e
    raw = d.pop(_KEY, None)
    meta = json.loads(str(raw)) if raw is not None else {"kind": None, "format": 0, "version": "0.1.15 이전"}
    if meta["format"] > FORMAT:
        from . import __version__
        raise ValueError(f"{p}: 더 새 flydnet({meta['version']})으로 저장한 파일 (형식 {meta['format']}, "
                         f"지금 flydnet {__version__}은 형식 {FORMAT}까지 읽음) - pip install -U flydnet")
    if kind is not None and meta["kind"] is not None and meta["kind"] != kind:
        raise ValueError(f"{p}: {kind} 파일이 아님 ({meta['kind']}를 저장한 파일, flydnet {meta['version']})")
    return d, meta
