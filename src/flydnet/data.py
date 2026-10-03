"""데이터 위치 설정과 다운로드

데이터 묶음 두 가지
  flywire : FlyWire v783 연결·뉴런 목록 (Shiu et al. 2024 모델 저장소) + 세포 유형 주석 (Schlegel et al. 2024)
  door    : DoOR 2.0 냄새 반응 (Münch & Galizia 2016, CC BY-SA 4.0)

위치를 찾는 순서 (묶음마다)
  1. 함수에 직접 준 경로
  2. 환경변수  FLYDNET_FLYWIRE / FLYDNET_DOOR  (FLYDNET_DATA도 flywire로 인정 - 이전 버전 호환)
  3. 설정 파일 ~/.flydnet/config.json  ← fd.set_data_dir()로 저장
  4. 기본값    ~/.flydnet/data/<묶음>

    import flydnet as fd
    fd.download()                                   # 없는 파일만 받음 (약 140 MB)
    fd.set_data_dir(flywire=r"D:\\my\\flywire")      # 이미 받아 둔 곳을 쓰려면
    fd.data_status()                                # 어디서 무엇을 찾았는지
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from ._console import say
from pathlib import Path

CONFIG = Path.home() / ".flydnet" / "config.json"
DEFAULT_ROOT = Path.home() / ".flydnet" / "data"

# 원본 저장소의 특정 커밋에 고정한 주소 → 원본이 나중에 바뀌거나 main이 움직여도 항상 같은 파일.
# 받은 뒤 크기와 SHA-256으로 내용까지 확인. 데이터 버전을 올릴 때는 커밋·크기·해시를 함께 바꿀 것
_GH = "https://raw.githubusercontent.com"
_SHIU = f"{_GH}/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
_ANN = f"{_GH}/flyconnectome/flywire_annotations/a83b2776d60d5764cef36b927f5f9679c16c47a2"
_DOOR = f"{_GH}/ropensci/DoOR.data/db323a496577c4b4a72b5c2fcd1859e07521ffb5/data"
# 파일 이름 → (URL, 바이트 크기, SHA-256)
SOURCES = {
    "flywire": {
        "Connectivity_783.parquet": (f"{_SHIU}/Connectivity_783.parquet", 100804642,
                                     "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
        "Completeness_783.csv": (f"{_SHIU}/Completeness_783.csv", 3327347,
                                 "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
        "flywire_annotations.tsv": (f"{_ANN}/supplemental_files/Supplemental_file1_neuron_annotations.tsv", 31720298,
                                    "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd"),
    },
    "door": {
        "door_response_matrix.csv": (f"{_DOOR}/door_response_matrix.csv", 295852,
                                     "bc2aa5414ff54d3a1399f5fe171e154bcadc754bcf323a3c27711a54f70848e2"),
        "door_mappings.csv": (f"{_DOOR}/door_mappings.csv", 12824,
                              "1197c492e769b1b587c907c5b750ffa2507d7e6c9fe9dfcb2ee8603871e00912"),
        "odor.csv": (f"{_DOOR}/odor.csv", 155880,
                     "a31d1841cf90ce23ec149760a5efa38eae02de7820bb2300c3a7dba221745940"),
    },
}
CITATIONS = {
    "flywire": "FlyWire: Dorkenwald et al. 2024, Schlegel et al. 2024 (Nature); "
               "연결 파일: Shiu et al. 2024 (Nature), github.com/philshiu/Drosophila_brain_model (MIT)",
    "door": "DoOR 2.0: Münch & Galizia 2016 (Sci Rep 6:21841), github.com/ropensci/DoOR.data (CC BY-SA 4.0)",
}
_ENV = {"flywire": ("FLYDNET_FLYWIRE", "FLYDNET_DATA"), "door": ("FLYDNET_DOOR",)}


def _config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def data_dir(kind: str = "flywire", path: str | Path | None = None) -> Path:
    """묶음(kind)의 데이터 폴더. 파일이 실제로 있는지는 확인하지 않음 (require()가 확인)"""
    if kind not in SOURCES:
        raise ValueError(f"kind는 {list(SOURCES)} 중 하나")
    if path is not None:
        return Path(path)
    for env in _ENV[kind]:
        if os.environ.get(env):
            return Path(os.environ[env])
    if kind in _config():
        return Path(_config()[kind])
    return DEFAULT_ROOT / kind


def set_data_dir(flywire: str | Path | None = None, door: str | Path | None = None):
    """이 컴퓨터에서 쓸 데이터 위치를 ~/.flydnet/config.json에 저장 (None인 항목은 그대로)"""
    cfg = _config()
    for kind, p in (("flywire", flywire), ("door", door)):
        if p is not None:
            cfg[kind] = str(Path(p).resolve())
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg


def missing(kind: str = "flywire", path=None) -> list[str]:
    d = data_dir(kind, path)
    return [f for f in SOURCES[kind] if not (d / f).exists()]


def require(kind: str = "flywire", path=None) -> Path:
    """데이터 폴더를 돌려주되, 파일이 없으면 무엇을 하면 되는지 알려주는 오류"""
    d = data_dir(kind, path)
    lack = missing(kind, path)
    if lack:
        raise FileNotFoundError(
            f"{kind} 데이터 파일이 없음: {lack}\n  찾은 위치: {d}\n"
            f"  받기: flydnet.download('{kind}')  또는  python -m flydnet download {kind}\n"
            f"  이미 있으면: flydnet.set_data_dir({kind}=r'경로') 또는 환경변수 {_ENV[kind][0]}")
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify(kind: str = "flywire", path=None) -> dict:
    """파일마다 'ok' / 'missing' / 'size' (크기 다름) / 'sha256' (내용 다름)"""
    d = data_dir(kind, path)
    out = {}
    for name, (_, size, sha) in SOURCES[kind].items():
        f = d / name
        out[name] = ("missing" if not f.exists() else "size" if f.stat().st_size != size else
                     "sha256" if _sha256(f) != sha else "ok")
    return out


def download(kinds=("flywire", "door"), path=None, overwrite: bool = False, quiet: bool = False) -> dict:
    """없거나 내용이 다른 파일만 받음. 받은 파일은 크기와 SHA-256으로 확인. 반환: {묶음: 폴더}"""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    out = {}
    for kind in kinds:
        d = data_dir(kind, path if len(kinds) == 1 else None)
        d.mkdir(parents=True, exist_ok=True)
        state = verify(kind, d)
        for name, (url, size, sha) in SOURCES[kind].items():
            f = d / name
            if state[name] == "ok" and not overwrite:
                if not quiet:
                    say(f"  있음  {f}")
                continue
            if state[name] in ("size", "sha256") and not quiet:
                say(f"  {name}: 기대한 버전과 내용이 달라 다시 받음")
            tmp = f.with_suffix(f.suffix + ".part")
            if not quiet:
                say(f"  받는 중 {name}", flush=True)
            _fetch(url, tmp, quiet)
            got = tmp.stat().st_size
            if got != size or _sha256(tmp) != sha:
                tmp.unlink()
                raise IOError(f"{name}: 받은 파일이 기대와 다름 (크기 {got:,} B, 기대 {size:,} B). "
                              f"네트워크 문제일 수 있으니 다시 시도. 계속되면 원본 주소 확인: {url}")
            tmp.replace(f)
        if not quiet:
            say(f"[{kind}] {d}\n  출처: {CITATIONS[kind]}")
        out[kind] = d
    return out


def _fetch(url: str, dest: Path, quiet: bool):
    """진행률은 한 줄에서 숫자만 바뀌게 (터미널·Colab 모두)"""
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(done / total * 100) if total else -1
            if not quiet and total and pct != shown and (pct % 5 == 0 or done == total):
                say(f"\r    {pct:3d}%  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                shown = pct
        if not quiet and total:
            say()


def data_status() -> dict:
    """묶음별 위치와 빠진 파일"""
    st = {k: dict(dir=str(data_dir(k)), missing=missing(k)) for k in SOURCES}
    for k, v in st.items():
        say(f"{k:<8} {'준비됨' if not v['missing'] else '빠짐: ' + ', '.join(v['missing'])}  ({v['dir']})")
    return st
