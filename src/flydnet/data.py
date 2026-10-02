"""데이터 위치 설정과 다운로드

데이터 묶음 두 가지
  flywire : FlyWire v783 연결·뉴런 목록 (Shiu et al. 2024 모델 저장소) + 세포 유형 주석 (Schlegel et al. 2024)
  door    : DoOR 2.0 냄새 반응 (Münch & Galizia 2016, CC BY-SA 4.0)

위치를 찾는 순서 (묶음마다)
  1. 함수에 직접 준 경로
  2. 환경변수  FLYDNET_FLYWIRE / FLYDNET_DOOR  (FLYDNET_DATA도 flywire로 인정 — 이전 버전 호환)
  3. 설정 파일 ~/.flydnet/config.json  ← fd.set_data_dir()로 저장
  4. 기본값    ~/.flydnet/data/<묶음>

    import flydnet as fd
    fd.download()                                   # 없는 파일만 받음 (약 140 MB)
    fd.set_data_dir(flywire=r"D:\\my\\flywire")      # 이미 받아 둔 곳을 쓰려면
    fd.data_status()                                # 어디서 무엇을 찾았는지
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

CONFIG = Path.home() / ".flydnet" / "config.json"
DEFAULT_ROOT = Path.home() / ".flydnet" / "data"

# 파일 이름 → (URL, 바이트 크기). 크기로 다운로드가 온전한지 확인
SOURCES = {
    "flywire": {
        "Connectivity_783.parquet":
            ("https://github.com/philshiu/Drosophila_brain_model/raw/main/Connectivity_783.parquet", 100804642),
        "Completeness_783.csv":
            ("https://github.com/philshiu/Drosophila_brain_model/raw/main/Completeness_783.csv", 3327347),
        "flywire_annotations.tsv":
            ("https://github.com/flyconnectome/flywire_annotations/raw/main/supplemental_files/"
             "Supplemental_file1_neuron_annotations.tsv", 31720298),
    },
    "door": {
        name: (f"https://raw.githubusercontent.com/ropensci/DoOR.data/master/data/{name}", None)
        for name in ("door_response_matrix.csv", "door_mappings.csv", "odor.csv")
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
            f"  받기: flydnet.download('{kind}')  또는  python -m flydnet.data {kind}\n"
            f"  이미 있으면: flydnet.set_data_dir({kind}=r'경로') 또는 환경변수 {_ENV[kind][0]}")
    return d


def download(kinds=("flywire", "door"), path=None, overwrite: bool = False, quiet: bool = False) -> dict:
    """없는 파일만 받음. 반환: {묶음: 폴더}"""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    out = {}
    for kind in kinds:
        d = data_dir(kind, path if len(kinds) == 1 else None)
        d.mkdir(parents=True, exist_ok=True)
        for name, (url, size) in SOURCES[kind].items():
            f = d / name
            if f.exists() and not overwrite and (size is None or f.stat().st_size == size):
                if not quiet:
                    print(f"  있음  {f}")
                continue
            tmp = f.with_suffix(f.suffix + ".part")
            if not quiet:
                print(f"  받는 중 {name} ← {url}", flush=True)
            _fetch(url, tmp, quiet)
            if size is not None and tmp.stat().st_size != size:
                got = tmp.stat().st_size
                tmp.unlink()
                raise IOError(f"{name} 크기가 다름 (받은 {got:,} B, 기대 {size:,} B) — 다시 시도")
            tmp.replace(f)
        if not quiet:
            print(f"[{kind}] {d}\n  출처: {CITATIONS[kind]}")
        out[kind] = d
    return out


def _fetch(url: str, dest: Path, quiet: bool):
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, max(total // 10, 1)
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if not quiet and total and done // step != (done - len(chunk)) // step:
                print(f"    {done / total * 100:3.0f}%", flush=True)


def data_status() -> dict:
    """묶음별 위치와 빠진 파일"""
    st = {k: dict(dir=str(data_dir(k)), missing=missing(k)) for k in SOURCES}
    for k, v in st.items():
        print(f"{k:<8} {'준비됨' if not v['missing'] else '빠짐: ' + ', '.join(v['missing'])}  ({v['dir']})")
    return st


if __name__ == "__main__":                               # python -m flydnet.data [flywire|door]
    import sys
    download(sys.argv[1:] or ("flywire", "door"))
