"""데이터 위치 설정과 다운로드

데이터 묶음
  flywire : FlyWire v783 연결·뉴런 목록 (Shiu et al. 2024 모델 저장소) + 세포 유형 주석 (Schlegel et al. 2024)
  door    : DoOR 2.0 냄새 반응 (Münch & Galizia 2016, CC BY-SA 4.0)
  worm    : 예쁜꼬마선충 자웅동체 커넥톰 (Cook et al. 2019, OpenWorm c302 저장소, MIT) - 기본 다운로드에는 없음

위치를 찾는 순서 (묶음마다)
  1. 함수에 직접 준 경로
  2. 환경변수  FLYDNET_FLYWIRE / FLYDNET_DOOR  (FLYDNET_DATA도 flywire로 인정 - 이전 버전 호환)
  3. 설정 파일 ~/.flydnet/config.json  ← fd.set_data_dir()로 저장
  4. 기본값    ~/.flydnet/data/<묶음>

    import flydnet as fd
    fd.download()                                   # 없는 파일만 받음 (약 140 MB)
    fd.set_data_dir(flywire=r"D:\\my\\flywire")      # 이미 받아 둔 곳을 쓰려면
    fd.data_status()                                # 어디서 무엇을 찾았는지

연결 parquet는 받은 뒤 Connectivity_783.npz로 한 번 바꿔 두고 그것을 읽음 (pyarrow 필요 없음, read_connectivity)
"""
from __future__ import annotations

import hashlib
import http.client
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
_WORM = f"{_GH}/openworm/c302/49acae1570131b2592220d5d67750dee8d7ff596/c302/data"
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
    "worm": {
        "herm_full_edgelist.csv": (f"{_WORM}/herm_full_edgelist.csv", 245463,
                                   "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca"),
    },
}
# Connectivity_783.parquet를 numpy 형식(pre, post, weight)으로 한 번 바꿔 둔 파일 → pyarrow 없이 읽음.
# download가 만들고, parquet를 처음 읽을 때도 만들어 둠. 압축된 파일 바이트는 zlib 버전마다 다를 수 있으므로
# 배열 내용(pre, post, weight를 차례로 int32 리틀 엔디언으로 이은 것)의 SHA-256으로 확인
CONNECTIVITY = "Connectivity_783.parquet"
CONNECTIVITY_NPZ = "Connectivity_783.npz"
_NPZ_EDGES = 15091983
_NPZ_SHA256 = "31150327a1370ae516441aebf83a7e87fc7cb4ec48e31f71f3878361f2ce0d89"
CITATIONS = {
    "flywire": "FlyWire: Dorkenwald et al. 2024, Schlegel et al. 2024 (Nature); "
               "연결 파일: Shiu et al. 2024 (Nature), github.com/philshiu/Drosophila_brain_model (MIT)",
    "door": "DoOR 2.0: Münch & Galizia 2016 (Sci Rep 6:21841), github.com/ropensci/DoOR.data (CC BY-SA 4.0)",
    "worm": "C. elegans: Cook et al. 2019 (Nature 571:63), github.com/openworm/c302 (MIT); "
            "GABA 뉴런: McIntire et al. 1993 (Nature 364:337)",
}
_ENV = {"flywire": ("FLYDNET_FLYWIRE", "FLYDNET_DATA"), "door": ("FLYDNET_DOOR",), "worm": ("FLYDNET_WORM",)}


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
        return Path(path).expanduser()
    for env in _ENV[kind]:
        if os.environ.get(env):
            return Path(os.environ[env]).expanduser()
    if kind in _config():
        return Path(_config()[kind]).expanduser()
    return DEFAULT_ROOT / kind


def set_data_dir(flywire: str | Path | None = None, door: str | Path | None = None, worm: str | Path | None = None):
    """이 컴퓨터에서 쓸 데이터 위치를 ~/.flydnet/config.json에 저장 (None인 항목은 그대로)"""
    cfg = _config()
    for kind, p in (("flywire", flywire), ("door", door), ("worm", worm)):
        if p is not None:
            cfg[kind] = str(Path(p).expanduser().resolve())
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg


def missing(kind: str = "flywire", path=None) -> list[str]:
    d = data_dir(kind, path)
    return [f for f in SOURCES[kind] if not (d / f).exists() and not _npz_instead(kind, f, d)]


def _npz_instead(kind: str, name: str, d: Path) -> bool:
    """연결 parquet가 없어도 바꿔 둔 npz가 있으면 됨 (npz만 복사해 온 컴퓨터 등)"""
    return kind == "flywire" and name == CONNECTIVITY and (d / CONNECTIVITY_NPZ).exists()


def require(kind: str = "flywire", path=None) -> Path:
    """데이터 폴더를 돌려주되, 파일이 없으면 무엇을 하면 되는지 알려주는 오류"""
    d = data_dir(kind, path)
    lack = missing(kind, path)
    if not lack:                                       # 크기 확인 (해시보다 싸고, 받다 끊기거나 다른 버전인 파일을 잡음)
        bad = [f"{n} ({(d / n).stat().st_size:,} B, 기대 {size:,} B)" for n, (_, size, _) in SOURCES[kind].items()
               if (d / n).exists() and (d / n).stat().st_size != size]
        if bad:
            raise ValueError(f"{kind} 데이터 파일 크기가 다름: {bad}\n  찾은 위치: {d}\n  받다가 끊겼거나 다른 버전 - "
                             f"다시 받기: python -m flydnet download {kind}  (내용 확인: python -m flydnet verify)")
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
    """파일마다 'ok' / 'missing' / 'size' (크기 다름) / 'sha256' (내용 다름) / 'not_needed' (연결 parquet가 없지만
    npz가 있음). flywire는 바꿔 둔 연결 npz도 (배열 내용의 SHA-256)"""
    d = data_dir(kind, path)
    out = {}
    for name, (_, size, sha) in SOURCES[kind].items():
        f = d / name
        out[name] = ("not_needed" if not f.exists() and _npz_instead(kind, name, d) else
                     "missing" if not f.exists() else "size" if f.stat().st_size != size else
                     "sha256" if _sha256(f) != sha else "ok")
    if kind == "flywire":
        out[CONNECTIVITY_NPZ] = _verify_npz(d / CONNECTIVITY_NPZ)
    return out


def _npz_digest(pre, post, weight) -> str:
    import numpy as np
    h = hashlib.sha256()
    for a in (pre, post, weight):
        h.update(np.ascontiguousarray(a, dtype="<i4").tobytes())
    return h.hexdigest()


def _load_npz(f: Path):
    """(pre, post, weight) 배열 - 형식이 다르면 ValueError"""
    import numpy as np
    try:
        with np.load(f, allow_pickle=False) as z:
            arrs = [z[k] for k in ("pre", "post", "weight")]
    except (OSError, KeyError, ValueError, EOFError) as e:
        raise ValueError(f"{f.name}을 읽을 수 없음 ({type(e).__name__}: {e})") from e
    if len({len(a) for a in arrs}) != 1 or any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrs):
        raise ValueError(f"{f.name}: pre·post·weight가 같은 길이의 1차원 정수 배열이 아님")
    return arrs


def _verify_npz(f: Path) -> str:
    if not f.exists():
        return "missing"
    try:
        arrs = _load_npz(f)
    except ValueError:
        return "sha256"
    return "size" if len(arrs[0]) != _NPZ_EDGES else "sha256" if _npz_digest(*arrs) != _NPZ_SHA256 else "ok"


def _read_parquet(f: Path):
    """연결 parquet → (pre, post, weight) int64. weight = 시냅스 수 x 부호 (흥분 +1 / 억제 -1)"""
    import pandas as pd
    try:
        df = pd.read_parquet(f, columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"])
    except ImportError as e:
        raise ImportError(
            f"{f.name}를 읽으려면 pyarrow가 필요함 (연결을 numpy 형식으로 바꿔 둔 {CONNECTIVITY_NPZ}가 없음).\n"
            "  한 번만: pip install pyarrow  →  python -m flydnet download flywire  (npz로 바꿔 둠, 그 뒤에는 pyarrow 필요 없음)\n"
            f"  또는 다른 컴퓨터에서 만든 {CONNECTIVITY_NPZ}를 같은 폴더({f.parent})에 복사") from e
    return (df.Presynaptic_Index.to_numpy(), df.Postsynaptic_Index.to_numpy(),
            df.Connectivity.to_numpy() * df.Excitatory.to_numpy())


def convert_connectivity(path=None, quiet: bool = False) -> Path:
    """FlyWire 연결 parquet → npz (한 번만, pyarrow 필요). 내용 해시를 확인한 뒤에만 저장. 반환: npz 경로"""
    import numpy as np
    d = data_dir("flywire", path)
    pre, post, w = _read_parquet(d / CONNECTIVITY)
    if len(pre) != _NPZ_EDGES or _npz_digest(pre, post, w) != _NPZ_SHA256:
        raise ValueError(f"{CONNECTIVITY}의 내용이 기대한 버전과 다름 - 다시 받기: python -m flydnet download flywire")
    if not quiet:
        say(f"  {CONNECTIVITY} → {CONNECTIVITY_NPZ} (한 번만, 약 10초)", flush=True)
    dest = d / CONNECTIVITY_NPZ
    tmp = d / f"{CONNECTIVITY_NPZ}.{os.getpid()}.part.npz"           # 여러 프로세스가 동시에 바꿔도 반쯤 쓴 파일을 읽지 않게
    try:
        np.savez_compressed(tmp, pre=pre.astype(np.int32), post=post.astype(np.int32), weight=w.astype(np.int32))
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def read_connectivity(path=None, name: str = CONNECTIVITY):
    """FlyWire 연결 (pre, post, weight) int64 배열 (pre·post = 전체 뇌 번호, weight = 시냅스 수 x 부호).
    바꿔 둔 npz가 있으면 그것 (pyarrow 필요 없음), 없으면 parquet를 읽고 npz도 만들어 둠.
    name: 다른 연결 파일 (.parquet 또는 pre·post·weight가 든 .npz)"""
    import numpy as np
    d = data_dir("flywire", path)
    if name.endswith(".npz"):
        return tuple(a.astype(np.int64) for a in _load_npz(d / name))
    if name != CONNECTIVITY:
        return _read_parquet(d / name)
    f = d / CONNECTIVITY_NPZ
    if f.exists():
        try:
            arrs = _load_npz(f)
            if len(arrs[0]) != _NPZ_EDGES:
                raise ValueError(f"{f.name}: 연결 {len(arrs[0]):,}개 (기대 {_NPZ_EDGES:,}개)")
            return tuple(a.astype(np.int64) for a in arrs)
        except ValueError as e:
            if not (d / CONNECTIVITY).exists():
                raise ValueError(f"{e} - 다시 만들기: python -m flydnet download flywire") from e
            import warnings
            warnings.warn(f"{e} - parquet에서 다시 만듦", stacklevel=2)
    out = _read_parquet(d / CONNECTIVITY)
    try:
        convert_connectivity(d)                                         # 다음부터는 pyarrow 없이 (실패해도 읽기는 됨)
    except (OSError, ValueError):
        pass
    return out


def download(kinds=("flywire", "door"), path=None, overwrite: bool = False, quiet: bool = False) -> dict:
    """없거나 내용이 다른 파일만 받음. 받은 파일은 크기와 SHA-256으로 확인. 반환: {묶음: 폴더}
    path: 묶음 하나면 그 폴더, 여러 개면 path/<묶음> (예전: 여러 개면 path를 조용히 무시하고 기본 위치에 받음)"""
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    unknown = [k for k in kinds if k not in SOURCES]
    if unknown:
        raise ValueError(f"모르는 데이터 묶음: {unknown} (있는 것: {list(SOURCES)})")
    out = {}
    for kind in kinds:
        d = data_dir(kind, path if (path is None or len(kinds) == 1) else Path(path).expanduser() / kind)
        d.mkdir(parents=True, exist_ok=True)
        state = verify(kind, d)
        for name, (url, size, sha) in SOURCES[kind].items():
            f = d / name
            if state[name] == "not_needed" or state[name] == "ok" and not overwrite:
                if not quiet:
                    say(f"  있음  {f}")
                continue
            if state[name] in ("size", "sha256") and not quiet:
                say(f"  {name}: 기대한 버전과 내용이 달라 다시 받음")
            tmp = f.with_suffix(f.suffix + ".part")
            if not quiet:
                say(f"  받는 중 {name}", flush=True)
            try:
                _fetch(url, tmp, quiet)
            except (OSError, ValueError, http.client.HTTPException) as e:   # URLError·시간 초과·연결 끊김 (받는 도중
                #                                                         끊기면 IncompleteRead - OSError가 아니라 예전엔 안 잡힘)
                if tmp.exists():
                    tmp.unlink()
                raise IOError(f"{name} 받기 실패 ({type(e).__name__}: {e}) - 인터넷 연결을 확인하고 다시: "
                              f"python -m flydnet download {kind}  (받은 파일만 건너뜀)") from e
            got = tmp.stat().st_size
            if got != size or _sha256(tmp) != sha:
                tmp.unlink()
                raise IOError(f"{name}: 받은 파일이 기대와 다름 (크기 {got:,} B, 기대 {size:,} B). "
                              f"네트워크 문제일 수 있으니 다시 시도. 계속되면 원본 주소 확인: {url}")
            tmp.replace(f)
        if kind == "flywire" and (overwrite or verify(kind, d)[CONNECTIVITY_NPZ] != "ok"):
            try:
                convert_connectivity(d, quiet)
            except ImportError:
                if not quiet:
                    say(f"  pyarrow가 없어 {CONNECTIVITY_NPZ}로 바꾸지 못함 - 회로를 만들려면 한 번만: pip install pyarrow → "
                        "python -m flydnet download flywire (그 뒤에는 pyarrow 필요 없음)")
        if not quiet:
            say(f"[{kind}] {d}\n  출처: {CITATIONS[kind]}")
            if path is not None and d.resolve() != data_dir(kind).resolve():
                say(f"  이 위치를 기본으로 쓰려면: flydnet.set_data_dir({kind}=r'{d}')")
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
