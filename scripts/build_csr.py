"""CPU 희소 행렬 C 커널(src/flydnet/ganglion/_csr.c)을 플랫폼별 공유 라이브러리로 빌드 → src/flydnet/ganglion/_lib/

  python scripts/build_csr.py           # 이 컴퓨터의 플랫폼만
  python scripts/build_csr.py --all     # 배포용: 6개 플랫폼 모두 (zig 교차 빌드, 한 컴퓨터에서 몇 초)

컴파일러는 zig (pip install ziglang - dev 설치에 포함). zig는 다른 도구 없이 Linux·macOS·Windows용을 모두 만듦.
결과 파일은 git에 커밋한다 (휠 하나에 모두 들어감 → CI가 시험한 파일 = 배포하는 파일).
_lib/SOURCE에 _csr.c의 SHA-256과 zig 버전을 적음 - release.py가 소스와 빌드가 어긋나면 막음.
FMA로 합치지 않게 -ffp-contract=off (scipy·numpy 경로와 같은 비트). 파이썬 API를 쓰지 않는 일반 C 라이브러리라
파이썬 버전과 상관없음 (ctypes로 엶)."""
import argparse
import hashlib
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "flydnet" / "ganglion" / "_csr.c"
LIB = SRC.parent / "_lib"
# 플랫폼 이름 (csr.py의 _platform_key와 같게) → zig 대상, 확장자. linux는 glibc 2.17 (manylinux2014와 같은 기준)
TARGETS = {
    "linux-x86_64": ("x86_64-linux-gnu.2.17", ".so"),
    "linux-aarch64": ("aarch64-linux-gnu.2.17", ".so"),
    "macos-x86_64": ("x86_64-macos.11.0", ".dylib"),
    "macos-arm64": ("aarch64-macos.11.0", ".dylib"),
    "windows-x86_64": ("x86_64-windows-gnu", ".dll"),
    "windows-arm64": ("aarch64-windows-gnu", ".dll"),
}
FLAGS = ["-O3", "-mcpu=baseline", "-std=c99", "-ffp-contract=off", "-shared", "-s"]


def platform_key() -> str:
    m = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64" if sys.platform == "darwin" else "aarch64",
            "aarch64": "aarch64"}.get(m, m)
    if sys.platform == "win32":
        arch = "arm64" if m == "arm64" else arch
    osname = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    return f"{osname}-{arch}"


def zig():
    if importlib.util.find_spec("ziglang") is None:
        raise SystemExit("zig가 없음 - pip install ziglang (개발 설치: pip install -e \".[dev]\")")
    return [sys.executable, "-m", "ziglang"]


def build(key):
    target, ext = TARGETS[key]
    out = LIB / f"_csr-{key}{ext}"
    cmd = [*zig(), "cc", "-target", target, *FLAGS, *([] if ext == ".dll" else ["-fPIC"]), "-o", str(out), str(SRC)]
    subprocess.run(cmd, check=True)
    for junk in (out.with_suffix(".lib"), out.with_suffix(".pdb"), LIB / "_csr.lib", LIB / "_csr.pdb"):   # zig가 Windows 대상에서 같이 만드는 파일
        if junk.exists():
            junk.unlink()
    print(f"  {key:<16} {target:<26} {out.stat().st_size:>7,} B")


def source_stamp() -> str:
    sha = hashlib.sha256(SRC.read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # 줄바꿈(CRLF)과 상관없이
    ver = subprocess.run([*zig(), "version"], capture_output=True, text=True, check=True).stdout.strip()
    return f"sha256 {sha}\nzig {ver}\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="6개 플랫폼 모두 (배포용)")
    a = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    keys = list(TARGETS) if a.all else [platform_key()]
    unknown = [k for k in keys if k not in TARGETS]
    if unknown:
        raise SystemExit(f"이 플랫폼용 대상이 없음: {unknown} (있는 것: {list(TARGETS)})")
    print(f"{SRC.name} → {LIB}")
    for k in keys:
        build(k)
    if a.all:
        (LIB / "SOURCE").write_text(source_stamp(), encoding="utf-8")
        print(f"  SOURCE: {(LIB / 'SOURCE').read_text(encoding='utf-8').strip()}")
