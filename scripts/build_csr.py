"""CPU 희소 행렬 C 커널(src/flydnet/ganglion/_csr.c)을 공유 라이브러리로 빌드 (소스에서 개발할 때)

  python scripts/build_csr.py                 # CC 환경변수 → zig (pip install ziglang) → cc·gcc·clang 순으로 찾음
  python scripts/build_csr.py --cc "zig cc"   # 컴파일러 직접

결과: src/flydnet/ganglion/_csr.dll (Windows) / _csr.so (Linux) / _csr.dylib (macOS) - flydnet이 자동으로 찾음.
배포 휠은 setup.py의 확장 모듈로 빌드되므로 이 스크립트가 필요 없음.
FMA로 합치지 않게 -ffp-contract=off (scipy·numpy 경로와 같은 비트)."""
import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "flydnet" / "ganglion" / "_csr.c"
EXT = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
OUT = SRC.with_name("_csr" + EXT)
FLAGS = ["-O3", "-std=c99", "-ffp-contract=off", "-shared"]


def find_cc():
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"])
    if importlib.util.find_spec("ziglang") is not None:
        return [sys.executable, "-m", "ziglang", "cc"]
    for name in ("cc", "gcc", "clang"):
        if shutil.which(name):
            return [name]
    raise SystemExit("C 컴파일러가 없음 - pip install ziglang 또는 gcc·clang 설치, 또는 --cc로 지정")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default=None, help='컴파일러 명령 (예: "zig cc", gcc)')
    a = ap.parse_args()
    cc = shlex.split(a.cc) if a.cc else find_cc()
    cmd = [*cc, *FLAGS, *([] if sys.platform == "win32" else ["-fPIC"]), "-o", str(OUT), str(SRC)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    for junk in (OUT.with_suffix(".lib"), OUT.with_suffix(".pdb")):        # zig가 Windows에서 같이 만드는 파일
        if junk.exists():
            junk.unlink()
    print(f"→ {OUT} ({OUT.stat().st_size:,} B)")
