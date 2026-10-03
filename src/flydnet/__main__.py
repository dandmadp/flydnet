"""명령줄:  python -m flydnet [status | download [flywire|door ...]]

  python -m flydnet                    # 데이터 상태 (어디서 무엇을 찾았는지)
  python -m flydnet download           # FlyWire v783 + DoOR 데이터 받기 (없는 파일만, 약 130 MB)
  python -m flydnet download flywire   # 한 묶음만
  python -m flydnet verify             # 받은 파일이 기대한 버전인지 (크기 + SHA-256)
"""
import sys

from ._console import say

from . import __version__
from .data import SOURCES, data_status, download, verify


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "status"
    if cmd == "status":
        say(f"flydnet {__version__}")
        data_status()
    elif cmd == "download":
        download(argv or ("flywire", "door"))
    elif cmd == "verify":
        bad = 0
        for kind in argv or SOURCES:
            for name, st in verify(kind).items():
                say(f"  {kind:<8} {name:<28} {st}")
                bad += st != "ok"
        return 1 if bad else 0
    else:
        say(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
