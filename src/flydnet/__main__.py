"""명령줄:  python -m flydnet [status | download [flywire|door ...]]

  python -m flydnet                    # 데이터 상태 (어디서 무엇을 찾았는지)
  python -m flydnet download           # FlyWire v783 + DoOR 데이터 받기 (없는 파일만, 약 130 MB)
  python -m flydnet download flywire   # 한 묶음만
"""
import sys

from . import __version__
from .data import data_status, download


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "status"
    if cmd == "status":
        print(f"flydnet {__version__}")
        data_status()
    elif cmd == "download":
        download(argv or ("flywire", "door"))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
