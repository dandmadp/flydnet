"""명령줄:  python -m flydnet [status | download [flywire|door ...] | verify | doctor]

  python -m flydnet                    # 데이터 상태 (어디서 무엇을 찾았는지)
  python -m flydnet download           # FlyWire v783 + DoOR 데이터 받기 (없는 파일만, 약 130 MB)
  python -m flydnet download flywire   # 한 묶음만
  python -m flydnet verify             # 받은 파일이 기대한 버전인지 (크기 + SHA-256)
  python -m flydnet doctor             # 설치 진단: GPU·CUDA·CuPy·torch, 어떤 설치 옵션을 쓸지
"""
import sys

from . import __version__
from ._console import say
from .data import SOURCES, data_status, download, verify


def _driver_cuda() -> str | None:
    """드라이버가 지원하는 CUDA 버전 (nvidia-smi). 없으면 None"""
    import re
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", out)
    return m.group(1) if m else None


def doctor() -> int:
    """설치 환경 진단. 문제가 있으면 1"""
    from importlib import metadata
    problems = 0
    say(f"flydnet {__version__}, Python {sys.version.split()[0]}")
    cuda = _driver_cuda()
    say(f"GPU 드라이버의 CUDA: {cuda or '없음 (NVIDIA GPU 또는 드라이버 없음)'}")
    cupys = sorted(d.metadata["Name"] for d in metadata.distributions()
                   if (d.metadata["Name"] or "").lower().startswith("cupy"))
    say(f"설치된 CuPy: {', '.join(cupys) or '없음'}")
    if len(cupys) > 1:
        problems += 1
        say(f"  ! CuPy가 {len(cupys)}개 - 서로 충돌함. 드라이버 CUDA에 맞는 하나만 남길 것: "
            f"pip uninstall -y {' '.join(c for c in cupys)} 후 하나만 다시 설치 (Colab이면 런타임 삭제 후 새로)")
    if cuda and cupys:
        major = cuda.split(".")[0]
        wrong = [c for c in cupys if "cuda" in c.lower() and f"cuda{major}x" not in c.lower()]
        if wrong:
            problems += 1
            say(f"  ! 드라이버는 CUDA {major}인데 {', '.join(wrong)}가 설치됨 - cupy-cuda{major}x를 쓸 것")
    try:
        import torch
        say(f"torch: {torch.__version__} (GPU {'사용 가능' if torch.cuda.is_available() else '없음'}) - flydnet.torch 사용 가능")
    except ImportError:
        say("torch: 없음 (필요 없음. torch 연동만 pip install \"flydnet[torch]\")")
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from .ganglion import backend as B
        dev = B.default_device()
    say(f"flydnet이 쓸 장치: {dev}")
    for w in caught:
        say(f"  ! {w.message}")
    if dev == "cpu" and cuda:
        if not cupys:
            say(f"  → GPU를 쓰려면: pip install \"flydnet[gpu-cuda{cuda.split('.')[0]}]\"")
        else:
            problems += 1
            say("  ! CuPy가 있는데 GPU 계산이 안 됨 (위 경고 참고)")
    say("문제 없음" if not problems else f"문제 {problems}개")
    return 1 if problems else 0


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
    elif cmd == "doctor":
        return doctor()
    else:
        say(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
