"""명령줄:  python -m flydnet [status | download [flywire|door ...] | verify | doctor]

  python -m flydnet                    # 데이터 상태 (어디서 무엇을 찾았는지)
  python -m flydnet download           # FlyWire v783 + DoOR 데이터 받기 (없는 파일만, 약 130 MB)
  python -m flydnet download flywire   # 한 묶음만
  python -m flydnet verify             # 받은 파일이 기대한 버전인지 (크기 + SHA-256)
  python -m flydnet doctor             # 설치 진단: GPU·CUDA·CuPy·torch, 어떤 설치 옵션을 쓸지
"""
import re
import sys

from . import __version__
from ._console import say
from .data import SOURCES, data_status, download, verify


def _driver_cuda() -> str | None:
    """드라이버가 지원하는 CUDA 버전 (nvidia-smi). 없으면 None"""
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


GPU_EXTRAS = (12, 13)                       # pyproject의 gpu-cuda12, gpu-cuda13


def _extra_hint(driver: int) -> str:
    """드라이버 CUDA 주 버전에 맞는 설치 옵션 (드라이버가 더 새것이면 지원하는 것 중 가장 새것 - 하위 호환)"""
    ok = [v for v in GPU_EXTRAS if v <= driver]
    if not ok:
        return f"CUDA {driver} 드라이버는 너무 오래됨 - 드라이버를 CUDA {GPU_EXTRAS[0]} 이상으로 업데이트"
    return f"pip install \"flydnet[gpu-cuda{ok[-1]}]\""


def _kernel_check() -> int:
    """전용 CUDA 커널을 실제로 컴파일·실행해 CuPy 기본 연산과 비교. 문제면 1"""
    import warnings
    import numpy as np
    import scipy.sparse as sps
    from .ganglion import backend as B, kernels as K
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = K._cuda() is not None
    for w in caught:
        say(f"  ! {w.message}")
    if not ok:
        return 1
    rng = np.random.default_rng(0)
    M = sps.random(50, 40, density=0.2, format="csr", dtype=np.float32, random_state=0)
    x = rng.standard_normal((40, 8)).astype(np.float32)
    Mg = B.sparse("gpu").csr_matrix(M)
    Mg.indices = Mg.indices.astype(np.int32)
    got = B.numpy(K.spmm(Mg, B.to(x, "gpu")))
    if not np.allclose(got, M @ x, atol=1e-4):
        say("  ! 전용 GPU 커널 결과가 다름 - FLYDNET_DEVICE=cpu로 쓰고 이슈로 알려 주세요")
        return 1
    say("전용 GPU 커널: 컴파일·실행·결과 확인됨")
    return 0


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
    driver = int(cuda.split(".")[0]) if cuda else None
    if driver and cupys:
        # 드라이버는 하위 호환: CUDA 13 드라이버에서 cupy-cuda12x도 돈다. 반대(CuPy가 드라이버보다 새것)는 안 됨
        for c in cupys:
            m = re.search(r"cuda(\d+)x", c.lower())
            if m and int(m.group(1)) > driver:
                problems += 1
                say(f"  ! {c}는 CUDA {m.group(1)}용인데 드라이버는 CUDA {driver} - 드라이버를 업데이트하거나 "
                    f"{_extra_hint(driver)}")
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
    import os
    forced = os.environ.get("FLYDNET_DEVICE", "").strip().lower()
    say(f"flydnet이 쓸 장치: {dev}" + (f" (환경변수 FLYDNET_DEVICE={forced}로 정함)" if forced in ("cpu", "gpu") else ""))
    for w in caught:
        say(f"  ! {w.message}")
    if dev == "cpu" and cuda and forced != "cpu":
        if not cupys:
            say(f"  → GPU를 쓰려면: {_extra_hint(driver)}")
        else:
            problems += 1
            say("  ! CuPy가 있는데 GPU 계산이 안 됨 (위 경고 참고)")
    if dev == "gpu":
        problems += _kernel_check()
    say("문제 없음" if not problems else f"문제 {problems}개")
    return 1 if problems else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "status"
    if cmd == "status":
        say(f"flydnet {__version__}")
        data_status()
    elif cmd == "download":
        try:
            download(argv or ("flywire", "door"))
        except (ValueError, IOError) as e:                           # 모르는 묶음·받기 실패: 트레이스백 대신 안내
            say(str(e))
            return 1
    elif cmd == "verify":
        unknown = [k for k in argv if k not in SOURCES]
        if unknown:
            say(f"모르는 데이터 묶음: {unknown} (있는 것: {list(SOURCES)})")
            return 1
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
