"""verify.py가 예제 하나를 돌릴 때 쓰는 실행기: 예제를 그대로 실행하면서
  - 실제로 실행된 flydnet 함수 (파일, 이름)를 기록 → 영향 분석 (--changed)
  - 이 프로세스의 GPU 메모리 최대치 (CuPy 풀 + torch)를 0.2초마다 재서 기록 → 병렬 실행 때 메모리 배분
결과는 JSON 한 줄을 파일로 (예제 출력과 섞이지 않게)

  python scripts/_example_runner.py <기록 파일.json> <예제.py> [예제 인자...]
"""
import json
import os
import pathlib
import runpy
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src" / "flydnet")
out_path, script, *rest = sys.argv[1:]
called = set()
peak = [0]


def tracer(frame, event, arg):
    """새 프레임마다 한 번 (줄 단위 추적은 안 함 → 느려지지 않음). flydnet 소스의 함수만"""
    co = frame.f_code
    f = co.co_filename
    if f.startswith(SRC):
        called.add((os.path.relpath(f, ROOT).replace("\\", "/"), getattr(co, "co_qualname", co.co_name)))
    return None


def watch():
    while True:
        used = 0
        cp = sys.modules.get("cupy")
        if cp is not None:
            try:
                used += int(cp.get_default_memory_pool().total_bytes())
            except Exception:                                            # noqa: BLE001
                pass
        th = sys.modules.get("torch")
        if th is not None:
            try:
                if th.cuda.is_available() and th.cuda.is_initialized():
                    used += int(th.cuda.memory_reserved())
            except Exception:                                            # noqa: BLE001
                pass
        peak[0] = max(peak[0], used)
        time.sleep(0.2)


threading.Thread(target=watch, daemon=True).start()
sys.argv = [script, *rest]
sys.path.insert(0, str(pathlib.Path(script).resolve().parent))
code = 0
t0 = time.time()
sys.settrace(tracer)
threading.settrace(tracer)
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException:                                                     # noqa: BLE001
    import traceback
    traceback.print_exc()
    code = 1
finally:
    sys.settrace(None)
    time.sleep(0.3)
    pathlib.Path(out_path).write_text(json.dumps(dict(code=code, seconds=time.time() - t0, gpu_peak=peak[0],
                                                      called=sorted(called))), encoding="utf-8")
sys.exit(code)
