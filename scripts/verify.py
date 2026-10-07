"""flydnet 전체 검증: 빠른 것부터, 실패하면 멈춤. 같은 검사를 더 짧은 시간에 (병렬·GPU 메모리 배분)

  python scripts/verify.py                  # 전체 (배포 전에는 늘 이것)
  python scripts/verify.py --jobs 3         # 예제를 최대 3개까지 동시에 (GPU 메모리가 허락할 때만)
  python scripts/verify.py --changed        # 실제로 실행되는 flydnet 함수가 바뀐 예제만 다시 (중간 점검용)
  python scripts/verify.py --only refs,tests
  python scripts/verify.py --keep-going     # 실패해도 다음 단계로

단계 (순서대로)
  refs      참조 검사 10종 (tests/ref_*.py): 연산·역전파 대 torch, 최적화기 대 torch.optim, LIF 커널 대 독립 구현 등. 약 1분
  snapshot  엔진 비트 단위 비교 (tests/snapshot.py, 기준 .verify/snapshot_base.npz)
  tests     전체 테스트 (경고도 실패로) - 파일을 둘로 나눠 동시에 + 옵션 조합 60개·퍼저 3종을 같이
  examples  예제 22개 원래 크기 그대로 + 원본 Brian2 검증 - GPU 메모리를 보며 병렬로, 오래 걸리는 것부터

--changed (영향 분석): 예제마다 지난 실행에서 실제로 호출된 flydnet 함수(와 그 함수가 든 모듈·클래스의 최상위 코드)를
기록해 두고, 그 소스가 한 글자도 바뀌지 않았으면 지난 결과를 그대로 씀 (같은 코드·같은 seed → 같은 결과).
호출된 함수가 하나라도 바뀌었거나, 예제 파일·numpy·cupy·torch 버전이 바뀌었으면 다시 실행. 기록이 없는 예제도 실행.
결과: .verify/last.json, 예제 출력 .verify/examples/<이름>.log, 지난 실행과 출력 비교
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
STATE = ROOT / ".verify"
EXLOG = STATE / "examples"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
GB = 1 << 30
# 예제 사이 순서: 앞 예제가 만든 캐시를 읽는 예제는 그 뒤에 (앞 예제를 다시 돌리면 뒤도 다시)
DEPENDS = {"mnist_dopamine": ["mnist_reservoir"], "continual_mnist": ["mnist_reservoir"]}


def say(*a):
    print(*a, flush=True)


# ─────────────── 병렬 실행 ───────────────
class Job:
    def __init__(self, name, cmd, log, need=0, record=None, after=()):
        self.name, self.cmd, self.log, self.need, self.record = name, cmd, log, need, record
        self.after = set(after)                                          # 이 작업들이 끝난 뒤에 시작
        self.proc = self.t0 = self.code = self.seconds = None

    def start(self):
        self.t0 = time.time()
        self.fh = open(self.log, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, env=ENV, stdout=self.fh, stderr=subprocess.STDOUT)

    def poll(self):
        if self.proc.poll() is None:
            return False
        self.code, self.seconds = self.proc.returncode, time.time() - self.t0
        self.fh.close()
        return True

    def tail(self, n=1):
        try:
            lines = [l for l in self.log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if l.strip() and "Warning" not in l and "return self" not in l]
        except FileNotFoundError:
            return ""
        return " | ".join(lines[-n:])


def gpu_total_free():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split(",")
        total, used = float(out[0]) * (1 << 20), float(out[1]) * (1 << 20)
        return total, total - used
    except Exception:                                                     # noqa: BLE001 - GPU 없음
        return 0.0, 0.0


def run_pool(jobs, workers, budget=None, on_done=None):
    """jobs를 workers개까지 동시에. budget(바이트)이면 돌고 있는 작업의 need 합이 budget을 넘지 않게"""
    waiting, running, done = list(jobs), [], []
    while waiting or running:
        for j in [j for j in running if j.poll()]:
            running.remove(j)
            done.append(j)
            if on_done:
                on_done(j)
        used = sum(j.need for j in running)
        finished = {d.name for d in done}
        for j in list(waiting):
            if len(running) >= workers:
                break
            if not j.after <= finished:
                continue
            if budget is not None and running and used + j.need > budget:
                continue                                                  # 메모리가 모자라면 작은 작업을 먼저 끼움
            waiting.remove(j)
            j.start()
            running.append(j)
            used += j.need
        time.sleep(0.5)
    return done


# ─────────────── 영향 분석: 함수별 소스 지문 ───────────────
def _strip_bodies(tree):
    """함수 본문을 비운 사본 (모듈·클래스의 최상위 코드 지문용)"""
    import copy
    t = copy.deepcopy(tree)
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.body = [ast.Pass()]
    return t


def fingerprints(path: pathlib.Path) -> dict:
    """{qualname: 지문} - co_qualname과 같은 이름 규칙 (중첩 함수는 f.<locals>.g, 람다·컴프리헨션은 같은 이름끼리 묶음)"""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: dict[str, list] = {}

    def add(q, node):
        out.setdefault(q, []).append(ast.dump(node, include_attributes=False))

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
                add(q, ch)
                walk(ch, q + ".<locals>.")
            elif isinstance(ch, ast.ClassDef):
                q = prefix + ch.name
                add(q, _strip_bodies(ch))
                walk(ch, q + ".")
            elif isinstance(ch, ast.Lambda):
                add(prefix + "<lambda>", ch)
                walk(ch, prefix + "<lambda>.<locals>.")
            elif isinstance(ch, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
                name = {ast.GeneratorExp: "<genexpr>", ast.ListComp: "<listcomp>", ast.SetComp: "<setcomp>",
                        ast.DictComp: "<dictcomp>"}[type(ch)]
                add(prefix + name, ch)
                walk(ch, prefix + name + ".")                             # 안쪽 컴프리헨션: f.<locals>.<dictcomp>.<listcomp>
            else:
                walk(ch, prefix)
    walk(tree, "")
    res = {k: hashlib.sha1("\n".join(v).encode()).hexdigest() for k, v in out.items()}
    res["<module>"] = _module_fp(_strip_bodies(tree))
    return res


def _module_fp(tree) -> str:
    """모듈 최상위 지문 "m2:<나머지>:<import들>:<이름 겹침>" - import 줄을 하나 더한 것(새 기능 내보내기)만으로는
    그 모듈을 쓰는 모든 예제가 다시 돌지 않게, 최상위 import 문은 따로 집합으로 (fp_same이 '예전 import ⊆ 지금'이면 같다고 봄).
    새 import가 이미 있던 이름을 덮으면 (같은 이름이 최상위에서 두 번 정해짐) 이름 겹침 = 1 → 엄격하게 비교"""
    imports, rest, names = [], [], []
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            imports.append(hashlib.sha1(ast.dump(st, include_attributes=False).encode()).hexdigest()[:16])
            names += [(al.asname or al.name).split(".")[0] for al in st.names]
        else:
            rest.append(ast.dump(st, include_attributes=False))
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(st.name)
            elif isinstance(st, (ast.Assign, ast.AnnAssign)):
                for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                    names += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
    dup = int(len(names) != len(set(names)))
    return f"m2:{hashlib.sha1(chr(10).join(rest).encode()).hexdigest()}:{','.join(sorted(imports))}:{dup}"


def fp_same(rel, qual, old) -> bool:
    """기록된 지문 old가 지금 소스와 같은지 (모듈 최상위는 import를 더하기만 했으면 같음)"""
    now = fp_of(rel, qual)
    if now == old:
        return True
    if qual != "<module>" or not (now and old and now.startswith("m2:") and old.startswith("m2:")):
        return False
    _, c_old, i_old, _ = old.split(":")
    _, c_now, i_now, dup = now.split(":")
    return c_old == c_now and dup == "0" and set(filter(None, i_old.split(","))) <= set(filter(None, i_now.split(",")))


_FP: dict = {}


def fp_of(rel, qual):
    if rel not in _FP:
        p = ROOT / rel
        _FP[rel] = fingerprints(p) if p.exists() else {}
    d = _FP[rel]
    # co_qualname의 컴프리헨션은 바깥 함수 이름에 붙음 (f.<locals>.<listcomp>) - 같은 규칙
    return d.get(qual)


def versions():
    v = {}
    for m in ("numpy", "scipy", "cupy", "torch", "pandas"):
        try:
            v[m] = __import__(m).__version__
        except Exception:                                                 # noqa: BLE001
            v[m] = None
    return v


def file_hash(p):
    return hashlib.sha1(pathlib.Path(p).read_bytes()).hexdigest()


# ─────────────── 단계 ───────────────
def stage_refs(a):
    refs = sorted((ROOT / "tests").glob("ref_*.py"))
    jobs = [Job(p.stem, [PY, "-W", "ignore", str(p)], STATE / f"{p.stem}.log") for p in refs]
    ok = True
    for j in run_pool(jobs, 3):
        good = j.code == 0
        ok &= good
        say(f"  {'✓' if good else '✗'} {j.name:<16} {j.seconds:5.0f}s  {j.tail()}")
    return ok


def stage_snapshot(a):
    j = Job("snapshot", [PY, "-W", "ignore", "tests/snapshot.py", "check"], STATE / "snapshot.log")
    run_pool([j], 1)
    good = j.code == 0 and "다름 0개" in j.tail()
    say(f"  {'✓' if good else '✗'} 엔진 비트 비교 {j.seconds:.0f}s  {j.tail()}")
    return good


# lab/의 긴 연구 실험: --lab에서도 빼고 직접 돌림 (python lab/<이름>.py). lab.Growth는 growth_odor가 확인
LAB_MANUAL = {"growth_lesion", "growth_lesion_mnist"}


def stage_tests(a):
    files = sorted((ROOT / "tests").glob("test_*.py"), key=lambda p: -p.stat().st_size)
    halves = [[], []]
    size = [0, 0]
    for f in files:                                                       # 크기로 고르게 둘로
        k = size.index(min(size))
        halves[k].append(str(f.relative_to(ROOT)))
        size[k] += f.stat().st_size
    base = [PY, "-m", "pytest", "-q", "-W", "error::UserWarning", "-p", "no:cacheprovider"]
    jobs = [Job(f"pytest {i + 1}/2", base + h, STATE / f"pytest{i + 1}.log") for i, h in enumerate(halves)]
    if a.lab:                                                             # lab 기능(growth·대조 학습) 시험: 기본 pytest는 건너뜀
        jobs += [Job("pytest lab", base + ["-m", "lab", "tests"], STATE / "pytest_lab.log")]
    jobs += [Job("combos", [PY, "tests/combo_check.py", "0", "60"], STATE / "combos.log")]
    jobs += [Job(n, [PY, f"tests/{n}.py"], STATE / f"{n}.log") for n in ("fuzz_ops", "fuzz_layers", "fuzz_sparse")]
    ok = True
    for j in run_pool(jobs, 4):
        tail = j.tail()
        good = j.code == 0 and not re.search(r"문제 [1-9]|다름 [1-9]|failed", tail)
        ok &= good
        say(f"  {'✓' if good else '✗'} {j.name:<12} {j.seconds:5.0f}s  {tail}")
    return ok


def _numbers(text):
    """출력 비교용: 경고·진행 줄을 빼고 숫자가 있는 줄"""
    keep = []
    for l in text.splitlines():
        if not l.strip() or "Warning" in l or "return self" in l or re.search(r"\d+(\.\d+)?s/step|\[\d+s\]|\(\d+s\)", l):
            continue
        if re.search(r"\d", l):
            keep.append(re.sub(r"\s+", " ", l.strip()))
    return keep


def stage_examples(a):
    EXLOG.mkdir(parents=True, exist_ok=True)
    prev = {}
    if (STATE / "examples.json").exists():
        prev = json.loads((STATE / "examples.json").read_text(encoding="utf-8"))
    ver = versions()
    exs = sorted(p for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("_"))   # _로 시작 = 공용 도우미
    if a.lab:                                                             # lab/: 실험적 기능의 연구 예제 (기본은 빼서 시간 줄임)
        exs += sorted(p for p in (ROOT / "lab").glob("*.py") if not p.name.startswith("_") and p.stem not in LAB_MANUAL)
    tasks = [(p.stem, [str(p)]) for p in exs] + [("shiu_brian2", ["validation/shiu2024/compare.py", "--run"])]
    total, free = gpu_total_free()
    budget = max(free - 0.5 * GB, 0) if total else None
    jobs, reused = [], {}
    for name, argv in tasks:
        rec = prev.get(name)
        if a.changed and rec and rec.get("code") == 0 and rec.get("versions") == ver \
                and rec.get("file") == file_hash(ROOT / argv[0]) and rec.get("called"):
            stale = [f"{f}:{q}" for f, q, h in rec["called"] if not fp_same(f, q, h)]
            if not stale:
                reused[name] = rec
                continue
        need = int((rec or {}).get("gpu_peak", 0) * 1.15 + 0.5 * GB) if rec and rec.get("gpu_peak") else int(5 * GB)
        out = STATE / f"ex_{name}.json"
        cmd = [PY, "-W", "always", "scripts/_example_runner.py", str(out), *argv]
        j = Job(name, cmd, EXLOG / f"{name}.log", need=need if budget else 0, record=out,
                after=[d for d in DEPENDS.get(name, ())])
        same_file = rec and rec.get("file") == file_hash(ROOT / argv[0])
        j.prev_seconds = rec.get("seconds", 1e9) if same_file else 1e9    # 예제가 바뀌었으면 지난 시간은 모름 → 앞으로
        jobs.append(j)
    names = {j.name for j in jobs}
    for j in jobs:
        j.after &= names                                                  # 재사용한 앞 예제는 기다리지 않음
    jobs.sort(key=lambda j: -j.prev_seconds)                              # 오래 걸리는 것부터 (모르는 것도 앞)
    if reused:
        say(f"  재사용 {len(reused)}개 (실행되는 함수가 바뀌지 않음): {', '.join(sorted(reused))}")
    say(f"  실행 {len(jobs)}개, 동시 {a.jobs}개까지" + (f", GPU 여유 {budget / GB:.1f} GB 안에서" if budget else ""))
    new = dict(prev)
    ok = True

    def done(j):
        nonlocal ok
        rec = {}
        try:
            rec = json.loads(j.record.read_text(encoding="utf-8"))
        except Exception:                                                 # noqa: BLE001
            pass
        text = j.log.read_text(encoding="utf-8", errors="replace")
        oom = re.search(r"out of memory|OutOfMemory|GPUMemoryError|메모리 부족", text)
        warns = len(re.findall(r"\w+Warning:", text))
        old = prev.get(j.name, {})
        rec_out = dict(code=j.code, seconds=j.seconds, gpu_peak=rec.get("gpu_peak", 0), warnings=warns,
                       versions=ver, file=file_hash(ROOT / j.cmd[5]), numbers=_numbers(text),
                       called=[[f, q, fp_of(f, q)] for f, q in rec.get("called", [])])
        j.oom = bool(oom) and j.code != 0
        if j.oom:
            return
        new[j.name] = rec_out
        good = j.code == 0
        ok &= good
        diff = ""
        if old.get("numbers") and good:
            a_, b_ = old["numbers"], rec_out["numbers"]
            changed = [x for x in b_ if x not in a_]
            diff = f", 지난 실행과 다른 줄 {len(changed)}개" if changed else ", 출력 숫자 지난 실행과 같음"
        say(f"  {'✓' if good else '✗'} {j.name:<22} {j.seconds:5.0f}s  GPU {rec_out['gpu_peak'] / GB:4.1f} GB  "
            f"경고 {warns}{diff}")

    finished = run_pool(jobs, a.jobs, budget, on_done=done)
    retry = [j for j in finished if getattr(j, "oom", False)]
    if retry:                                                             # 동시에 돌다 메모리가 모자랐던 것: 혼자 다시
        say(f"  GPU 메모리 부족으로 실패한 {len(retry)}개를 혼자 다시: {', '.join(j.name for j in retry)}")
        again = [Job(j.name, j.cmd, j.log, 0, j.record) for j in retry]   # 혼자 (순서는 이미 지켜짐)
        run_pool(again, 1, None, on_done=done)
        for j in again:
            if getattr(j, "oom", False):
                ok = False
                say(f"  ✗ {j.name}: 혼자 돌려도 GPU 메모리 부족")
    for name, rec in reused.items():
        say(f"  = {name:<22} (재사용, 지난 {rec['seconds']:.0f}s)")
    (STATE / "examples.json").write_text(json.dumps(new, ensure_ascii=False, indent=0), encoding="utf-8")
    # 원본 Brian2 검증: 결과 파일이 바뀌지 않았는지
    if "shiu_brian2" not in reused:
        d = subprocess.run(["git", "diff", "--stat", "--", "validation/shiu2024/flydnet_results.json"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        say(f"  {'✓' if not d else '✗'} 원본 Brian2 결과 파일 {'변화 없음' if not d else '바뀜: ' + d}")
        ok &= not d
    return ok


STAGES = {"refs": ("참조 검사", stage_refs), "snapshot": ("엔진 비트 비교", stage_snapshot),
          "tests": ("테스트·조합·퍼저", stage_tests), "examples": ("예제·원본 Brian2", stage_examples)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3, help="예제 동시 실행 수 (GPU 메모리가 허락하는 만큼만)")
    ap.add_argument("--changed", action="store_true", help="실행되는 함수가 바뀐 예제만 (중간 점검)")
    ap.add_argument("--only", default=None, help="단계 이름 쉼표로: " + ",".join(STAGES))
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--lab", action="store_true", help="lab/의 실험적 기능 예제도 (growth·보정·학습 규칙을 고쳤을 때)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    STATE.mkdir(exist_ok=True)
    names = a.only.split(",") if a.only else list(STAGES)
    unknown = set(names) - set(STAGES)
    if unknown:
        sys.exit(f"모르는 단계: {sorted(unknown)} (있는 것: {list(STAGES)})")
    t0 = time.time()
    summary = {}
    for n in names:
        title, fn = STAGES[n]
        say(f"[{n}] {title}")
        s = time.time()
        good = fn(a)
        summary[n] = dict(ok=good, seconds=time.time() - s)
        say(f"  → {'통과' if good else '실패'} ({time.time() - s:.0f}s)\n")
        if not good and not a.keep_going:
            say("실패 - 멈춤 (--keep-going이면 계속)")
            break
    (STATE / "last.json").write_text(json.dumps(dict(summary=summary, seconds=time.time() - t0,
                                                     changed=a.changed, when=time.strftime("%Y-%m-%d %H:%M:%S")),
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    good = all(v["ok"] for v in summary.values()) and len(summary) == len(names)
    say(f"전체 {'통과' if good else '실패'}: {time.time() - t0:.0f}초" + (" (--changed: 바뀐 예제만)" if a.changed else ""))
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
