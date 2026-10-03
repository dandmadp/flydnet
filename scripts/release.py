"""배포 도우미 — 수정·배포를 여러 번 반복해도 실수가 없도록

  python scripts/release.py bump patch       # 0.1.0 → 0.1.1 (minor: 0.2.0, major: 1.0.0) + CHANGELOG 항목 틀
  python scripts/release.py check            # 배포 전 검사 + 빌드 (업로드는 안 함)
  python scripts/release.py check --upload   # 검사 통과하면 PyPI에 업로드
  python scripts/release.py check --test     # 연습용 TestPyPI에 업로드

check가 하는 일 (하나라도 실패하면 멈춤):
  1. git에 커밋 안 된 변경이 없는지          → 커밋 안 된 코드를 올리는 것 방지
  2. CHANGELOG.md에 이 버전 항목이 있는지     → 버전마다 무엇이 바뀌었는지 기록
  3. PyPI에 이 버전이 아직 없고 최신보다 큰지 → 버전 올리기를 잊은 업로드 방지 (같은 번호는 영구히 재사용 불가)
  4. 테스트 전체 통과
  5. dist/ 를 비우고 새로 빌드 + twine check  → 옛 버전 파일이 같이 올라가는 것 방지
  6. 빌드한 wheel을 임시 폴더에 따로 설치해 불러오기·버전·명령줄 확인
  7. git 태그 v<버전>                        → 어떤 커밋을 배포했는지 기록
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "flydnet" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"
NAME = "flydnet"
PY = sys.executable


def version() -> str:
    m = re.search(r'^__version__ = "([^"]+)"', INIT.read_text(encoding="utf-8"), re.M)
    return m.group(1)


def vtuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw)


def fail(msg):
    print(f"\n✗ {msg}")
    sys.exit(1)


def ok(msg):
    print(f"✓ {msg}", flush=True)


def bump(part: str):
    old = version()
    major, minor, patch = vtuple(old)
    new = {"major": f"{major + 1}.0.0", "minor": f"{major}.{minor + 1}.0", "patch": f"{major}.{minor}.{patch + 1}"}[part]
    INIT.write_text(INIT.read_text(encoding="utf-8").replace(f'__version__ = "{old}"', f'__version__ = "{new}"'),
                    encoding="utf-8")
    log = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else "# 변경 기록\n"
    if f"## [{new}]" not in log:
        head, sep, rest = log.partition("\n## ")
        entry = f"\n## [{new}] - {datetime.date.today()}\n\n- (바뀐 점을 적기)\n"
        log = head.rstrip("\n") + "\n" + entry + (("\n## " + rest) if sep else "")
        CHANGELOG.write_text(log, encoding="utf-8")
    print(f"버전 {old} → {new}\n  다음: CHANGELOG.md의 [{new}] 항목을 채우고 커밋한 뒤  python scripts/release.py check")


def pypi_versions(test: bool) -> list[str]:
    host = "test.pypi.org" if test else "pypi.org"
    try:
        with urllib.request.urlopen(f"https://{host}/pypi/{NAME}/json", timeout=20) as r:
            return list(json.load(r)["releases"])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []                                     # 아직 한 번도 안 올림
        raise


def check(upload: bool, test: bool):
    v = version()
    print(f"flydnet {v} 배포 검사\n")

    st = run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if st:
        fail(f"커밋 안 된 변경이 있음:\n{st}\n  → 커밋한 뒤 다시")
    ok("git 작업 트리 깨끗함")

    if not CHANGELOG.exists() or f"## [{v}]" not in CHANGELOG.read_text(encoding="utf-8"):
        fail(f"CHANGELOG.md에 [{v}] 항목이 없음 → python scripts/release.py bump ... 로 만들거나 직접 추가")
    if "(바뀐 점을 적기)" in CHANGELOG.read_text(encoding="utf-8").split(f"## [{v}]")[1].split("\n## ")[0]:
        fail(f"CHANGELOG.md의 [{v}] 항목이 아직 틀 그대로임 → 바뀐 점을 적기")
    ok(f"CHANGELOG에 [{v}] 있음")

    released = pypi_versions(test)
    if v in released:
        fail(f"{v}는 이미 {'TestPyPI' if test else 'PyPI'}에 있음 (같은 번호는 지워도 다시 못 씀) "
             f"→ python scripts/release.py bump patch")
    if released and vtuple(v) <= max(vtuple(r) for r in released):
        fail(f"{v}가 이미 올린 최신 버전({max(released, key=vtuple)})보다 크지 않음")
    ok(f"{'TestPyPI' if test else 'PyPI'}에 아직 없는 새 버전 (올린 버전: {', '.join(sorted(released, key=vtuple)) or '없음'})")

    if run([PY, "-m", "pytest", "-q"]).returncode:
        fail("테스트 실패")
    ok("테스트 통과")

    for d in ("dist", "build", f"src/{NAME}.egg-info"):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    if run([PY, "-m", "build"], capture_output=True).returncode:
        fail("빌드 실패 → python -m build 로 직접 확인")
    files = sorted((ROOT / "dist").iterdir())
    if any(v not in f.name for f in files):
        fail(f"dist/에 다른 버전 파일이 섞임: {[f.name for f in files]}")
    if run([PY, "-m", "twine", "check", *map(str, files)]).returncode:
        fail("twine check 실패 (README 형식 등)")
    ok(f"빌드: {', '.join(f'{f.name} ({f.stat().st_size // 1024} KB)' for f in files)}")

    whl = next(f for f in files if f.suffix == ".whl")
    with tempfile.TemporaryDirectory() as tmp:
        if run([PY, "-m", "pip", "install", "-q", "--no-deps", "--target", tmp, str(whl)]).returncode:
            fail("wheel 설치 실패")
        env = dict(os.environ, PYTHONPATH=tmp)
        code = (f"import flydnet, pathlib; assert flydnet.__version__ == '{v}', flydnet.__version__; "
                f"assert pathlib.Path(flydnet.__file__).resolve().is_relative_to(pathlib.Path(r'{tmp}').resolve()), "
                f"flydnet.__file__")
        if subprocess.run([PY, "-c", code], cwd=tmp, env=env).returncode:
            fail("설치한 wheel을 불러오지 못함 (또는 버전이 다름)")
        if subprocess.run([PY, "-m", NAME, "status"], cwd=tmp, env=env, capture_output=True).returncode:
            fail("python -m flydnet 실행 실패")
    ok("빌드한 wheel을 따로 설치해 불러오기·버전·명령줄 확인")

    tag = f"v{v}"
    if run(["git", "tag", "--list", tag], capture_output=True, text=True).stdout.strip():
        ok(f"git 태그 {tag} 이미 있음")
    else:
        run(["git", "tag", "-a", tag, "-m", f"flydnet {v}"])
        ok(f"git 태그 {tag}")

    cmd = [PY, "-m", "twine", "upload", *(["--repository", "testpypi"] if test else []), *map(str, files)]
    if not upload:
        print(f"\n준비 완료. 올리려면:\n  python scripts/release.py check --upload{' --test' if test else ''}"
              f"\n  (또는 직접: {' '.join(cmd)})")
        return
    print("\n업로드 (토큰을 물으면 pypi-로 시작하는 값 전체를 붙여 넣기)")
    if run(cmd).returncode:
        fail("업로드 실패 — 위 메시지 확인. 태그는 남아 있으니 고친 뒤 다시 check --upload 하면 됨")
    url = f"https://{'test.' if test else ''}pypi.org/project/{NAME}/{v}/"
    ok(f"업로드 완료: {url}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")         # 윈도우 콘솔에서 한글·기호 깨짐 방지
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bump", help="버전 올리기")
    b.add_argument("part", choices=["patch", "minor", "major"])
    c = sub.add_parser("check", help="배포 전 검사 + 빌드 (+ 업로드)")
    c.add_argument("--upload", action="store_true", help="검사 통과하면 업로드")
    c.add_argument("--test", action="store_true", help="TestPyPI 사용")
    a = ap.parse_args()
    bump(a.part) if a.cmd == "bump" else check(a.upload, a.test)
