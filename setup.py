"""flydnet 빌드: 메타데이터는 pyproject.toml, 여기서는 CPU 희소 행렬 C 커널 확장 모듈만

  - 확장 모듈 flydnet.ganglion._csr (src/flydnet/ganglion/_csr.c). 파이썬 API를 쓰지 않으므로 안정 ABI(abi3)로 빌드 →
    플랫폼마다 휠 하나가 Python 3.10 이상 모든 버전에서 동작. 실행 중에는 import하지 않고 ctypes로 연다 (csr.py)
  - FLYDNET_PURE=1이면 확장 없이 → 순수 파이썬 휠 (py3-none-any): CPU 희소 연산은 scipy(있으면)나 numpy
  - 컴파일러가 없어 빌드가 실패해도 설치는 계속 (경고만) → 순수 파이썬과 같음
  - FMA로 합치지 않게 (-ffp-contract=off; MSVC는 기본이 합치지 않음) → scipy·numpy 경로와 같은 비트
"""
import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """컴파일러별 최적화 옵션, 실패하면 확장 없이"""

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            args = ["/O2", "/fp:precise"]
        else:
            args = ["-O3", "-std=c99", "-ffp-contract=off", "-fvisibility=hidden"]
        for ext in self.extensions:
            ext.extra_compile_args = args
        super().build_extensions()

    def run(self):
        try:
            super().run()
        except Exception as e:                                           # noqa: BLE001 - 컴파일러 없음 등
            self._skip(e)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:                                           # noqa: BLE001
            self._skip(e)

    def _skip(self, e):
        if os.environ.get("FLYDNET_REQUIRE_EXT") == "1":                 # 배포 휠(cibuildwheel)은 실패하면 멈춤
            raise e
        print(f"경고: CPU 희소 행렬 C 커널을 빌드하지 못함 ({type(e).__name__}: {e}) - flydnet은 scipy 또는 numpy로 동작 "
              "(느림). 나중에: python scripts/build_csr.py", file=sys.stderr)


ext_modules = []
cmdclass = {}
options = {}
if os.environ.get("FLYDNET_PURE") != "1":
    ext_modules = [Extension("flydnet.ganglion._csr", ["src/flydnet/ganglion/_csr.c"],
                             define_macros=[("FLYDNET_PYEXT", "1"), ("Py_LIMITED_API", "0x030A0000")],
                             py_limited_api=True)]
    cmdclass = {"build_ext": OptionalBuildExt}
    options = {"bdist_wheel": {"py_limited_api": "cp310"}}

setup(ext_modules=ext_modules, cmdclass=cmdclass, options=options)
