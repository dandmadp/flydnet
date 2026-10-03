"""안전한 출력: 출력이 파일·파이프로 갈 때 한국어 윈도우는 cp949라 표현 못 하는 문자(예: 이름의 ü)에서 죽음 → 대체 문자로"""
import sys


def say(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        stream = kwargs.get("file") or sys.stdout
        enc = getattr(stream, "encoding", None) or "utf-8"
        safe = [str(a).encode(enc, errors="replace").decode(enc) for a in args]
        print(*safe, **kwargs)
