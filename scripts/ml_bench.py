"""ML 성능 기준표와 안전장치: 기존 수치에서 나빠지면 잡아내고 (원하면) 작업 전 상태로 되돌림

  python scripts/ml_bench.py baseline            # 지금 수치를 기준으로 저장 (benchmarks/ml_baseline.json, git이 추적)
  python scripts/ml_bench.py checkpoint          # 작업 시작 전 상태를 기억 (git stash로 저장 후 그대로 되돌려 놓음)
  python scripts/ml_bench.py check               # 다시 재서 기준과 비교 (정확도가 tol보다 떨어지면 실패, 종료 코드 1)
  python scripts/ml_bench.py check --rollback    # 실패하면 지금 변경을 stash에 보관하고 checkpoint 상태로 되돌림
  python scripts/ml_bench.py rollback            # 직접 되돌리기
  python scripts/ml_bench.py check --allow mnist/learner_lda --reason "rate 모드로 30배 빠름"
                                                 # 의도한 거래: 그 칸만 떨어져도 통과 (이유 필수, benchmarks/ml_allowances.jsonl에 기록)
  python scripts/ml_bench.py run --tasks odor    # 일부만 재서 보기 (비교 없음)

과제 (같은 seed면 같은 수: GPU 결정론적)
  odor   합성 냄새 20클래스 (synthetic_odors, 원형 5, 잡음 0.5), 학습 20 / 평가 50 x 클래스
  door   DoOR 실제 냄새 12개 (door_task, 잡음 0.8), 24 / 24 x 냄새
  mnist  MNIST 픽셀 앞 10,000 / 평가 앞 2,000
  cifar  CIFAR-100 ResNet-18 특징 (data/cifar100_resnet18_feats.npz), 클래스 100개를 10개씩 10과제로 차례로 배움
         (연속 학습: 과제마다 그 과제 데이터만) → 마지막에 100클래스 전체 정확도. lr_joint는 한꺼번에 (상한)
방법: learner_lda / learner_assoc / learner_backprop (fd.Learner 기본 회로), lr (로지스틱 회귀), mlp (은닉 256), cifar는
lr_finetune (과제마다 이어서 학습 - 잊는 기준선)
되돌리기는 아무것도 지우지 않음: 실패한 변경은 "ml_bench 되돌림" stash로 남음 (git stash list / git stash apply <해시>)
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE = ROOT / "benchmarks" / "ml_baseline.json"
HIST = ROOT / "benchmarks" / "history"                                    # 바뀌기 전 기준 (지우지 않고 날짜·커밋과 함께)
ALLOW_LOG = ROOT / "benchmarks" / "ml_allowances.jsonl"
CKPT = ROOT / ".verify" / "ml_checkpoint.json"
SEEDS = (0, 1)
TASKS = ("odor", "door", "mnist", "cifar")


def say(*a):
    print(*a, flush=True)


# ─────────────── 데이터 ───────────────
def _idx(path):
    with gzip.open(path, "rb") as f:
        b = f.read()
    nd = b[3]
    shape = tuple(int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd))
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(shape)


def data(task, seed):
    import flydnet as fd
    if task == "odor":
        return fd.synthetic_odors(20, 56, 20, 50, protos_per_class=5, noise=0.5, seed=seed)
    if task == "door":
        return fd.door_task(n_odors=12, samples=24, noise=0.8, seed=seed)
    if task == "mnist":
        raw = ROOT / "data" / "MNIST" / "raw"
        Xtr = _idx(raw / "train-images-idx3-ubyte.gz")[:10000].reshape(10000, -1).astype(np.float32) / 255
        ytr = _idx(raw / "train-labels-idx1-ubyte.gz")[:10000].astype(np.int64)
        Xte = _idx(raw / "t10k-images-idx3-ubyte.gz")[:2000].reshape(2000, -1).astype(np.float32) / 255
        yte = _idx(raw / "t10k-labels-idx1-ubyte.gz")[:2000].astype(np.int64)
        return Xtr, ytr, Xte, yte
    if task == "cifar":
        d = np.load(ROOT / "data" / "cifar100_resnet18_feats.npz")
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    raise ValueError(task)


# ─────────────── 일반 방법 (torch, CPU) ───────────────
def _torch_fit(Xtr, ytr, n_cls, hidden, seed, steps=300, lr=None, model=None, mu=None, sd=None):
    import torch
    torch.manual_seed(seed)
    if mu is None:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    y = torch.tensor(ytr)
    if model is None:
        d = X.shape[1]
        model = (torch.nn.Linear(d, n_cls) if not hidden else
                 torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, n_cls)))
    opt = torch.optim.Adam(model.parameters(), lr=lr or (1e-2 if not hidden else 1e-3), weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(X), y).backward()
        opt.step()
    return model, mu, sd


def _torch_acc(model, mu, sd, Xte, yte):
    import torch
    with torch.no_grad():
        p = model(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).argmax(1).numpy()
    return float((p == yte).mean())


# ─────────────── 한 칸 재기 ───────────────
def measure(task, method, seed):
    import flydnet as fd
    Xtr, ytr, Xte, yte = data(task, seed)
    n_cls = int(max(ytr.max(), yte.max())) + 1
    t0 = time.perf_counter()
    if task == "cifar":
        order = np.random.default_rng(seed).permutation(100)              # 클래스 순서 (seed마다)
        tasks = [order[i * 10:(i + 1) * 10] for i in range(10)]
        if method.startswith("learner_"):
            L = fd.Learner(rule=method[8:], seed=seed)
            for cl in tasks:
                m = np.isin(ytr, cl)
                L.learn(Xtr[m], ytr[m])
            acc = L.score(Xte, yte)
        elif method == "lr_finetune":
            model = mu = sd = None
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6                       # 특징 정규화만 전체 통계 (라벨 안 씀)
            for cl in tasks:
                m = np.isin(ytr, cl)
                model, mu, sd = _torch_fit(Xtr[m], ytr[m], 100, 0, seed, steps=100, model=model, mu=mu, sd=sd)
            acc = _torch_acc(model, mu, sd, Xte, yte)
        elif method == "lr_joint":
            acc = _torch_acc(*_torch_fit(Xtr, ytr, 100, 0, seed), Xte, yte)
        else:
            raise ValueError(method)
    elif method.startswith("learner_"):
        L = fd.Learner(rule=method[8:], seed=seed)
        L.learn(Xtr, ytr)
        acc = L.score(Xte, yte)
    elif method in ("lr", "mlp"):
        acc = _torch_acc(*_torch_fit(Xtr, ytr, n_cls, 0 if method == "lr" else 256, seed), Xte, yte)
    else:
        raise ValueError(method)
    return float(acc), time.perf_counter() - t0


METHODS = {t: ["learner_lda", "learner_assoc", "learner_backprop", "lr", "mlp"] for t in ("odor", "door", "mnist")}
METHODS["cifar"] = ["learner_lda", "learner_assoc", "lr_finetune", "lr_joint"]


def run(tasks):
    import warnings
    warnings.filterwarnings("ignore")
    res = {}
    for t in tasks:
        res[t] = {}
        for m in METHODS[t]:
            accs, secs = [], []
            for s in SEEDS:
                a, sec = measure(t, m, s)
                accs.append(a); secs.append(sec)
            res[t][m] = dict(acc=accs, sec=secs)
            say(f"  {t:<6} {m:<17} 정확도 {np.mean(accs) * 100:6.2f}%  ({', '.join(f'{a * 100:.2f}' for a in accs)})"
                f"  {np.mean(secs):7.1f}s")
    return res


def meta():
    import flydnet as fd
    head = git("rev-parse", "--short", "HEAD", check=False)
    dirty = bool(git("status", "--porcelain", check=False))
    return dict(date=datetime.datetime.now().isoformat(timespec="seconds"), version=fd.__version__,
                git=head + ("+변경" if dirty else ""), seeds=list(SEEDS))


# ─────────────── 비교 ───────────────
def compare(base, now, tol, time_ratio, allow=()):
    """반환: (실패, 경고, 허용된 하락 [(칸, 기준, 지금)])"""
    bad, warn, allowed = [], [], []
    for t, ms in base["results"].items():
        for m, b in ms.items():
            n = now.get(t, {}).get(m)
            if n is None:
                bad.append(f"{t}/{m}: 이번에 재지 않음 (기준에는 있음)")
                continue
            d = (np.mean(n["acc"]) - np.mean(b["acc"])) * 100
            if d < -tol and f"{t}/{m}" in allow:
                allowed.append((f"{t}/{m}", float(np.mean(b["acc"])), float(np.mean(n["acc"]))))
            elif d < -tol:
                bad.append(f"{t}/{m}: 정확도 {np.mean(b['acc']) * 100:.2f}% → {np.mean(n['acc']) * 100:.2f}% ({d:+.2f}%p)")
            r = np.mean(n["sec"]) / max(np.mean(b["sec"]), 1e-9)
            if r > time_ratio and np.mean(n["sec"]) > 1.0:
                warn.append(f"{t}/{m}: 시간 {np.mean(b['sec']):.1f}s → {np.mean(n['sec']):.1f}s ({r:.1f}배)")
    unused = set(allow) - {k for k, _, _ in allowed}
    warn += [f"{k}: 허용했지만 떨어지지 않음 (또는 없는 칸)" for k in sorted(unused)]
    return bad, warn, allowed


# ─────────────── 되돌리기 (git stash - 지우지 않음) ───────────────
def git(*a, check=True):
    # UTF-8로 읽음 (예전: 윈도우 기본 cp949로 읽다가 한글 stash 메시지에서 멈춰, 되돌리기가 반쯤 된 채로 끝났음)
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(a)} 실패: {(r.stderr or '').strip()}")
    return (r.stdout or "").strip()


def _stash(msg):
    """지금 변경(추적 안 되는 파일 포함, 무시 대상 제외)을 stash에 저장. 저장한 해시 (변경이 없으면 None)"""
    before = git("rev-parse", "-q", "--verify", "refs/stash", check=False)
    git("stash", "push", "-u", "-m", msg)
    after = git("rev-parse", "-q", "--verify", "refs/stash", check=False)
    return after if after and after != before else None


def checkpoint():
    h = _stash(f"ml_bench checkpoint {datetime.datetime.now():%Y-%m-%d %H:%M}")
    if h:
        git("stash", "apply", "--index", h)                                # 작업 트리는 그대로 (기억만 해 둠)
    CKPT.parent.mkdir(exist_ok=True)
    CKPT.write_text(json.dumps(dict(head=git("rev-parse", "HEAD"), stash=h,
                                    date=datetime.datetime.now().isoformat(timespec="seconds"))), encoding="utf-8")
    say(f"checkpoint: HEAD {git('rev-parse', '--short', 'HEAD')}" + (f" + 변경 (stash {h[:8]})" if h else " (변경 없음)"))


def rollback():
    if not CKPT.exists():
        raise SystemExit("checkpoint가 없음 - 먼저 python scripts/ml_bench.py checkpoint")
    c = json.loads(CKPT.read_text(encoding="utf-8"))
    if git("rev-parse", "HEAD") != c["head"]:
        raise SystemExit(f"checkpoint 뒤에 커밋이 바뀜 ({c['head'][:8]} → {git('rev-parse', '--short', 'HEAD')}) - "
                         "자동으로 되돌리지 않음. 직접 확인할 것")
    kept = _stash(f"ml_bench 되돌림: 실패한 변경 보관 {datetime.datetime.now():%Y-%m-%d %H:%M}")
    if c["stash"]:
        try:
            git("stash", "apply", "--index", c["stash"])
        except SystemExit as e:                                            # 반쯤 된 상태로 두지 않게 복구 방법을 알림
            raise SystemExit(f"{e}\n되돌리기 중 checkpoint 적용 실패 - 아무것도 지워지지 않음. 복구: "
                             f"git stash apply --index {c['stash'][:8]} (checkpoint), "
                             f"실패한 변경은 {kept[:8] if kept else '없음'}")
    say(f"되돌림: checkpoint({c['date']}) 상태로." +
        (f" 실패한 변경은 stash {kept[:8]}에 보관 (git stash apply {kept[:8]} 로 다시 볼 수 있음)" if kept else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["baseline", "check", "run", "checkpoint", "rollback"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--tol", type=float, default=0.5, help="정확도가 이만큼(%%p) 넘게 떨어지면 실패")
    ap.add_argument("--time-ratio", type=float, default=1.5, help="시간이 이 배수를 넘으면 경고")
    ap.add_argument("--rollback", action="store_true", help="check가 실패하면 checkpoint로 되돌림")
    ap.add_argument("--allow", action="append", default=[], help="떨어져도 되는 칸 '과제/방법' (여러 번 가능, --reason 필수)")
    ap.add_argument("--reason", default="", help="--allow의 이유 (기록에 남음)")
    a = ap.parse_args()
    if a.allow and not a.reason.strip():
        raise SystemExit("--allow에는 --reason \"이유\"가 필요함 (왜 떨어져도 되는지 기록)")
    if a.cmd == "checkpoint":
        return checkpoint()
    if a.cmd == "rollback":
        return rollback()
    tasks = [t for t in a.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown or not tasks:                                              # 예전: 잘못 쓴 이름을 걸러 아무것도 안 재고 "통과"
        raise SystemExit(f"알 수 없는 과제 {unknown} - 쓸 수 있는 것: {', '.join(TASKS)}" if unknown else "과제가 없음")
    if a.cmd == "check":
        if not BASE.exists():
            raise SystemExit("기준이 없음 - 먼저 python scripts/ml_bench.py baseline")
        base = json.loads(BASE.read_text(encoding="utf-8"))
        missing = [t for t in tasks if t not in base["results"]]
        if missing:                                                       # 기준에 없는 과제를 조용히 빼면 아무것도 안 재고 통과
            raise SystemExit(f"기준에 없는 과제 {missing} - 기준에 있는 것: {', '.join(base['results'])}")
        base["results"] = {t: base["results"][t] for t in tasks}
    say(f"[ml_bench {a.cmd}] 과제 {', '.join(tasks)}, seed {list(SEEDS)}")
    res = run(tasks)
    if a.cmd == "baseline":
        BASE.parent.mkdir(exist_ok=True)
        if BASE.exists():                                                 # 옛 기준은 지우지 않고 보관
            old = json.loads(BASE.read_text(encoding="utf-8"))["meta"]
            HIST.mkdir(exist_ok=True)
            stamp = old["date"].replace(":", "").replace("-", "")            # 날짜와 시각까지 (예전: 날짜만이라 같은 날 두 번
            tag = old["git"].replace("+변경", "_dirty").replace("+", "_")       # 저장하면 앞 보관본을 덮어씀) - 파일 이름은 영문만
            keep = HIST / f"ml_baseline_{stamp}_{tag}.json"
            keep.write_text(BASE.read_text(encoding="utf-8"), encoding="utf-8")
            say(f"옛 기준 보관: {keep.relative_to(ROOT)}")
        BASE.write_text(json.dumps(dict(meta=meta(), results=res), ensure_ascii=False, indent=1), encoding="utf-8")
        say(f"기준 저장: {BASE.relative_to(ROOT)}")
    elif a.cmd == "check":
        bad, warn, allowed = compare(base, res, a.tol, a.time_ratio, set(a.allow))
        for w in warn:
            say(f"  경고  {w}")
        for k, b0, n0 in allowed:
            say(f"  허용  {k}: {b0 * 100:.2f}% → {n0 * 100:.2f}% (이유: {a.reason})")
        if bad:
            say("실패 - 기준보다 나빠짐:")
            for b in bad:
                say(f"  ✗ {b}")
            if a.rollback:
                rollback()
            raise SystemExit(1)
        if allowed:                                                       # 통과했을 때만 기록 (예전: 실패해 되돌리면 기록이
            m = meta()                                                    # 실패한 변경과 함께 stash로 들어갔음)
            with open(ALLOW_LOG, "a", encoding="utf-8") as f:
                for k, b0, n0 in allowed:
                    f.write(json.dumps(dict(date=m["date"], git=m["git"], cell=k, base=b0, now=n0, reason=a.reason),
                                       ensure_ascii=False) + "\n")
        say(f"통과: 정확도가 기준보다 {a.tol}%p 넘게 떨어진 곳 없음 (기준 {base['meta']['git']}, {base['meta']['date']})")


if __name__ == "__main__":
    main()
