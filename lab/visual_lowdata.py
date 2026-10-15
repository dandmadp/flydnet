"""
실험 ⑬: 배선이 이기는 조건이 있나 - 적은 데이터·처음 보는 조건에서 국소 운동 방향 (visual_local_motion과 같은 과제)

  python lab/visual_lowdata.py --model real  --n-train 16 --seed 0
  python lab/visual_lowdata.py --model local --n-train 16 --seed 0     # 위치는 지키고 T4a-d·T5a-d 아형만 섞음
  python lab/visual_lowdata.py --model cnn   --n-train 16 --seed 0     # 일반 신경망: 칸마다 규칙 격자의 같은 자극
  python lab/visual_lowdata.py --model real  --n-train 0               # 0 = 매 스텝 새 자극 (데이터 무한, 예제와 같음)

가설: 실제 배선 + 세포 유형 단위 매개변수(그룹 쌍 배율·유형별 바이어스·시간 상수 - 수백 개)는 데이터가 적거나 학습 때
없던 속도·대비·잡음에서 일반 신경망보다 잘 일반화한다 (Lappalainen et al. 2024가 쓴 방식).
학습 자극 n-train개를 처음에 고정 (잡음 포함) → 스텝마다 그중 batch개. 자극 하나 = 시야의 모든 칸 (칸마다 다른 방향).
최종 평가 (다른 seed의 새 자극): 학습과 같은 조건 + 느림(2 Hz)·빠름(8 Hz)·낮은 대비(0.3)·큰 잡음(1.0).
결과: --out의 JSON (조건별 정확도, 학습 매개변수 수)
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="real", choices=["real", "local", "cnn"])
ap.add_argument("--n-train", type=int, default=16, help="고정 학습 자극 수 (0 = 매 스텝 새 자극)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--t-ms", type=float, default=300)
ap.add_argument("--count-from", type=float, default=50)
ap.add_argument("--dt", type=float, default=2.0)
ap.add_argument("--region", type=float, default=8.0)
ap.add_argument("--min-per-type", type=int, default=15)
ap.add_argument("--wavelength", type=float, default=6.0)
ap.add_argument("--hz", type=float, default=4.0)
ap.add_argument("--noise", type=float, default=0.5)
ap.add_argument("--local-radius", type=float, default=2.0)
ap.add_argument("--w-syn", type=float, default=3.0)
ap.add_argument("--bias", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=3e-2)
ap.add_argument("--lr-bias", type=float, default=0.01)
ap.add_argument("--lr-readout", type=float, default=1e-2)
ap.add_argument("--lr-cnn", type=float, default=3e-3)
ap.add_argument("--grid", type=int, default=5, help="cnn: 칸마다 k x k 규칙 격자")
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--checkpoint-every", type=int, default=50)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--n-val", type=int, default=64)
ap.add_argument("--n-test", type=int, default=256)
ap.add_argument("--out", default="data/visual_lowdata")
args = ap.parse_args()
warnings.filterwarnings("ignore", message="중복 없이")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.75)
DIRS = 8
OUT = [f"T{k}{d}" for k in "45" for d in "abcd"]
FRAMES = int(args.t_ms / 2)

# ───────────── 회로·칸 (모델과 상관없이 같은 칸 → 같은 과제) ─────────────
vc = fd.visual_circuit()
xy_full = fd.column_map(vc)
keep = np.concatenate([vc.groups[g] for g in fd.MOTION_PATHWAY if g in vc.groups])
xy = xy_full[keep]
vc = vc.subset(fd.MOTION_PATHWAY)
if args.model == "local":
    merge = {f"T{k}{d}": f"T{k}" for k in "45" for d in "abcd"}
    circ = vc.shuffled(seed=args.seed, local=(xy, args.local_radius), merge=merge)
else:
    circ = vc
circ = circ.normalized()
engine = fd.ConnectomeLayer(circ, fd.PHOTORECEPTORS, OUT, t_ms=args.t_ms, dt=args.dt, neuron="graded",
                            params={"w_syn": args.w_syn}, device="gpu" if dev == "cuda" else "cpu",
                            bias={g: (0.0 if g in fd.PHOTORECEPTORS else args.bias) for g in circ.groups},
                            count_from_ms=args.count_from, trainable=True, share="pair",
                            train_neurons=True, checkpoint_every=args.checkpoint_every)
in_np, out_np = fd.ganglion.backend.numpy(engine.in_idx), fd.ganglion.backend.numpy(engine.out_idx)
lo = np.nanmin(xy[np.r_[in_np, out_np]], 0)
cell = lambda idx: tuple(np.floor((xy[idx] - lo) / args.region).astype(int).T)
gx, gy = cell(out_np)
nx, ny = gx.max() + 1, gy.max() + 1
out_cell = gx * ny + gy
out_type = np.concatenate([np.full(len(circ.groups[g]), i) for i, g in enumerate(OUT)])
cnt = np.zeros((nx * ny, len(OUT)), int)
np.add.at(cnt, (out_cell, out_type), 1)
cells = np.nonzero((cnt >= args.min_per_type).all(1))[0]
NC = len(cells)
ix, iy = cell(in_np)
in_cell = np.clip(ix, 0, nx - 1) * ny + np.clip(iy, 0, ny - 1)
in_r = torch.tensor(np.searchsorted(cells, in_cell).clip(0, NC - 1))
in_used = torch.tensor(np.isin(in_cell, cells))
pr_xy = torch.tensor(np.nan_to_num(xy[in_np]))
# cnn: 칸마다 k x k 규칙 격자 (칸 안 같은 좌표계 - 광수용체가 받는 것과 같은 격자 무늬)
k = args.grid
off = (np.arange(k) + 0.5) / k * args.region
cxy = np.stack(np.divmod(cells, ny), 1).astype(np.float64) * args.region + lo    # 칸 왼쪽 아래 (칸 번호 = gx * ny + gy)
grid_xy = torch.tensor((cxy[:, None, None, :] + np.stack(np.meshgrid(off, off, indexing="ij"), -1)[None])
                       .reshape(NC, k * k, 2))


def lum(y, ph, pos, cell_of, used, hz, contrast, noise, gen):
    """밝기 (B, T, 점): 점마다 자기 칸의 격자 (방향 y, 위상 ph: (B, 칸))"""
    t = (torch.arange(FRAMES) + 0.5) * args.t_ms / FRAMES / 1000.0
    th = (y.float() * 360.0 / DIRS * np.pi / 180)[:, cell_of]
    kk = 2 * np.pi / args.wavelength
    proj = kk * (torch.cos(th) * pos[:, 0] + torch.sin(th) * pos[:, 1]) + ph[:, cell_of]
    v = contrast * torch.sin(proj[:, None, :] - 2 * np.pi * hz * t[None, :, None]) * used
    if noise:
        v = v + noise * torch.randn(v.shape, generator=gen)
    return 0.5 * (1 + v.clamp(-1, 1))


def make(n, gen, hz=None, contrast=1.0, noise=None):
    """자극 n개: 모델 입력과 라벨 (B, 칸)"""
    hz = args.hz if hz is None else hz
    noise = args.noise if noise is None else noise
    y = torch.randint(0, DIRS, (n, NC), generator=gen)
    ph = torch.rand(n, NC, generator=gen) * 2 * np.pi
    if args.model == "cnn":
        flat = grid_xy.reshape(-1, 2)
        cell_of = torch.arange(NC).repeat_interleave(k * k)
        x = lum(y, ph, flat, cell_of, torch.ones(len(flat), dtype=torch.bool), hz, contrast, noise, gen)
        x = x.reshape(n, FRAMES, NC, k, k).permute(0, 2, 1, 3, 4)            # (B, 칸, T, k, k)
    else:
        x = lum(y, ph, pr_xy, in_r, in_used, hz, contrast, noise, gen)      # (B, T, n_in)
    return x.float(), y                                                      # 좌표가 float64라 밝기도 float64였음


# ───────────── 모델 ─────────────
if args.model == "cnn":
    class CellCNN(nn.Module):
        """칸마다 같은 작은 시공간 합성곱 (모든 칸 공유) → 8개 특징 (T4/T5 아형 8개와 같은 수)"""
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 8, (9, 3, 3))
            self.pool = nn.AdaptiveAvgPool3d(1)

        def forward(self, x):                                                # (B, 칸, T, k, k)
            B = x.shape[0]
            h = torch.relu(self.conv(x.reshape(B * NC, 1, FRAMES, k, k) - 0.5))
            return self.pool(h).reshape(B, NC, 8)
    net = CellCNN().to(dev)
    features = lambda x: net(x.to(dev))
    circuit_params = list(net.parameters())
    groups = [{"params": circuit_params, "lr": args.lr_cnn}]
else:
    layer = fd.torch.bridge(engine)
    pool = torch.zeros(NC, len(OUT), engine.n_out)
    for r, c in enumerate(cells):
        for t in range(len(OUT)):
            m = (out_cell == c) & (out_type == t)
            pool[r, t, np.nonzero(m)[0]] = 1.0 / m.sum()
    pool = pool.reshape(-1, engine.n_out).to(dev)
    features = lambda x: (pool @ layer(x.to(dev)).T).T.reshape(len(x), NC, len(OUT))
    circuit_params = [layer.log_scale, layer.log_t_mbr, layer.bias]
    groups = [{"params": [layer.log_scale, layer.log_t_mbr], "lr": args.lr}, {"params": [layer.bias], "lr": args.lr_bias}]
readout = nn.Linear(len(OUT), DIRS).to(dev)
norm = lambda f: (f - f.mean((0, 1))) / f.std((0, 1)).clamp_min(1e-6)
n_params = sum(int(p.numel()) for p in circuit_params) + sum(int(p.numel()) for p in readout.parameters())
print(f"모델 {args.model}, 학습 자극 {args.n_train or '무한'}개, 칸 {NC}개, 학습 매개변수 {n_params:,}개", flush=True)

opt = torch.optim.Adam(groups + [{"params": readout.parameters(), "lr": args.lr_readout}])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                          0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))
gen = torch.Generator().manual_seed(args.seed)
if args.n_train:
    Xtr, Ytr = make(args.n_train, gen)                                     # 고정 학습 자극 (잡음 포함)


def evaluate(n, seed, **cond):
    g = torch.Generator().manual_seed(seed)
    acc, loss = [], []
    with torch.no_grad():
        for _ in range(max(1, n // args.batch)):
            x, y = make(args.batch, g, **cond)
            logit = readout(norm(features(x))).cpu()
            acc.append((logit.argmax(-1) == y).float().mean().item())
            loss.append(nn.functional.cross_entropy(logit.reshape(-1, DIRS), y.reshape(-1)).item())
    return float(np.mean(acc)), float(np.mean(loss))


def snapshot():
    mods = [readout] + ([net] if args.model == "cnn" else [layer])
    return [{kk: v.detach().clone() for kk, v in m.state_dict().items()} for m in mods]


best = dict(loss=float("inf"), step=0)
t0 = time.time()
for step in range(1, args.steps + 1):
    if args.n_train:
        j = torch.randint(0, args.n_train, (args.batch,), generator=gen)
        x, y = Xtr[j], Ytr[j]
    else:
        x, y = make(args.batch, gen)
    loss = nn.functional.cross_entropy(readout(norm(features(x))).reshape(-1, DIRS), y.reshape(-1).to(dev))
    opt.zero_grad(); loss.backward()
    for p in circuit_params:
        if p.grad is not None:
            torch.nan_to_num_(p.grad, 0.0)
    torch.nn.utils.clip_grad_norm_(circuit_params, args.clip)
    opt.step(); sched.step()
    if args.model != "cnn":
        with torch.no_grad():
            layer.log_t_mbr.clamp_(np.log(1.0), np.log(100.0))
            layer.bias.clamp_(-2.0, 2.0)
    if step % args.eval_every == 0 or step == args.steps:
        acc, vloss = evaluate(args.n_val, 12345)
        mark = ""
        if vloss < best["loss"]:
            best = dict(loss=vloss, step=step, state=snapshot()); mark = " *"
        print(f"step {step}: loss {loss.item():.3f}, 검증 acc {acc * 100:.1f}% 손실 {vloss:.3f}{mark} "
              f"({(time.time() - t0) / step:.2f}s/step)", flush=True)

for m, st in zip([readout] + ([net] if args.model == "cnn" else [layer]), best["state"]):
    m.load_state_dict(st)
conds = {"같은 조건": {}, "느림 2Hz": dict(hz=2.0), "빠름 8Hz": dict(hz=8.0), "낮은 대비 0.3": dict(contrast=0.3),
         "큰 잡음 1.0": dict(noise=1.0)}
res = {name: evaluate(args.n_test, 54321 + i, **c)[0] for i, (name, c) in enumerate(conds.items())}
tag = f"{args.model}_n{args.n_train}_s{args.seed}"
print(f"\n최종 ({tag}, 가장 좋은 step {best['step']}): " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in res.items()),
      flush=True)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
(out / f"{tag}.json").write_text(json.dumps(dict(args=vars(args), n_params=n_params, best_step=best["step"],
                                                 seconds=time.time() - t0, acc=res), ensure_ascii=False, indent=1),
                                 encoding="utf-8")
