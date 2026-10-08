"""
실험 ⑩: 국소 운동 방향 — 실제 배선 대 무작위 배선

  python examples/visual_local_motion.py --circuit real
  python examples/visual_local_motion.py --circuit shuffled
  python examples/visual_local_motion.py --circuit real --freeze-circuit    # 리드아웃만 학습 (대조군)

실험 ⑨(전체 시야 격자)는 실제·무작위 배선 모두 100%로 풀렸지만, 뉴런 하나하나는 방향을 거의 가리지 않았음:
수천 개 뉴런 평균에 남은 0.1%짜리 차이를 리드아웃이 증폭한 지름길.
여기서는 시야를 칸(약 --region 기둥 크기)으로 나눠 칸마다 다른 방향(8개 중 무작위)·위상의 격자를 보여 주고,
칸마다 그 안의 T4a–d·T5a–d 아형별 평균(8개)으로 그 칸의 방향을 맞힘. 리드아웃은 모든 칸이 공유
→ 시야 어디서나 같은 아형이 같은 방향을 가리켜야(일관된 방향 선택성) 풀림. 칸마다 다른 우연한 차이는 못 씀.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd

ap = argparse.ArgumentParser()
ap.add_argument("--circuit", default="real", choices=["real", "shuffled", "local"],
                help="shuffled = 같은 유형 안에서 시야 전체로 섞음 (위치 대응까지 깨짐) / "
                     "local = 반경 --local-radius 기둥 안에서, T4a~d·T5a~d를 한 묶음으로 섞음 "
                     "(위치 대응은 유지, 아형별 방향 구조만 사라짐)")
ap.add_argument("--local-radius", type=float, default=2.0)
ap.add_argument("--shuffle-seed", type=int, default=0)
ap.add_argument("--freeze-circuit", action="store_true", help="회로는 학습하지 않고 리드아웃만")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--t-ms", type=float, default=300)
ap.add_argument("--count-from", type=float, default=50)
ap.add_argument("--dt", type=float, default=2.0)
ap.add_argument("--pathway", default="motion", choices=["motion", "full"],
                help="motion = 광수용체→라미나→메둘라→T4/T5 주요 유형만 (뉴런 2.4만, 연결 43만, 약 13배 빠름) / full = 시각계 전체")
ap.add_argument("--region", type=float, default=8.0, help="칸 한 변 (기둥 수)")
ap.add_argument("--min-per-type", type=int, default=15, help="칸 안 아형별 뉴런이 이보다 적으면 그 칸은 안 씀")
ap.add_argument("--wavelength", type=float, default=6.0)
ap.add_argument("--hz", type=float, default=4.0)
ap.add_argument("--noise", type=float, default=0.5, help="광수용체·프레임마다 독립인 밝기 잡음 표준편차")
ap.add_argument("--w-syn", type=float, default=3.0)
ap.add_argument("--bias", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=3e-2)
ap.add_argument("--lr-bias", type=float, default=0.01)
ap.add_argument("--lr-readout", type=float, default=1e-2)
ap.add_argument("--clip", type=float, default=1.0, help="회로 매개변수 기울기 노름 상한")
ap.add_argument("--warmup", type=int, default=50, help="학습률을 0에서 올리는 스텝 수 (이후 코사인으로 감소)")
ap.add_argument("--checkpoint-every", type=int, default=50)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/visual_local_motion")
args = ap.parse_args()
warnings.filterwarnings("ignore", message="중복 없이")
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.75)            # 넘치면 오류 (Windows 가상 메모리로 C: 채우지 않게)
DIRS = 8

vc = fd.visual_circuit()
xy_full = fd.column_map(vc)                                           # 지도는 실제 배선(시각계 전체)으로 만들고 모든 조건에 똑같이
keep = np.concatenate([vc.groups[g] for g in fd.MOTION_PATHWAY if g in vc.groups]) if args.pathway == "motion"     else np.arange(vc.N)
xy = xy_full[keep]                                              # subset은 그룹 순서대로 뉴런을 남김
if args.pathway == "motion":
    vc = vc.subset(fd.MOTION_PATHWAY)
if args.circuit == "real":
    circ = vc
elif args.circuit == "shuffled":
    circ = vc.shuffled(seed=args.shuffle_seed)
else:
    merge = {f"T{k}{d}": f"T{k}" for k in "45" for d in "abcd"}
    circ = vc.shuffled(seed=args.shuffle_seed, local=(xy, args.local_radius), merge=merge)
circ = circ.normalized()
OUT = [f"T{k}{d}" for k in "45" for d in "abcd"]
train = not args.freeze_circuit
engine = fd.ConnectomeLayer(circ, fd.PHOTORECEPTORS, OUT, t_ms=args.t_ms, dt=args.dt, neuron="graded",
                           params={"w_syn": args.w_syn}, device="gpu" if dev == "cuda" else "cpu",
                           bias={g: (0.0 if g in fd.PHOTORECEPTORS else args.bias) for g in circ.groups},
                           count_from_ms=args.count_from, trainable=train, share="pair",
                           train_neurons=train, checkpoint_every=args.checkpoint_every)
layer = fd.torch.bridge(engine)                                 # 계산은 자체 엔진, 학습 루프는 torch
print(engine, flush=True)

# 칸 나누기: 광수용체와 T4/T5 모두 시야 좌표로 칸 번호
in_np, out_np = fd.ganglion.backend.numpy(engine.in_idx), fd.ganglion.backend.numpy(engine.out_idx)
lo = np.nanmin(xy[np.r_[in_np, out_np]], 0)
cell = lambda idx: tuple(np.floor((xy[idx] - lo) / args.region).astype(int).T)
gx, gy = cell(out_np)
nx, ny = gx.max() + 1, gy.max() + 1
out_cell = gx * ny + gy
out_type = np.concatenate([np.full(len(circ.groups[g]), i) for i, g in enumerate(OUT)])
cnt = np.zeros((nx * ny, len(OUT)), int)
np.add.at(cnt, (out_cell, out_type), 1)
cells = np.nonzero((cnt >= args.min_per_type).all(1))[0]       # 아형 8개 모두 충분한 칸만
print(f"칸 {len(cells)}개 사용 (격자 {nx}×{ny}, 칸당 아형별 뉴런 중앙값 {int(np.median(cnt[cells]))})", flush=True)
pool = torch.zeros(len(cells), len(OUT), engine.n_out)
for r, c in enumerate(cells):
    for t in range(len(OUT)):
        m = (out_cell == c) & (out_type == t)
        pool[r, t, np.nonzero(m)[0]] = 1.0 / m.sum()
pool = pool.reshape(-1, engine.n_out).to(dev)                   # (칸×8, n_out) 밀집 - torch 희소 곱(cuSPARSE)은 실행마다 반올림이 달라 학습이 갈렸음
ix, iy = cell(in_np)
in_cell = np.clip(ix, 0, nx - 1) * ny + np.clip(iy, 0, ny - 1)
in_r = torch.tensor(np.searchsorted(cells, in_cell).clip(0, len(cells) - 1))
in_used = torch.tensor(np.isin(in_cell, cells))                 # 쓰지 않는 칸의 광수용체 → 회색 + 잡음
pr_xy = torch.tensor(np.nan_to_num(xy[in_np]))
readout = nn.Linear(len(OUT), DIRS).to(dev)                     # 모든 칸이 공유
norm = lambda f: (f - f.mean((0, 1))) / f.std((0, 1)).clamp_min(1e-6)   # 묶음·칸 전체 통계 (라벨 안 씀)


def stimulus(y, ph, gen=None, noise=None):
    """y, ph: (B, 칸) → 광수용체 밝기 (B, T, n_in). 광수용체마다 자기 칸의 격자"""
    noise = args.noise if noise is None else noise
    frames = int(args.t_ms / 2)
    t = (torch.arange(frames) + 0.5) * args.t_ms / frames / 1000.0
    th = (y.float() * 360.0 / DIRS * np.pi / 180)[:, in_r]                  # (B, n_in)
    k = 2 * np.pi / args.wavelength
    proj = k * (torch.cos(th) * pr_xy[:, 0] + torch.sin(th) * pr_xy[:, 1]) + ph[:, in_r]
    lum = torch.sin(proj[:, None, :] - 2 * np.pi * args.hz * t[None, :, None]) * in_used
    if noise:
        lum = lum + noise * torch.randn(lum.shape, generator=gen)
    return 0.5 * (1 + lum.clamp(-1, 1))


def sample(n, gen):
    y = torch.randint(0, DIRS, (n, len(cells)), generator=gen)
    ph = torch.rand(n, len(cells), generator=gen) * 2 * np.pi
    return stimulus(y, ph, gen), y


def features(x):
    return (pool @ layer(x.to(dev)).T).T.reshape(len(x), len(cells), len(OUT))       # (B, 칸, 8)


def evaluate(n=64, seed=12345):
    """seed 12345 = 검증 세트 (가장 좋은 시점 고르기), 최종 평가는 다른 seed (고른 시점에 유리한 편향 없게)"""
    g = torch.Generator().manual_seed(seed)
    F, Y = [], []
    with torch.no_grad():
        for _ in range(n // args.batch):
            x, y = sample(args.batch, g)
            F.append(features(x)); Y.append(y)
        F, Y = torch.cat(F), torch.cat(Y)
        logit = readout(norm(F)).cpu()
    pred = logit.argmax(-1)
    loss = nn.functional.cross_entropy(logit.reshape(-1, DIRS), Y.reshape(-1)).item()
    return (pred == Y).float().mean().item(), (pred == (Y + DIRS // 2) % DIRS).float().mean().item(), loss


def neuron_dsi(reps=4):
    """뉴런별 방향 선택 지수 (전체 시야 격자, 잡음 없이). 실제 T4/T5는 대략 0.3~0.8.
    agree = 같은 아형 뉴런들의 선호 방향 일치도 (0~1)"""
    g = torch.Generator().manual_seed(999)
    R = torch.zeros(DIRS, engine.n_out)
    with torch.no_grad():
        for _ in range(reps):
            ph = (torch.rand(DIRS, 1, generator=g) * 2 * np.pi).expand(-1, len(cells))
            y = torch.arange(DIRS)[:, None].expand(-1, len(cells))
            R += layer(stimulus(y, ph, noise=0.0).to(dev)).cpu() / reps
    th = torch.tensor(np.radians(np.arange(DIRS) * 360.0 / DIRS), dtype=torch.float32)
    v = (R * torch.exp(1j * th)[:, None]).sum(0)
    d = (v.abs() / R.sum(0).clamp_min(1e-9)).numpy()
    u = np.exp(1j * np.angle(v.numpy()))
    res = {}
    for t, gname in enumerate(OUT):
        m = out_type == t
        res[gname] = dict(median=float(np.median(d[m])), p90=float(np.percentile(d[m], 90)),
                          agree=float(abs(u[m].mean())), pref=float(np.degrees(np.angle(u[m].mean()))))
    return res


params = [{"params": readout.parameters(), "lr": args.lr_readout}]
if train:
    params += [{"params": [layer.log_scale, layer.log_t_mbr], "lr": args.lr},
               {"params": [layer.bias], "lr": args.lr_bias}]
opt = torch.optim.Adam(params)
circuit_params = [layer.log_scale, layer.log_t_mbr, layer.bias] if train else []
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                          0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))
gen = torch.Generator().manual_seed(args.seed)
hist = []
best = dict(loss=float("inf"), step=0)


def snapshot():
    return dict(layer={k: v.detach().clone() for k, v in layer.state_dict().items()},
                readout={k: v.detach().clone() for k, v in readout.state_dict().items()})
t0 = time.time()
for step in range(1, args.steps + 1):
    x, y = sample(args.batch, gen)
    logit = readout(norm(features(x)))
    loss = nn.functional.cross_entropy(logit.reshape(-1, DIRS), y.reshape(-1).to(dev))
    opt.zero_grad(); loss.backward()
    if train:
        for p in circuit_params:
            torch.nan_to_num_(p.grad, 0.0)
        torch.nn.utils.clip_grad_norm_(circuit_params, args.clip)   # 한 번에 크게 망가지지 않게
    opt.step(); sched.step()
    if train:
        with torch.no_grad():
            layer.log_t_mbr.clamp_(np.log(1.0), np.log(100.0))
            layer.bias.clamp_(-2.0, 2.0)
    if step % 10 == 0:
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        print(f"step {step}: loss {loss.item():.3f}, {(time.time() - t0) / step:.1f}s/step, GPU {mem:.2f}GB", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        acc, opp, vloss = evaluate()
        mark = ""
        if vloss < best["loss"]:
            best = dict(loss=vloss, step=step, state=snapshot()); mark = " *"
        print(f"  eval step {step}: acc {acc * 100:.1f}% (정반대 {opp * 100:.1f}%) 검증 손실 {vloss:.3f}{mark}", flush=True)
        hist.append(dict(step=step, acc=acc, opp=opp, loss=loss.item(), val_loss=vloss))

layer.load_state_dict(best["state"]["layer"]); readout.load_state_dict(best["state"]["readout"])   # 제자리 복사 → 엔진에도
print(f"검증 손실이 가장 낮았던 step {best['step']} ({best['loss']:.3f})의 모델로 최종 평가", flush=True)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
tag = (f"{args.circuit}{args.shuffle_seed if args.circuit != 'real' else ''}"
       f"{'_frozen' if args.freeze_circuit else ''}_{args.pathway}_r{args.region:g}_s{args.seed}")
acc, opp, test_loss = evaluate(256, seed=54321)
nd = neuron_dsi()
print(f"\n최종 ({tag}, 256개 × 칸 {len(cells)}개): acc {acc * 100:.1f}%, 정반대 방향 오답 {opp * 100:.1f}%", flush=True)
print("뉴런별 방향 선택 지수 (중앙값/상위10%/선호방향 일치/선호방향):",
      " ".join(f"{k}:{v['median']:.3f}/{v['p90']:.3f}/{v['agree']:.2f}/{v['pref']:.0f}°" for k, v in nd.items()),
      flush=True)
(out / f"{tag}.json").write_text(json.dumps(dict(args=vars(args), hist=hist,
                                                 final=dict(acc=acc, opp=opp, test_loss=test_loss, best_step=best["step"],
                                                            neuron_dsi=nd, n_cells=len(cells))),
                                            ensure_ascii=False, indent=1), encoding="utf-8")
if train:
    engine.save(out / f"{tag}_layer")                             # 자체 엔진 저장 (.npz)
torch.save(readout.state_dict(), out / f"{tag}_readout.pt")
