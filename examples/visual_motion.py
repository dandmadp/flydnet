"""
실험 ⑨: 시각계 운동 방향 학습 — 실제 배선 대 무작위 배선

  python examples/visual_motion.py --circuit real
  python examples/visual_motion.py --circuit shuffled

광수용체에 움직이는 격자(8방향, 위상 무작위)를 넣고, T4a–d·T5a–d 아형별 평균 발화율 8개로 방향을 맞힘 (찍기 12.5%).
시야 전체를 평균하므로 뉴런 하나하나가 방향을 가려야 풀림. 결이(방위)만 구분하면 정반대 방향을 헷갈려 최대 50%
→ 50%를 넘으면 진짜 방향 선택성.
학습: 연결 종류별 세기(share="pair"), 세포 유형별 bias·막 시간 상수 (Lappalainen et al. 2024와 같은 방식).
뉴런: 기본은 연속값 뉴런(--neuron graded) + 받는 뉴런별 입력 정규화. 스파이킹(--neuron lif)은 운동 신호가
T4/T5까지 거의 전달되지 않아 학습이 안 됨 (라미나·메둘라 뉴런은 실제로도 스파이크 없이 막전위로 신호를 보냄).
무작위 대조군: 세포 유형 쌍별 연결 수·뉴런별 입출력 수는 같고, 같은 유형 안에서 '누가 누구에게'만 섞음
→ 시야 위치(기둥) 대응과 Mi4/Mi9 어긋남 같은 미세 배선만 사라짐.
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
ap.add_argument("--circuit", default="real", choices=["real", "shuffled"])
ap.add_argument("--shuffle-seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--neuron", default="graded", choices=["graded", "lif"])
ap.add_argument("--t-ms", type=float, default=300)
ap.add_argument("--count-from", type=float, default=50)
ap.add_argument("--dt", type=float, default=None, help="기본: graded 1.0, lif 0.5")
ap.add_argument("--dirs", type=int, default=8)
ap.add_argument("--wavelength", type=float, default=8.0)
ap.add_argument("--hz", type=float, default=4.0)
ap.add_argument("--noise", type=float, default=0.5,
                help="광수용체·프레임마다 독립인 밝기 잡음 표준편차 (격자 진폭 1 기준). 잡음이 없으면 결정론적 시뮬레이션의 "
                     "아주 작은 차이(0.1%%)까지 리드아웃이 증폭해 맞혀 버림")
ap.add_argument("--w-syn", type=float, default=None, help="기본: graded 3.0 (정규화 회로), lif 0.05")
ap.add_argument("--bias", type=float, default=None, help="기본: graded 0.2, lif 10 mV")
ap.add_argument("--lr", type=float, default=3e-2, help="연결 종류별 log 배율, log 막 시간 상수")
ap.add_argument("--lr-bias", type=float, default=None, help="bias. 기본: graded 0.01, lif 0.3 mV")
ap.add_argument("--lr-readout", type=float, default=1e-2)
ap.add_argument("--readout", default="t4t5", choices=["t4t5", "lptc"])
ap.add_argument("--checkpoint-every", type=int, default=50)
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/visual_motion")
args = ap.parse_args()
G = args.neuron == "graded"
args.dt = args.dt or (1.0 if G else 0.5)
args.w_syn = args.w_syn or (3.0 if G else 0.05)
args.lr_bias = args.lr_bias or (0.01 if G else 0.3)
args.bias = args.bias if args.bias is not None else (0.2 if G else 10.0)
warnings.filterwarnings("ignore", message="중복 없이")
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.75)            # 넘치면 오류 (Windows 가상 메모리로 C: 채우지 않게)

vc = fd.visual_circuit()
xy = fd.column_map(vc)                                                # 지도는 실제 배선으로 만들고 두 조건에 똑같이 씀
circ = vc if args.circuit == "real" else vc.shuffled(seed=args.shuffle_seed)
if G:
    circ = circ.normalized()
OUT = [f"T{k}{d}" for k in "45" for d in "abcd"] if args.readout == "t4t5" else [g for g in fd.LPTC if g in vc.groups]
engine = fd.ConnectomeLayer(circ, fd.PHOTORECEPTORS, OUT, t_ms=args.t_ms, dt=args.dt, input_mode="regular",
                           neuron=args.neuron, device="gpu" if dev == "cuda" else "cpu",
                           params={"w_syn": args.w_syn} if G else {"w_syn": args.w_syn, "f_poi": 250 * 0.275 / args.w_syn},
                           bias={g: (0.0 if g in fd.PHOTORECEPTORS else args.bias) for g in circ.groups},
                           v_init="random", count_from_ms=args.count_from, trainable=True, share="pair",
                           train_neurons=True, checkpoint_every=args.checkpoint_every)
layer = fd.torch.bridge(engine, seed=0)                         # 계산은 자체 엔진, 학습 루프는 torch
print(engine, flush=True)
pr_xy = xy[fd.ganglion.backend.numpy(engine.in_idx)]
# 출력 뉴런 → 그룹 평균 (T4/T5는 아형별 평균, LPTC는 뉴런이 적어 그룹 평균도 같은 방식)
sizes = [len(circ.groups[g]) for g in OUT]
pool = torch.zeros(len(OUT), engine.n_out, device=dev)
o = 0
for i, n in enumerate(sizes):
    pool[i, o:o + n] = 1.0 / n; o += n
readout = nn.Linear(len(OUT), args.dirs).to(dev)
# 특징 표준화는 묶음 자체의 평균·표준편차로 (라벨은 안 씀). 회로가 학습되며 특징 크기가 계속 바뀌므로
# 이동 평균(BatchNorm의 평가 모드)을 쓰면 학습·평가가 어긋남
norm = lambda f: (f - f.mean(0)) / f.std(0).clamp_min(1e-6)


def batch(n, gen):
    y = torch.randint(0, args.dirs, (n,), generator=gen)
    ph = torch.rand(n, generator=gen) * 2 * np.pi
    lum = torch.from_numpy(fd.drifting_grating(pr_xy, y.numpy() * 360.0 / args.dirs, args.t_ms, int(args.t_ms / 2),
                                               wavelength=args.wavelength, temporal_hz=args.hz, phase=ph.numpy()))
    lum = (lum + args.noise * torch.randn(lum.shape, generator=gen)).clamp(-1, 1)
    return ((0.5 if G else 100.0) * (1 + lum)).to(dev), y


def forward(x):
    r = layer(x)                                               # (B, n_out) Hz
    f = r @ pool.T                                             # (B, 8) 그룹 평균
    return readout(norm(f)), f


def evaluate(n=64):
    g = torch.Generator().manual_seed(12345)
    F, Y = [], []
    with torch.no_grad():
        for _ in range(n // args.batch):
            x, y = batch(args.batch, g)
            F.append(layer(x) @ pool.T); Y.append(y)
        F, Y = torch.cat(F), torch.cat(Y)
        pred = readout(norm(F)).argmax(1).cpu()               # 평가 묶음 전체의 통계로 표준화
    F = F.cpu()
    correct = (pred == Y).sum().item()
    opp = (pred == (Y + args.dirs // 2) % args.dirs).sum().item()
    # 아형별 방향 선택 지수 (평균 응답의 방향 벡터 길이)
    th = torch.tensor(np.radians(np.arange(args.dirs) * 360.0 / args.dirs), dtype=torch.float32)
    tune = torch.stack([F[Y == d].mean(0) for d in range(args.dirs)])        # (dirs, 8)
    vec = (tune * torch.exp(1j * th)[:, None]).sum(0) / tune.sum(0).clamp_min(1e-9)
    dsi = {g: (round(abs(v).item(), 4), round(np.degrees(np.angle(v.item())))) for g, v in zip(OUT, vec)}
    return correct / len(Y), opp / len(Y), dsi


opt = torch.optim.Adam([
    {"params": [layer.log_scale, layer.log_t_mbr], "lr": args.lr},    # bridge의 Parameter = 엔진 값과 같은 메모리
    {"params": [layer.bias], "lr": args.lr_bias},
    {"params": readout.parameters(), "lr": args.lr_readout},
])
gen = torch.Generator().manual_seed(args.seed)
hist = []
acc, opp, dsi = evaluate()
print(f"step 0: acc {acc * 100:.1f}% (opposite {opp * 100:.1f}%) DSI {dsi}", flush=True)
hist.append(dict(step=0, acc=acc, opp=opp, dsi=dsi))
t0 = time.time()
for step in range(1, args.steps + 1):
    x, y = batch(args.batch, gen)
    logit, f = forward(x)
    loss = nn.functional.cross_entropy(logit, y.to(dev))
    opt.zero_grad(); loss.backward()
    for p in (layer.log_scale, layer.log_t_mbr, layer.bias):
        torch.nan_to_num_(p.grad, 0.0)
    opt.step()
    with torch.no_grad():
        layer.log_t_mbr.clamp_(np.log(1.0), np.log(100.0))     # 막 시간 상수 1~100 ms
        layer.bias.clamp_(*((-2.0, 2.0) if G else (-20.0, 30.0)))
    if step % 5 == 0:
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        print(f"step {step}: loss {loss.item():.3f}, {(time.time() - t0) / step:.1f}s/step, GPU {mem:.2f}GB", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        acc, opp, dsi = evaluate()
        print(f"  eval step {step}: acc {acc * 100:.1f}% (opposite {opp * 100:.1f}%) DSI {dsi}", flush=True)
        hist.append(dict(step=step, acc=acc, opp=opp, dsi=dsi, loss=loss.item()))

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
tag = f"{args.neuron}_{args.circuit}{args.shuffle_seed if args.circuit == 'shuffled' else ''}_{args.readout}_s{args.seed}"
acc, opp, dsi = evaluate(256)
print(f"\n최종 ({tag}, 256개): acc {acc * 100:.1f}%, 정반대 방향 오답 {opp * 100:.1f}%", flush=True)


def neuron_dsi(reps=4):
    """뉴런 하나하나의 방향 선택 지수 (방향마다 위상 reps개 평균 응답 → 벡터 길이 / 합, 잡음 없이).
    실제 T4/T5는 대략 0.3~0.8. agree = 같은 아형 뉴런들의 선호 방향이 얼마나 일치하는지 (0~1)"""
    g = torch.Generator().manual_seed(999)
    th = np.arange(args.dirs) * 360.0 / args.dirs
    R = torch.zeros(args.dirs, engine.n_out)
    with torch.no_grad():
        for _ in range(reps):
            lum = torch.from_numpy(fd.drifting_grating(pr_xy, th, args.t_ms, int(args.t_ms / 2), wavelength=args.wavelength,
                                                       temporal_hz=args.hz,
                                                       phase=(torch.rand(args.dirs, generator=g) * 2 * np.pi).numpy()))
            R += layer(((0.5 if G else 100.0) * (1 + lum)).to(dev)).cpu() / reps
    v = (R * torch.exp(1j * torch.tensor(np.radians(th), dtype=torch.float32))[:, None]).sum(0)
    d = (v.abs() / R.sum(0).clamp_min(1e-9)).numpy()
    u = np.exp(1j * np.angle(v.numpy()))
    res, o = {}, 0
    for gname, n in zip(OUT, sizes):
        res[gname] = dict(median=float(np.median(d[o:o + n])), p90=float(np.percentile(d[o:o + n], 90)),
                          agree=float(abs(u[o:o + n].mean())))
        o += n
    return res


nd = neuron_dsi()
print("뉴런별 방향 선택 지수 (중앙값/상위10%/선호방향 일치):",
      " ".join(f"{k}:{v['median']:.3f}/{v['p90']:.3f}/{v['agree']:.2f}" for k, v in nd.items()), flush=True)
(out / f"{tag}.json").write_text(json.dumps(dict(args=vars(args), hist=hist, final=dict(acc=acc, opp=opp, dsi=dsi,
                                                                                       neuron_dsi=nd)),
                                           ensure_ascii=False, indent=1), encoding="utf-8")
engine.save(out / f"{tag}_layer")                                 # 자체 엔진 저장 (.npz)
torch.save(readout.state_dict(), out / f"{tag}_readout.pt")
