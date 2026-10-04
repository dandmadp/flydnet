"""
사용자 정의 뉴런 모델 예: AdEx (적응형 지수 적분-발화, Brette & Gerstner 2005)를 직접 정의해 버섯체에 넣기

  python examples/custom_neuron.py              # 약 1분 (GPU)

정의한 것은 한 스텝뿐인데 따라오는 것:
  시뮬레이션 (지연·입력·체크포인팅) / 역전파 (자동 미분) / fd.genetics (끄기·켜기) / fd.explain / 저장·불러오기
explain의 확인(verify)으로 기울기가 믿을 만한지 실제로 꺼서 확인한다 (감쇠 없이는 방향이 틀렸음)
속도: 같은 계산을 내장 LIF (손으로 유도한 합친 연산)와 플러그인 LIF로 재서 비교
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 설치 없이 실행
import flydnet as fd
from flydnet.ganglion import Signal, fire, where


@fd.neurons.register
class AdEx(fd.neurons.NeuronModel):
    """C dv/dt = -gL (v - EL) + gL ΔT exp((v - VT)/ΔT) - w + g,  τw dw/dt = a (v - EL) - w
    v > 0 mV면 v = Vr, w += b. 시냅스 입력 g는 tau_syn으로 감쇠. 단위: mV, ms (C/gL = τm)"""
    state = ("v", "w", "g")

    def __init__(self, tau_m=20.0, EL=-70.0, VT=-50.0, dT=2.0, Vr=-58.0, a=0.5, b=7.0, tau_w=100.0, tau_syn=5.0,
                 gL=1.0, slope=1.0):
        super().__init__(tau_m=tau_m, EL=EL, VT=VT, dT=dT, Vr=Vr, a=a, b=b, tau_w=tau_w, tau_syn=tau_syn, gL=gL,
                         slope=slope)

    def init(self, ctx):
        q = self.params
        return {"v": ctx.full(q["EL"]), "w": ctx.zeros(), "g": ctx.zeros()}

    def step(self, st, I_syn, I_ext, ctx):
        q, dt = self.params, ctx.dt
        v, w, g = st["v"], st["w"], st["g"]
        g = g * float(np.exp(-dt / q["tau_syn"])) + I_syn
        # 스파이크 개시(지수항)는 가팔라서 역전파로 곱해지면 폭발 → 값만 쓰고 기울기는 끊음 (Signal(배열) = 상수)
        vv = ctx.xp.minimum(v.data, -30.0)
        expo = Signal(ctx.xp.exp((vv - q["VT"]) / q["dT"]) * q["dT"])
        dv = ((v - q["EL"]) * -1.0 + expo + (g - w) * (1.0 / q["gL"])) * (dt / q["tau_m"])
        v1 = v + dv + I_ext
        w1 = w + ((v - q["EL"]) * q["a"] - w) * (dt / q["tau_w"])
        spk = fire((v1 - 0.0) * (1.0 / 20.0), 0.0, q["slope"])
        fired = spk.data > 0
        v2 = where(fired, np.float32(q["Vr"]), v1)
        return {"v": v2, "w": w1 + spk * q["b"], "g": g}, spk


fd.ganglion.limit_gpu_memory(0.6)
mb = fd.Circuit.from_flywire()
enc = fd.GlomerularEncoder(mb)
door = fd.door_odors(enc.glomeruli)
P = door["X"][np.argsort(door["X"].sum(1))[::-1][:8]]
X = (P / P.max()).astype(np.float32)
kw = dict(t_ms=100, input_mode="regular", trainable=["KC>MBON"])

print("1) AdEx 버섯체 - 냄새 8개에 대한 MBON 반응")
layer = fd.ConnectomeLayer(mb, "PN", "MBON", neuron=AdEx(), gains={"PN>KC": 8.0, "KC>MBON": 3.0}, **kw)
with fd.quiescent():
    r = layer(enc(X), seed=0, return_all=True).numpy()
print(f"   PN {r[:, mb.groups['PN']].mean():.1f} Hz, KC {r[:, mb.groups['KC']].mean():.2f} Hz, "
      f"MBON {r[:, mb.groups['MBON']].mean():.1f} Hz")

print("2) 가상 유전학: APL(억제 뉴런)을 끄면 KC 활동이 늘어나는가")
with fd.genetics.silence(layer, fd.genetics.driver(mb, group="APL")), fd.quiescent():
    r2 = layer(enc(X), seed=0, return_all=True).numpy()
print(f"   KC {r[:, mb.groups['KC']].mean():.2f} → {r2[:, mb.groups['KC']].mean():.2f} Hz")

print("3) 역전파 (자동 미분) + explain으로 기울기가 믿을 만한지 실제로 꺼서 확인")
rep = fd.explain(lambda L, s: L(enc(X), seed=s).sum(), layer, by="cell_type", verify=4)
print(f"   예측(기울기)과 실제로 끈 결과의 순위 상관 {rep.agreement:+.2f} (surrogate_damp {layer.surrogate_damp:g})")
if rep.agreement is not None and rep.agreement >= 0.7:
    print("   → 기울기를 믿을 만함 (기본 surrogate_damp 0.1이 1000스텝 동안 기울기가 부푸는 것을 막음;")
    print("     감쇠 없이(surrogate_damp=1) 돌리면 순위 상관 -0.42로 방향까지 틀림)")
else:
    print("   → 기울기를 믿으면 안 됨: fd.tune_surrogate로 감쇠를 고르거나 fd.ThreeFactor, 해석은 실제로 끈 결과로")

print("4) 세포 유형별 실제 손상 효과 (확인한 유형)")
print("  ", ", ".join(f"{n} {v:.0f}" for n, v in zip(rep.verified.name, rep.verified.actual_drop)))

print("5) 저장·불러오기")
layer.save("adex_mb.npz")
again = fd.ConnectomeLayer.load("adex_mb.npz")
with fd.quiescent():
    same = np.array_equal(again(enc(X), seed=0).numpy(), layer(enc(X), seed=0).numpy())
print(f"   {type(again.neuron).__name__} 다시 만듦, 출력 같음: {same}")
Path("adex_mb.npz").unlink()

print("6) 속도·메모리: 내장 LIF (합친 연산) 대 플러그인 LIF (Signal 연산), 학습 1스텝, 배치 32, 체크포인팅 50스텝")
print("   (플러그인은 스텝 안의 중간값을 모두 저장하므로 긴 시뮬레이션에는 checkpoint_every가 사실상 필요)")
x = enc(np.repeat(X, 4, 0))
for name, neuron in [("내장 LIF", "lif"), ("플러그인 LIF", fd.neurons.LIF())]:
    fd.ganglion.backend.gpu_memory_peak_reset()
    L = fd.ConnectomeLayer(mb, "PN", "MBON", neuron=neuron, checkpoint_every=50, gains={"PN>KC": 3.0, "KC>MBON": 3.0},
                           **kw)
    ts = []
    for r_ in range(4):
        t = time.perf_counter()
        (L(x, seed=r_) * 0.01).sum().retrograde()
        L.clear_retro()
        float(L.log_scale.numpy().sum())
        ts.append(time.perf_counter() - t)
    print(f"   {name:<12} {np.median(ts[1:]) * 1000:.0f} ms, GPU 메모리 풀 {fd.ganglion.backend.gpu_memory_used() / 2**30:.2f} GB")
