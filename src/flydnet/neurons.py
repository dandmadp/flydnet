"""사용자 정의 뉴런 모델: 한 스텝만 정의하면 ConnectomeLayer의 시뮬레이션·역전파·체크포인팅·genetics·explain이 따라옴

  @fd.neurons.register
  class MyNeuron(fd.neurons.NeuronModel):
      state = ("v",)                                         # 상태 변수 이름 (뉴런 x 배치 배열 하나씩)
      def init(self, ctx):                                   # 처음 상태
          return {"v": ctx.full(-65.0)}
      def step(self, st, I_syn, I_ext, ctx):                 # 한 스텝: 새 상태, 스파이크 (0/1 Signal)
          v = st["v"] + ctx.dt * (-(st["v"] + 65.0) / 10.0) + I_syn + I_ext
          spk = fd.ganglion.fire(v, -50.0, slope=1.0)          # 대리 기울기가 있는 발화
          v = fd.ganglion.where(spk.data > 0, -65.0, v)        # 리셋
          return {"v": v}, spk

  layer = fd.ConnectomeLayer(circuit, "in", "out", neuron=MyNeuron())

step 안에서는 Signal 연산(+, *, exp, where, fire ...)만 쓰면 역전파가 자동 (자체 엔진의 자동 미분).
주의
  - 메모리: 스텝 안의 중간값을 모두 저장하므로 학습할 때는 ConnectomeLayer(checkpoint_every=50 등)를 쓸 것
    (내장 LIF는 필요한 값만 남기는 합친 연산이라 덜 듦). 속도는 내장 LIF의 약 1.4배 (체크포인팅 포함)
  - 기울기 폭발: 지수 함수처럼 가파른 항(AdEx의 스파이크 개시 등)은 시간을 거슬러 곱해지며 폭발함 →
    Signal(배열)로 값만 쓰고 기울기는 끊을 것. 발화는 fire(대리 기울기), 리셋은 where(fired, 상수, v)로 기울기를 끊음
  I_syn: 이번 스텝에 도착한 시냅스 입력 (연결 세기 x w_syn mV, 지연 t_dly 뒤) - LIF라면 전류 G에 더할 양
  I_ext: 입력 뉴런·활성화(fd.genetics.activate) 뉴런의 외부 자극 (스파이크 하나 = w_syn x f_poi mV, 다른 뉴런은 0)
  ctx:   dt (ms), N, batch, inputs (외부 자극을 받는 뉴런 번호), p (층의 매개변수 dict), full(값), zeros(),
         cache (이번 실행에만 쓰는 dict - init에서 미리 계산한 상수는 self가 아니라 여기에 둘 것: 같은 모델 객체를
         여러 층이 함께 쓰거나, 체크포인팅이 역전파 중에 step을 다시 부를 때 다른 실행의 값이 섞이지 않게)
저장: register한 모델은 layer.save / ConnectomeLayer.load로 다시 만들어짐 (매개변수는 __init__ 인자 그대로 저장)

내장 모델
  LIF(...)          Shiu et al. 2024 LIF (timing="brian"과 같은 계산을 Signal 연산으로 - 플러그인 구조의 기준·검증용)
  Izhikevich(...)   Izhikevich 2003 (a, b, c, d로 regular spiking·bursting·fast spiking 등), 시냅스 전류는 지수 감쇠
"""
from __future__ import annotations

from . import _check as _C
import numpy as np

from .ganglion.physiology import fire
from .ganglion.signal import Signal, where

REGISTRY: dict = {}


def register(cls):
    """뉴런 모델 클래스를 등록 (저장 파일에서 이름으로 다시 만들 수 있게). 데코레이터로"""
    if not (isinstance(cls, type) and issubclass(cls, NeuronModel)):
        raise TypeError("NeuronModel을 상속한 클래스만 등록")
    REGISTRY[cls.__name__] = cls
    return cls


class Context:
    """step·init에 주는 정보"""

    def __init__(self, xp, N, batch, dt, inputs, p):
        self.xp, self.N, self.batch, self.dt, self.inputs, self.p = xp, N, batch, dt, inputs, p
        self.cache = {}                                              # 이번 실행 전용 (init에서 계산한 상수 등)

    def full(self, value, dtype=None) -> Signal:
        return Signal(self.xp.full((self.N, self.batch), value, dtype=dtype or self.xp.float32))

    def zeros(self) -> Signal:
        return self.full(0.0)


class NeuronModel:
    """뉴런 모델의 기본 클래스. state(이름 목록), init(ctx), step(st, I_syn, I_ext, ctx)를 정의"""
    state: tuple = ()

    def __init__(self, **params):
        self.params = params

    def init(self, ctx: Context) -> dict:
        raise NotImplementedError

    def step(self, st: dict, I_syn: Signal, I_ext: Signal, ctx: Context):
        raise NotImplementedError

    def config(self) -> dict:
        name = type(self).__name__
        if REGISTRY.get(name) is not type(self):
            raise ValueError(f"{name}이 등록되지 않음 - 저장하려면 @fd.neurons.register")
        return {"model": name, "params": dict(self.params)}

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


def from_config(cfg: dict) -> NeuronModel:
    if cfg["model"] not in REGISTRY:
        raise KeyError(f"등록되지 않은 뉴런 모델: {cfg['model']} - 저장할 때 쓴 클래스를 import하고 @fd.neurons.register")
    return REGISTRY[cfg["model"]](**cfg["params"])


@register
class LIF(NeuronModel):
    """Shiu et al. 2024 LIF를 플러그인으로 (ConnectomeLayer 기본 계산 timing="brian"과 같은 스파이크).
    빠른 내장판(neuron="lif")이 있으므로 실제로는 기준·검증·변형의 출발점으로 씀"""
    state = ("V", "G", "refr")

    def __init__(self, slope: float = 10.0):
        _C.pos('slope', slope)
        super().__init__(slope=slope)

    def init(self, ctx):
        p = ctx.p
        rfc = np.full((ctx.N, 1), float(int(round(p["t_rfc"] / ctx.dt))), np.float32)
        rfc[np.asarray(_cpu(ctx.inputs))] = 0                        # 입력·활성화 뉴런은 불응기 없음 (Shiu et al.)
        tm, tau, dt = p["t_mbr"], p["tau"], ctx.dt
        ev, gd = float(np.exp(-dt / tm)), float(np.exp(-dt / tau))
        eg = float(dt / tm * np.exp(-dt / tm)) if abs(tm - tau) < 1e-9 else float(tau / (tau - tm) * (gd - ev))
        ctx.cache["lif"] = (ctx.xp.asarray(rfc - 1), ev, gd, eg)          # self에 두면 층끼리 섞임 (Context 설명)
        return {"V": ctx.full(p["v_0"]), "G": ctx.zeros(), "refr": ctx.zeros()}

    def step(self, st, I_syn, I_ext, ctx):
        p, xp = ctx.p, ctx.xp
        V, G, refr = st["V"], st["G"], st["refr"]
        rfc_set, ev, gd, eg = ctx.cache["lif"]
        act = refr.data <= 0
        V1 = where(act, (V - p["v_0"]) * ev + G * eg + p["v_0"], V)
        G1 = where(act, G * gd, G)
        spk = fire((V1 - p["v_th"]) * (1.0 / (p["v_th"] - p["v_rst"])), 0.0, self.params["slope"])
        G2 = where(act, G1 + I_syn, G1)                               # 불응기 중 도착한 입력은 버림
        V2 = V1 + I_ext
        fired = spk.data > 0
        V3 = where(fired, np.float32(p["v_rst"]), V2)
        G3 = where(fired, np.float32(0), G2)
        refr = Signal(xp.where(fired, rfc_set, refr.data - 1))
        return {"V": V3, "G": G3, "refr": refr}, spk


@register
class Izhikevich(NeuronModel):
    """Izhikevich 2003: v' = 0.04 v^2 + 5 v + 140 - u + g,  u' = a (b v - u),  v >= 30 mV면 v = c, u += d.
    시냅스 입력 g는 시간 상수 tau_syn ms로 감쇠 (도착한 입력 x gain을 더함). 외부 자극은 v에 바로 더함.
    regular spiking: a=0.02 b=0.2 c=-65 d=8 / fast spiking: a=0.1 b=0.2 c=-65 d=2 / bursting: a=0.02 b=0.2 c=-50 d=2"""
    state = ("v", "u", "g")

    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = -65.0, d: float = 8.0, tau_syn: float = 5.0,
                 gain: float = 1.0, slope: float = 1.0):
        _C.pos('tau_syn', tau_syn)
        _C.pos('slope', slope)
        _C.finite('gain', gain)
        super().__init__(a=a, b=b, c=c, d=d, tau_syn=tau_syn, gain=gain, slope=slope)

    def init(self, ctx):
        q = self.params
        return {"v": ctx.full(q["c"]), "u": ctx.full(q["b"] * q["c"]), "g": ctx.zeros()}

    def step(self, st, I_syn, I_ext, ctx):
        q, dt = self.params, ctx.dt
        v, u, g = st["v"], st["u"], st["g"]
        g = g * float(np.exp(-dt / q["tau_syn"])) + I_syn * q["gain"]
        # 0.04 v^2 + 5 v: 값은 그대로, 역전파 기울기는 0.08 v + 5를 0 이하로 자름. 문턱 근처에서 양수라
        # 스텝마다 1보다 큰 배율이 곱해져 기울기가 지수적으로 폭발함 (30 ms에 1e10, 방향도 틀림)
        vd = v.data
        slope = v.xp.minimum(vd * 0.08 + 5.0, 0).astype(vd.dtype)
        quad = v * slope + Signal((vd * vd * 0.04 + vd * 5.0 - vd * slope).astype(vd.dtype))
        v1 = v + (quad + 140.0 - u + g) * dt + I_ext
        u1 = u + (v * q["b"] - u) * (q["a"] * dt)
        spk = fire((v1 - 30.0) * (1.0 / 30.0), 0.0, q["slope"])
        fired = spk.data > 0
        v2 = where(fired, np.float32(q["c"]), where(v1.data > 30.0, np.float32(30.0), v1))   # 발산 막기
        u2 = u1 + spk * q["d"]
        return {"v": v2, "u": u2, "g": g}, spk


def _cpu(a):
    from .ganglion import backend as B
    return B.numpy(a)
