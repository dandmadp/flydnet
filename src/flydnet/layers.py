"""Circuit 배선으로 스파이크를 전파하는 LIF 층 (배치 처리, GPU, 역전파 가능)

flybrain_local/fly_sim.py와 같은 뉴런 모델(Shiu et al. 2024)이지만
- trial 차원 대신 배치 차원: 샘플마다 다른 입력 발화율
- 시냅스 전파를 희소행렬 곱 한 번으로 처리
- trainable: 배선은 고정, 연결별 세기를 nn.Parameter로 학습 (대리 기울기로 역전파)
"""
import warnings

import numpy as np
import torch
import torch.nn as nn

from .circuit import Circuit

DEFAULT_PARAMS = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,   # mV
    t_mbr=20.0, tau=5.0,                  # ms
    t_rfc=2.2, t_dly=1.8,                 # ms
    w_syn=0.275,                          # mV / synapse
    f_poi=250,                            # Poisson 입력 1회 = w_syn*f_poi mV
    dt=0.1,                               # ms
)


class SpikeFn(torch.autograd.Function):
    """순전파: 문턱을 넘으면 1 (진짜 스파이크)
    역전파: 계단 함수 대신 빠른 시그모이드의 기울기 1 / (1 + slope·|x|)² (대리 기울기, SuperSpike)"""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + ctx.slope * x.abs()) ** 2, None


class SparsePropagate(torch.autograd.Function):
    """시냅스 전파 out = W @ spk (W는 연결 목록 values로 만든 CSR 희소행렬)

    torch.sparse.mm의 역전파는 연결 기울기를 뉴런 수 × 뉴런 수 크기로 만들어서 큰 회로에서 메모리가 터짐
    (전체 뇌 13.9만 뉴런 → 77 GB). 여기서는 필요한 연결 칸만 계산:
      d values[e] = Σ_b grad[post_e, b] · spk[pre_e, b]   (sampled_addmm, 메모리 = 연결 수)
      d spk       = Wᵀ @ grad                              (미리 정렬해 둔 전치 CSR)
    """

    @staticmethod
    def forward(ctx, values, spk, crow, col, crow_t, col_t, perm_t):
        N = len(crow) - 1
        ctx.save_for_backward(values, spk, crow, col, crow_t, col_t, perm_t)
        return _csr(crow, col, values, N) @ spk

    @staticmethod
    def backward(ctx, grad):
        values, spk, crow, col, crow_t, col_t, perm_t = ctx.saved_tensors
        N = len(crow) - 1
        d_values = d_spk = None
        if ctx.needs_input_grad[0]:
            pattern = _csr(crow, col, torch.zeros_like(values), N)
            d_values = torch.sparse.sampled_addmm(pattern, grad, spk.T, beta=0.0, alpha=1.0).values()
        if ctx.needs_input_grad[1]:
            d_spk = _csr(crow_t, col_t, values[perm_t], N) @ grad
        return d_values, d_spk, None, None, None, None, None


def _csr(crow, col, values, N):
    with warnings.catch_warnings():                              # "CSR은 베타" 경고 숨김
        warnings.simplefilter("ignore", UserWarning)
        return torch.sparse_csr_tensor(crow, col, values, (N, N), check_invariants=False)


class ConnectomeLayer(nn.Module):
    """입력 그룹 뉴런에 발화율(B, n_in) Hz를 넣고, 출력 그룹 뉴런의 발화율(B, n_out) Hz를 돌려줌

    gains:      {"PN>KC": 1.5, ...} 그룹 간 시냅스 세기 배율 (기본 1)
    input_mode: "poisson" = 무작위 스파이크 (생물학적, 같은 입력에도 매번 다른 반응)
                "regular" = 같은 발화율의 일정 간격 스파이크 (같은 입력 → 같은 반응)
                두 모드 모두 입력 발화율에 대해 미분 가능: 순전파는 진짜 스파이크, 역전파는 스파이크를
                '스텝당 발화 확률'로 보고 기울기를 넘김 (straight-through) → 앞쪽 층까지 학습 가능
    trainable:  False = 고정 층 / True = 모든 연결 세기 학습 / ["KC>MBON", ...] = 그 연결 종류만 학습.
                연결별 세기 = 원래 세기 × exp(log_scale). 부호(흥분/억제)는 바뀌지 않음 (Dale의 법칙)
    dt:         시간 간격 ms (기본 0.1). 학습할 때는 0.5 정도로 키우면 스텝 수·메모리가 1/5
    slope:      대리 기울기의 날카로움 (막전위를 문턱−휴지 간격으로 나눈 단위 기준)
    checkpoint_every: 역전파 메모리 절약 (그래디언트 체크포인팅). n이면 n스텝 구간마다 중간 상태를 버리고
                역전파 때 다시 계산 → 메모리는 대략 (전체 스텝/n + n)에 비례, 계산은 약 1.3~2배.
                학습할 때(기울기 필요)만 적용, 추론에는 영향 없음
    share:      "edge" = 연결마다 따로 학습 / "pair" = 같은 연결 종류(예: "Mi1>T4a")는 배율 하나를 공유
                (커넥톰이 정한 시냅스 수 비율은 그대로, 종류별 세기만 학습 → 매개변수가 훨씬 적음)

    뉴런 매개변수 (그룹 = 세포 유형마다 하나, 셋 중 하나라도 주면 켜짐):
    bias:       휴지 전위에 더할 값 mV (숫자 또는 {그룹: mV}). 문턱(v_th − v_0 = 7 mV)보다 크면 입력 없이도
                꾸준히 발화 → 억제·감소 신호도 전달 가능 (시각계처럼 평소에도 활동하는 회로)
    t_mbr:      막 시간 상수 ms (숫자 또는 {그룹: ms}). 세포 유형마다 반응 속도가 다르게
    train_neurons: True면 그룹별 bias와 막 시간 상수도 학습
    v_init:     "rest" = 모든 뉴런이 휴지 전위에서 시작 / "random" = 리셋~문턱 사이 고정된 무작위 값
                (꾸준히 발화하는 뉴런들이 처음에 한꺼번에 발화하지 않도록)
    count_from_ms: 이 시각부터의 스파이크만 셈 (처음의 과도 반응 제외)

    입력: rates (B, n_in) = 일정한 발화율, (B, T, n_in) = 시간에 따라 바뀌는 발화율 (T 프레임을 시간에 고르게)

    neuron:     "lif" = 스파이킹 뉴런 (기본) /
                "graded" = 스파이크 없이 연속값을 전달하는 뉴런 (라미나·메둘라처럼 막전위로 신호를 보내는 회로).
                  τ dV/dt = bias − V + Σ w·r_pre,  r = clamp(V, 0, r_max)   (V는 휴지 전위 기준, 단위 없음)
                  w = ±시냅스 수 × params["w_syn"] × 배율. 입력 뉴런은 r = 입력값 (rates를 활동값으로 씀),
                  출력은 count_from_ms 이후 평균 r. 시냅스 지연·불응기 없음, input_mode·v_init 무시
    """

    def __init__(self, circuit: Circuit, inputs="PN", outputs=("KC",), t_ms: float = 100.0,
                 params: dict | None = None, gains: dict | None = None, device: str | None = None,
                 input_mode: str = "poisson", trainable=False, dt: float | None = None, slope: float = 10.0,
                 checkpoint_every: int | None = None, share: str = "edge", bias=None, t_mbr=None,
                 train_neurons: bool = False, v_init: str = "rest", count_from_ms: float = 0.0,
                 neuron: str = "lif"):
        super().__init__()
        listify = lambda x: x if isinstance(x, str) else list(x)
        self._init_args = dict(inputs=listify(inputs), outputs=listify(outputs),
                               t_ms=t_ms, params=dict(params or {}), input_mode=input_mode,
                               trainable=trainable if isinstance(trainable, bool) else list(trainable),
                               dt=dt, slope=slope, checkpoint_every=checkpoint_every, share=share,
                               bias=dict(bias) if isinstance(bias, dict) else bias,
                               t_mbr=dict(t_mbr) if isinstance(t_mbr, dict) else t_mbr,
                               train_neurons=train_neurons, v_init=v_init, count_from_ms=count_from_ms,
                               neuron=neuron)
        if neuron not in ("lif", "graded"):
            raise ValueError(neuron)
        self.neuron = neuron
        self.circuit = circuit
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        if dt is not None:
            self.p["dt"] = dt
        self.t_ms, self.slope, self.checkpoint_every = t_ms, slope, checkpoint_every
        self.count_from_ms = count_from_ms
        if not 0 <= count_from_ms < t_ms:
            raise ValueError("count_from_ms는 0 이상 t_ms 미만")
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        outputs = [outputs] if isinstance(outputs, str) else list(outputs)
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        self.out_names, self.in_names = outputs, inputs
        self.register_buffer("in_idx", torch.tensor(np.concatenate([circuit.groups[i] for i in inputs]),
                                                    device=self.dev))
        self.register_buffer("out_idx", torch.tensor(np.concatenate([circuit.groups[o] for o in outputs]),
                                                     device=self.dev))
        self.n_in, self.n_out = len(self.in_idx), len(self.out_idx)
        self.gains = dict(gains or {})
        if input_mode not in ("poisson", "regular"):
            raise ValueError(input_mode)
        self.input_mode = input_mode
        # regular 모드: 입력 뉴런마다 고정된 시작 위상 (모두 동시에 발화하지 않도록)
        self.register_buffer("phase0", torch.rand(self.n_in, 1, generator=torch.Generator().manual_seed(0))
                             .to(self.dev))

        # 연결 목록 (post, pre) 정렬·중복 합치기 → 희소행렬 순서와 같게
        g = circuit.group_of()
        key = (circuit.post.astype(np.int64) * circuit.N + circuit.pre)
        uniq, inv = np.unique(key, return_inverse=True)
        w = np.zeros(len(uniq), np.float64); np.add.at(w, inv, circuit.weight)
        post, pre = uniq // circuit.N, uniq % circuit.N
        self.edge_key = g[pre] + ">" + g[post]                  # 연결 종류 (예: "KC>MBON")
        self.register_buffer("w_idx", torch.tensor(np.stack([post, pre]), device=self.dev))
        self.register_buffer("w_syn", torch.tensor(w, dtype=torch.float32, device=self.dev))   # ±시냅스 수
        # CSR 구조 (행 = 받는 뉴런)와 전치 CSR (행 = 보내는 뉴런). w_idx에서 다시 만들 수 있어 저장하지 않음
        N = circuit.N
        crow = np.zeros(N + 1, np.int64); crow[1:] = np.cumsum(np.bincount(post, minlength=N))
        perm_t = np.lexsort((post, pre))                         # (pre, post) 순 정렬
        crow_t = np.zeros(N + 1, np.int64); crow_t[1:] = np.cumsum(np.bincount(pre, minlength=N))
        for name, arr in (("crow", crow), ("col", pre), ("crow_t", crow_t), ("col_t", post[perm_t]),
                          ("perm_t", perm_t)):
            self.register_buffer(name, torch.tensor(arr, dtype=torch.long, device=self.dev), persistent=False)

        # 학습할 연결
        if trainable is True:
            pos = np.arange(len(uniq))
        elif trainable:
            unknown = set(trainable) - set(self.edge_key)
            if unknown:
                raise ValueError(f"회로에 없는 연결 종류: {sorted(unknown)}")
            pos = np.nonzero(np.isin(self.edge_key, list(trainable)))[0]
        else:
            pos = np.array([], dtype=np.int64)
        if share not in ("edge", "pair"):
            raise ValueError(share)
        self.share = share
        if share == "pair":                                       # 학습 연결 → 연결 종류 번호
            self.train_pairs, which = np.unique(self.edge_key[pos], return_inverse=True)
        else:
            which = np.arange(len(pos))
        self.register_buffer("train_pos", torch.tensor(pos, dtype=torch.long, device=self.dev))
        self.register_buffer("train_which", torch.tensor(which, dtype=torch.long, device=self.dev), persistent=False)
        n_scale = int(which.max()) + 1 if len(pos) else 0
        self.log_scale = nn.Parameter(torch.zeros(n_scale, device=self.dev)) if len(pos) else None

        # 뉴런 매개변수 (그룹마다)
        self.neuron_params = bias is not None or t_mbr is not None or train_neurons
        if v_init not in ("rest", "random"):
            raise ValueError(v_init)
        self.v_init = v_init
        self.register_buffer("v_frac", torch.rand(N, 1, generator=torch.Generator().manual_seed(1)).to(self.dev),
                             persistent=False)
        if self.neuron_params:
            self.group_names = list(circuit.groups)
            gi = np.full(N, len(self.group_names), np.int64)      # 어느 그룹에도 없는 뉴런 → 마지막 칸
            for k, name in enumerate(self.group_names):
                gi[circuit.groups[name]] = k
            self.register_buffer("group_idx", torch.tensor(gi, device=self.dev), persistent=False)
            per = lambda v, default: [float(v.get(n, default) if isinstance(v, dict) else
                                            (default if v is None else v)) for n in self.group_names + ["_"]]
            b0 = torch.tensor(per(bias, 0.0), device=self.dev)
            t0 = torch.tensor(per(t_mbr, self.p["t_mbr"]), device=self.dev).log()
            if train_neurons:
                self.bias, self.log_t_mbr = nn.Parameter(b0), nn.Parameter(t0)
            else:
                self.register_buffer("bias", b0); self.register_buffer("log_t_mbr", t0)
        self._build()

    @property
    def trainable(self) -> bool:
        return self.log_scale is not None

    def _gain_vec(self) -> torch.Tensor:
        s = np.ones(len(self.edge_key), np.float32)
        for k, v in self.gains.items():
            s[self.edge_key == k] *= v
        return torch.tensor(s, device=self.dev)

    def _build(self):
        """고정 부분 (원래 세기 × 배율 × mV/시냅스). 고정 층이면 빠른 CSR 행렬도 만듦"""
        self.w_base = self.w_syn * self._gain_vec() * self.p["w_syn"]
        if not self.trainable:
            self.W_T = _csr(self.crow, self.col, self.w_base, self.circuit.N)

    def set_gain(self, key: str, value: float):
        self.gains[key] = value
        self._build()

    def weights(self) -> torch.Tensor:
        """현재 연결별 세기 (mV/스파이크, 부호 포함). 순서는 self.w_idx (post, pre)"""
        if not self.trainable:
            return self.w_base
        scale = torch.ones_like(self.w_base).index_copy(0, self.train_pos, torch.exp(self.log_scale)[self.train_which])
        return self.w_base * scale

    def scale_of(self) -> dict:
        """share="pair"일 때 연결 종류별 학습된 배율 {"Mi1>T4a": 1.3, ...}"""
        if self.share != "pair" or not self.trainable:
            raise ValueError("share='pair'로 학습하는 층에서만")
        return dict(zip(self.train_pairs, torch.exp(self.log_scale).tolist()))

    def neuron_table(self):
        """그룹별 bias(mV)와 막 시간 상수(ms)"""
        import pandas as pd
        if not self.neuron_params:
            raise ValueError("뉴런 매개변수(bias/t_mbr/train_neurons)를 켠 층에서만")
        return pd.DataFrame({"bias_mV": self.bias.detach().cpu()[:-1].numpy(),
                             "t_mbr_ms": self.log_t_mbr.detach().exp().cpu()[:-1].numpy()}, index=self.group_names)

    def _propagate(self, spk: torch.Tensor, values: torch.Tensor | None) -> torch.Tensor:
        if values is None:                                       # 고정 층 (기울기 필요 없으면 빠른 길)
            return self.W_T @ spk if not spk.requires_grad else                 SparsePropagate.apply(self.w_base, spk, self.crow, self.col, self.crow_t, self.col_t, self.perm_t)
        return SparsePropagate.apply(values, spk, self.crow, self.col, self.crow_t, self.col_t, self.perm_t)

    def forward(self, rates: torch.Tensor, seed: int | None = None, return_all: bool = False, record=None):
        """record: 뉴런 번호 목록(회로 번호)이면 그 뉴런들의 스텝별 스파이크도 돌려줌 → (rate, (B, steps, k))
        분석용 (체크포인팅과 같이 쓰지 않음)"""
        if self.neuron == "graded":
            return self._forward_graded(rates, return_all, record)
        p, N, dev = self.p, self.circuit.N, self.dev
        rates = rates.to(dev)
        B = rates.shape[0]
        g = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        dt = p["dt"]; steps = int(round(self.t_ms / dt))
        dly = max(int(round(p["t_dly"] / dt)), 1); R = dly + 1
        rfc = int(round(p["t_rfc"] / dt))
        gd = float(np.exp(-dt / p["tau"]))
        if self.neuron_params:                                   # 그룹별 휴지 전위·반응 속도 (N, 1)
            v_eq = p["v_0"] + self.bias[self.group_idx][:, None]
            a = (dt / self.log_t_mbr.exp()[self.group_idx][:, None]).clamp(max=1.0)
        else:
            v_eq, a = p["v_0"], dt / p["t_mbr"]
        poi_w = p["w_syn"] * p["f_poi"]
        if rates.dim() == 2:
            p_all = (rates * dt / 1000.0).T[None]               # (1, n_in, B) 스텝당 입력 스파이크 확률
        elif rates.dim() == 3:
            p_all = (rates * dt / 1000.0).permute(1, 2, 0)      # (T, n_in, B)
        else:
            raise ValueError("rates는 (B, n_in) 또는 (B, T, n_in)")
        T = p_all.shape[0]
        s_cnt = int(round(self.count_from_ms / dt))
        rfc_vec = torch.full((N, 1), float(rfc), device=dev); rfc_vec[self.in_idx] = 0
        scale = p["v_th"] - p["v_rst"]
        values = self.weights() if self.trainable else None      # 연결 세기는 순전파당 한 번만 계산

        def run(s0, s1, gen_state, V, G, refr, counts, phase, *buf):
            """s0 ~ s1 스텝 진행. 체크포인팅 때 역전파에서 다시 불리므로 같은 입력 → 같은 결과여야 함"""
            if g is not None and gen_state is not None:
                g.set_state(gen_state)                          # 다시 계산할 때도 같은 난수
            buf = list(buf)                                      # 지연 중인 시냅스 입력 (원형 버퍼)
            for s in range(s0, s1):
                p_spk = p_all[s * T // steps]
                G = G + buf[s % R]
                act = refr <= 0
                V = torch.where(act, V + (v_eq - V + G) * a, V)
                G = torch.where(act, G * gd, G)
                if self.input_mode == "poisson":
                    inp = (torch.rand(p_spk.shape, device=dev, generator=g) < p_spk).float()
                else:
                    phase = phase + p_spk.detach(); inp = (phase >= 1).float(); phase = phase - inp
                inp = inp + (p_spk - p_spk.detach())             # 값은 스파이크 그대로, 기울기는 확률로
                V = V.index_add(0, self.in_idx, inp * poi_w)
                spk = SpikeFn.apply((V - p["v_th"]) / scale, self.slope)
                fired = spk.detach() > 0
                if s >= s_cnt:
                    counts = counts + spk
                if rec is not None:
                    rec.append(spk.detach()[rec_idx])
                V = torch.where(fired, p["v_rst"], V)            # 리셋은 기울기 끊음 (표준)
                G = torch.where(fired, 0.0, G)
                refr = torch.where(fired, rfc_vec, refr - 1)
                buf[(s + dly) % R] = self._propagate(spk, values)  # dly 스텝 뒤 G에 도착 (이 칸은 방금 전 스텝에 읽음)
            return (V, G, refr, counts, phase, *buf)

        zeros = lambda: torch.zeros((N, B), device=dev)
        if self.v_init == "random":
            V0 = (p["v_rst"] + self.v_frac * (p["v_th"] - p["v_rst"])).expand(-1, B).clone()
        else:
            V0 = torch.full((N, B), p["v_0"], device=dev)
        state = (V0, zeros(), zeros(), zeros(), self.phase0.expand(-1, B).clone(), *[zeros() for _ in range(R)])
        rec = rec_idx = None
        ce = self.checkpoint_every
        needs_grad = self.trainable or rates.requires_grad or (self.neuron_params and self.bias.requires_grad)
        if ce and torch.is_grad_enabled() and needs_grad:
            if record is not None:
                raise ValueError("record는 체크포인팅과 같이 쓸 수 없음 (torch.no_grad()에서 쓰기)")
            for s0 in range(0, steps, ce):                      # 구간마다 중간 상태를 버리고 역전파 때 다시 계산
                state = torch.utils.checkpoint.checkpoint(
                    run, s0, min(s0 + ce, steps), g.get_state() if g is not None else None, *state,
                    use_reentrant=False)
        else:
            rec = [] if record is not None else None
            if rec is not None:
                rec_idx = torch.as_tensor(np.asarray(record), dtype=torch.long, device=dev)
            state = run(0, steps, None, *state)
        rate = state[3] / ((self.t_ms - s_cnt * dt) / 1000.0)    # (N, B) Hz
        out = rate.T if return_all else rate[self.out_idx].T
        if record is not None:
            return out, torch.stack(rec).permute(2, 0, 1)
        return out

    def _forward_graded(self, x: torch.Tensor, return_all: bool, record):
        """연속값 뉴런 (neuron="graded"). 오일러 적분, 입력 뉴런은 활동값을 그대로 고정"""
        p, N, dev = self.p, self.circuit.N, self.dev
        x = x.to(dev)
        B = x.shape[0]
        dt = p["dt"]; steps = int(round(self.t_ms / dt))
        r_max = p.get("r_max", 10.0)
        if x.dim() == 2:
            x_all = x.T[None]                                    # (1, n_in, B)
        elif x.dim() == 3:
            x_all = x.permute(1, 2, 0)                           # (T, n_in, B)
        else:
            raise ValueError("입력은 (B, n_in) 또는 (B, T, n_in)")
        T = x_all.shape[0]
        s_cnt = int(round(self.count_from_ms / dt))
        if self.neuron_params:
            b = self.bias[self.group_idx][:, None]
            a = (dt / self.log_t_mbr.exp()[self.group_idx][:, None]).clamp(max=1.0)
        else:
            b, a = 0.0, dt / p["t_mbr"]
        values = self.weights() if self.trainable else None
        is_in = torch.zeros(N, 1, dtype=torch.bool, device=dev); is_in[self.in_idx] = True

        def run(s0, s1, V, r, acc):
            for s in range(s0, s1):
                I = self._propagate(r, values)
                V = V + (b - V + I) * a
                r = V.clamp(0.0, r_max)
                r = r.index_copy(0, self.in_idx, x_all[s * T // steps])   # 입력 뉴런은 입력값 그대로
                if s >= s_cnt:
                    acc = acc + r
                if rec is not None:
                    rec.append(r.detach()[rec_idx])
            return V, r, acc

        zeros = torch.zeros((N, B), device=dev)
        state = (zeros, zeros, zeros)
        rec = rec_idx = None
        ce = self.checkpoint_every
        needs_grad = self.trainable or x.requires_grad or (self.neuron_params and self.bias.requires_grad)
        if ce and torch.is_grad_enabled() and needs_grad:
            if record is not None:
                raise ValueError("record는 체크포인팅과 같이 쓸 수 없음 (torch.no_grad()에서 쓰기)")
            for s0 in range(0, steps, ce):
                state = torch.utils.checkpoint.checkpoint(run, s0, min(s0 + ce, steps), *state, use_reentrant=False)
        else:
            if record is not None:
                rec = []
                rec_idx = torch.as_tensor(np.asarray(record), dtype=torch.long, device=dev)
            state = run(0, steps, *state)
        mean = state[2] / (steps - s_cnt)                        # (N, B) 평균 활동
        out = mean.T if return_all else mean[self.out_idx].T
        if record is not None:
            return out, torch.stack(rec).permute(2, 0, 1)
        return out

    # ─────────────── 저장 / 불러오기 ───────────────
    FORMAT = "flydnet.ConnectomeLayer/1"

    def save(self, path):
        """회로 배선 + 설정 + 학습한 연결 세기를 파일 하나에. FlyWire 데이터 없이도 load()로 다시 만들 수 있음"""
        from . import __version__
        torch.save(dict(format=self.FORMAT, version=__version__, circuit=self.circuit.to_dict(),
                        config=dict(self._init_args, gains=dict(self.gains)),
                        state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()}), path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "ConnectomeLayer":
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d.get("format") != cls.FORMAT:
            raise ValueError(f"flydnet ConnectomeLayer 파일이 아님 (format={d.get('format')})")
        layer = cls(Circuit.from_dict(d["circuit"]), device=device, **d["config"])
        layer.load_state_dict(d["state_dict"])
        layer._build()
        return layer

    def extra_repr(self):
        tr = f", 학습 연결 {len(self.train_pos):,}개" if self.trainable else ""
        if self.neuron == "graded":
            tr = ", 연속값 뉴런" + tr
        if self.trainable and self.share == "pair":
            tr += f" (종류 {len(self.log_scale):,}개가 배율 공유)"
        if self.neuron_params:
            tr += f", 뉴런 매개변수 그룹 {len(self.group_names)}개{' 학습' if self.bias.requires_grad else ''}"
        return (f"{self.circuit.name}: in {self.n_in} ({'+'.join(self.in_names)}) → out {self.n_out} "
                f"({'+'.join(self.out_names)}), {self.t_ms} ms, dt {self.p['dt']} ms, 입력 {self.input_mode}{tr}")
