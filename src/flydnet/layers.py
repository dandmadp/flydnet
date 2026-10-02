"""Circuit 배선으로 스파이크를 전파하는 LIF 층 (배치 처리, GPU)

flybrain_local/fly_sim.py와 같은 뉴런 모델(Shiu et al. 2024)이지만
- trial 차원 대신 배치 차원: 샘플마다 다른 입력 발화율
- 시냅스 전파를 희소행렬 곱 한 번으로 처리
"""
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


class ConnectomeLayer(nn.Module):
    """입력 그룹 뉴런에 발화율(B, n_in)을 넣고, 출력 그룹 뉴런의 발화율(B, n_out)을 돌려줌

    gains: {"PN>KC": 1.5, ...} 처럼 그룹 간 시냅스 세기 배율 (기본 1)
    input_mode: "poisson" = 무작위 스파이크 (생물학적, 같은 입력에도 매번 다른 반응)
                "regular" = 같은 발화율의 일정 간격 스파이크 (같은 입력 → 같은 반응, 학습용)
    """

    def __init__(self, circuit: Circuit, inputs: str = "PN", outputs=("KC",), t_ms: float = 100.0,
                 params: dict | None = None, gains: dict | None = None, device: str | None = None,
                 input_mode: str = "poisson"):
        super().__init__()
        self.circuit = circuit
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.t_ms = t_ms
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        outputs = [outputs] if isinstance(outputs, str) else list(outputs)
        self.in_idx = torch.tensor(circuit.groups[inputs], device=self.dev)
        self.out_names = outputs
        self.out_idx = torch.tensor(np.concatenate([circuit.groups[o] for o in outputs]), device=self.dev)
        self.n_in, self.n_out = len(self.in_idx), len(self.out_idx)
        self.gains = dict(gains or {})
        if input_mode not in ("poisson", "regular"):
            raise ValueError(input_mode)
        self.input_mode = input_mode
        # regular 모드: 입력 뉴런마다 고정된 시작 위상 (모두 동시에 발화하지 않도록)
        self.phase0 = torch.rand(self.n_in, 1, generator=torch.Generator().manual_seed(0)).to(self.dev)
        self._build()

    def _build(self):
        c = self.circuit
        w = c.weight * self.p["w_syn"]
        if self.gains:
            g = c.group_of()
            key = g[c.pre] + ">" + g[c.post]
            for k, s in self.gains.items():
                w = np.where(key == k, w * s, w)
        # W_T[post, pre]: 스파이크 벡터(N, B)에 곱하면 뉴런별 시냅스 입력
        W = torch.sparse_coo_tensor(np.stack([c.post, c.pre]), w.astype(np.float32), (c.N, c.N)).coalesce()
        self.W_T = W.to_sparse_csr().to(self.dev)

    def set_gain(self, key: str, value: float):
        self.gains[key] = value
        self._build()

    @torch.no_grad()
    def forward(self, rates: torch.Tensor, seed: int | None = None, return_all: bool = False):
        p, N, dev = self.p, self.circuit.N, self.dev
        rates = rates.to(dev)
        B = rates.shape[0]
        g = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        dt = p["dt"]; steps = int(self.t_ms / dt)
        dly = int(round(p["t_dly"] / dt)); R = dly + 1
        rfc = int(round(p["t_rfc"] / dt))
        a = dt / p["t_mbr"]; gd = float(np.exp(-dt / p["tau"]))
        poi_w = p["w_syn"] * p["f_poi"]
        p_spk = (rates * dt / 1000.0).T                         # (n_in, B)

        V = torch.full((N, B), p["v_0"], device=dev)
        G = torch.zeros((N, B), device=dev)
        refr = torch.zeros((N, B), device=dev)
        rfc_vec = torch.full((N, 1), float(rfc), device=dev); rfc_vec[self.in_idx] = 0
        buf = torch.zeros((R, N, B), device=dev)
        counts = torch.zeros((N, B), device=dev)
        phase = self.phase0.expand(-1, B).clone()

        for s in range(steps):
            k = s % R
            G += buf[k]
            act = refr <= 0
            V = torch.where(act, V + (p["v_0"] - V + G) * a, V)
            G = torch.where(act, G * gd, G)
            if self.input_mode == "poisson":
                inp = torch.rand(p_spk.shape, device=dev, generator=g) < p_spk
            else:
                phase += p_spk; inp = phase >= 1; phase -= inp.float()
            V[self.in_idx] += inp * poi_w
            spk = V > p["v_th"]
            counts += spk
            V = torch.where(spk, p["v_rst"], V)
            G = torch.where(spk, 0.0, G)
            refr = torch.where(spk, rfc_vec, refr - 1)
            buf[(s + dly) % R] = self.W_T @ spk.float()         # dly 스텝 뒤 G에 도착 (이 칸은 직전 스텝에 이미 읽음)
        rate = counts / (self.t_ms / 1000.0)                     # (N, B) Hz
        return rate.T if return_all else rate[self.out_idx].T

    def extra_repr(self):
        return f"{self.circuit.name}: in {self.n_in} → out {self.n_out} ({'+'.join(self.out_names)}), {self.t_ms} ms"
