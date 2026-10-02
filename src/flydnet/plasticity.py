"""역전파 없는 도파민 학습 리드아웃 — 버섯체 KC→MBON 시냅스 규칙

    w[c, k] ← w[c, k] · (1 − lr · a_k · DA_c)        (ltd, 처벌 도파민: 시냅스 약화)
    w[c, k] ← w[c, k] + lr · a_k · DA_c · (w_max − w) (ltp, 보상 도파민: 시냅스 강화)
    w[c, k] ← max(0, w[c, k] + lr · a_k · DA_c)        (bidir, DA_c = +1 정답 / −1 이긴 오답, 틀렸을 때만)
    w[c, k] ← w[c, k] + DA_c · (a_k − w[c, k]) / n_c   (assoc, DA_c = 1 정답만. 정답 출력이 그 클래스 평균 패턴이 됨)

a_k  = KC k의 활동 (샘플 안에서 최대 발화율로 정규화, 0~1)
DA_c = 출력 c를 담당하는 도파민 뉴런의 신호 (정답/오답에서 결정)
출력 c의 점수 = Σ_k w[c, k] · a_k  (가장 큰 출력 = 예측)

시냅스 하나의 변화는 그 시냅스 앞 KC 활동과 그 출력의 도파민만으로 정해짐 (국소 규칙)

homeostasis=True: 출력 뉴런마다 시냅스 총량을 일정하게 맞춰 점수 계산 (synaptic scaling, 실험적).
  ltd/ltd_err/ltp의 출력 쏠림을 막으려 넣었으나 MNIST에서는 효과 없었음
"""
import torch


class DopamineReadout:
    MODES = ("bidir", "assoc", "ltd", "ltd_err", "ltp")

    def __init__(self, n_in: int, n_classes: int, mode: str = "bidir", lr: float = 0.01,
                 binary: bool = False, homeostasis: bool = False, device: str | None = None):
        """mode
        bidir   : 틀렸을 때만, 정답 출력은 강화 + 이긴 오답 출력은 약화 (양방향 가소성, 권장)
                  초파리 DA 가소성은 실제로 양방향 (Berry et al. 2018, Handler et al. 2019)
        assoc   : 정답 출력만 강화 (보상 연합 학습) + 출력별 시냅스 총량 정규화(코사인 점수).
                  다른 출력을 건드리지 않아 학습 순서와 무관 → 연속 학습에서 망각 없음
                  (Shen, Dasgupta & Navlakha 2023의 버섯체 영감 연속 학습과 같은 발상)
        ltd     : 정답이 아닌 모든 출력에 처벌 도파민 (항상)
        ltd_err : 정답보다 점수가 높거나 같았던 틀린 출력에만 처벌 도파민 (틀렸을 때만)
        ltp     : 정답 출력에 보상 도파민 (항상)
        binary  : True면 KC 활동을 켜짐/꺼짐(0/1)으로
        """
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode, self.lr, self.binary, self.homeostasis = mode, lr, binary, homeostasis
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        init = {"ltp": 0.0, "assoc": 0.0, "bidir": 0.5}.get(mode, 1.0)
        self.W = torch.full((n_classes, n_in), init, device=self.dev)
        self.n_seen = torch.zeros(n_classes, device=self.dev)      # assoc: 출력별 받은 도파민 횟수

    def activity(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.dev).float()
        if self.binary:
            return (X > 0).float()
        return X / X.amax(1, keepdim=True).clamp_min(1e-8)

    def scores(self, X: torch.Tensor) -> torch.Tensor:
        W = self.W
        if self.mode == "assoc":                         # 출력별 시냅스 벡터 크기를 같게 (L2)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        elif self.homeostasis:                             # 출력별 총 시냅스 세기를 같게 (평균 1)
            W = W / W.mean(1, keepdim=True).clamp_min(1e-8)
        return self.activity(X) @ W.T

    def predict(self, X: torch.Tensor, classes=None) -> torch.Tensor:
        s = self.scores(X)
        if classes is not None:                          # 지금까지 본 클래스만 후보로
            mask = torch.full((self.n_classes,), float("-inf"), device=self.dev)
            mask[list(classes)] = 0
            s = s + mask
        return s.argmax(1)

    def dopamine(self, X: torch.Tensor, y: torch.Tensor, classes=None) -> torch.Tensor:
        """(B, C) 도파민 신호"""
        y = y.to(self.dev)
        onehot = torch.nn.functional.one_hot(y, self.n_classes).float()
        if self.mode == "bidir":
            win = self.predict(X, classes)
            wrong = (win != y).float()[:, None]
            return wrong * (onehot - torch.nn.functional.one_hot(win, self.n_classes).float())
        if self.mode in ("ltp", "assoc"):
            return onehot
        if self.mode == "ltd":
            return 1 - onehot
        s = self.scores(X)
        return ((s >= s.gather(1, y[:, None])) & (onehot == 0)).float()

    @torch.no_grad()
    def step(self, X: torch.Tensor, y: torch.Tensor, classes=None):
        """샘플 묶음 하나로 시냅스 갱신 (묶음 안 변화는 평균). classes: 경쟁할 출력 (bidir)"""
        a = self.activity(X)
        if self.mode == "assoc":                         # 누적 평균: 순서가 바뀌어도 결과 같음
            da = self.dopamine(X, y, classes)
            self.n_seen += da.sum(0)
            self.W += (da.T @ a - da.sum(0)[:, None] * self.W) / self.n_seen.clamp_min(1)[:, None]
            return
        drive = self.dopamine(X, y, classes).T @ a / len(a)   # (C, K): 출력별 '도파민 × KC 활동'
        if self.mode == "bidir":
            self.W = (self.W + self.lr * drive).clamp_min(0)  # 시냅스는 음수가 될 수 없음
        elif self.mode == "ltp":
            self.W += self.lr * drive * (1 - self.W)
        else:
            self.W *= (1 - self.lr * drive).clamp_min(0)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 1, batch: int = 32, seed: int = 0, classes=None):
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), batch):
                j = perm[i:i + batch]
                self.step(X[j], y[j], classes)
        return self

    def accuracy(self, X, y, classes=None) -> float:
        return (self.predict(X, classes).cpu() == y.cpu()).float().mean().item()


class AssocReadout:
    """보상 연합 학습 리드아웃, 클래스마다 출력(원형) 여러 개 — 연속 학습용

    클래스 c마다 출력 뉴런 per_class개 (MBON 여러 개가 같은 도파민 구역을 공유하는 것처럼).
    샘플 (a, y)가 오면 클래스 y의 출력 중 가장 잘 맞는 하나에만 보상 도파민 →
        w ← w + (a − w) / n        (그 출력이 받은 샘플들의 평균 패턴이 됨)
    비어 있는 출력이 있으면 그것부터 채움 (클래스당 온라인 k-평균과 같음).
    점수 = 코사인 (출력별 시냅스 벡터 크기를 같게), 클래스 점수 = 그 클래스 출력 중 최대.

    다른 클래스의 시냅스는 절대 바뀌지 않음 → 클래스를 차례로 배워도 앞의 것을 잊지 않음.
    per_class=1이면 DopamineReadout(mode="assoc")와 같음.
    """

    def __init__(self, n_in: int, n_classes: int, per_class: int = 1, binary: bool = False,
                 device: str | None = None):
        self.n_classes, self.k, self.binary = n_classes, per_class, binary
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.W = torch.zeros(n_classes * per_class, n_in, device=self.dev)
        self.count = torch.zeros(n_classes * per_class, device=self.dev)

    activity = DopamineReadout.activity

    def _proto_scores(self, a: torch.Tensor) -> torch.Tensor:
        """(B, C*k) 코사인 점수, 빈 출력은 −inf"""
        s = a @ (self.W / self.W.norm(dim=1, keepdim=True).clamp_min(1e-8)).T
        return s.masked_fill(self.count == 0, float("-inf"))

    def scores(self, X: torch.Tensor) -> torch.Tensor:
        """(B, C) 클래스 점수"""
        return self._proto_scores(self.activity(X)).view(len(X), self.n_classes, self.k).amax(2)

    predict = DopamineReadout.predict
    accuracy = DopamineReadout.accuracy

    @torch.no_grad()
    def step(self, X: torch.Tensor, y: torch.Tensor, classes=None):
        a = self.activity(X)
        y = y.to(self.dev)
        own = self._proto_scores(a).view(len(a), self.n_classes, self.k)[torch.arange(len(a)), y]   # (B, k)
        empty = (self.count.view(self.n_classes, self.k) == 0)[y]                                    # (B, k)
        first_empty = empty & (empty.cumsum(1) == 1)            # 클래스의 첫 빈 출력부터 채움
        own = torch.where(first_empty, torch.full_like(own, float("inf")), own)
        idx = y * self.k + own.argmax(1)                         # 보상 도파민을 받을 출력
        n = torch.zeros_like(self.count).index_add_(0, idx, torch.ones(len(a), device=self.dev))
        s = torch.zeros_like(self.W).index_add_(0, idx, a)
        self.count += n
        hit = n > 0
        self.W[hit] += (s[hit] - n[hit, None] * self.W[hit]) / self.count[hit, None]

    fit = DopamineReadout.fit
