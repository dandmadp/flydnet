"""발화율 특징 위에 학습하는 리드아웃"""
import torch
import torch.nn as nn


def extract(layer, encoder, X: torch.Tensor, batch: int = 256, seed: int = 0, log_every: int = 0) -> torch.Tensor:
    """데이터 X 전체를 층에 통과시켜 출력 발화율 특징 (n, n_out)을 CPU로 반환"""
    out = []
    for i in range(0, len(X), batch):
        out.append(layer(encoder(X[i:i + batch].to(layer.dev)), seed=seed + i).cpu())
        if log_every and (i // batch) % log_every == 0:
            print(f"    {i + len(out[-1]):6d}/{len(X)}", flush=True)
    return torch.cat(out)


def train_linear(Xtr, ytr, Xte, yte, n_classes=None, epochs=30, lr=1e-2, wd=1e-4, batch=512,
                 device=None, scale: str = "global") -> dict:
    """정규화 + 로지스틱 회귀(소프트맥스). 정확도 반환

    scale: "global"  = 특징별 평균 빼고 전체 표준편차 하나로 나눔 (드물게 켜지는 특징이 폭주하지 않음)
           "feature" = 특징별 표준화 (드문 특징은 학습 때 sd≈0 → 테스트에서 값이 폭주할 수 있음)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = n_classes or int(ytr.max()) + 1
    mu = Xtr.mean(0)
    sd = Xtr.std(0).clamp_min(1e-6) if scale == "feature" else (Xtr - mu).std().clamp_min(1e-6)
    f = lambda X: ((X - mu) / sd).float().to(device)
    Xtr_, Xte_, ytr_, yte_ = f(Xtr), f(Xte), ytr.to(device), yte.to(device)
    model = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_), generator=g).to(device)
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            loss = nn.functional.cross_entropy(model(Xtr_[j]), ytr_[j])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        acc = lambda X, y: (model(X).argmax(1) == y).float().mean().item()
        return dict(train_acc=acc(Xtr_, ytr_), test_acc=acc(Xte_, yte_), model=model)
