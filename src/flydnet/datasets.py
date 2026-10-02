"""시험용 데이터 생성"""
import torch


def synthetic_odors(n_classes: int, n_glomeruli: int, n_train: int, n_test: int, protos_per_class: int = 1,
                    active_frac: float = 0.2, noise: float = 0.5, add_noise: float = 0.1, seed: int = 0):
    """합성 냄새 분류 과제

    냄새 원형: 사구체마다 active_frac 확률로 켜짐, 세기 U(0.2, 1)
    클래스:   원형 protos_per_class개의 묶음 (예: '먹이' = 사과 냄새 또는 효모 냄새).
              2개 이상이면 사구체 공간에서 선형 분리가 어려워짐 → 확장 층(KC)이 필요한 과제
    샘플:     클래스의 원형 하나를 골라 × exp(N(0, noise)) (사구체별 세기 흔들림) + |N(0, add_noise)| (배경)
    반환: (Xtr, ytr, Xte, yte), X는 (n, n_glomeruli), 클래스당 n_train / n_test 개
    """
    g = torch.Generator().manual_seed(seed)
    P = n_classes * protos_per_class
    on = torch.rand(P, n_glomeruli, generator=g) < active_frac
    protos = on * (0.2 + 0.8 * torch.rand(P, n_glomeruli, generator=g))

    def sample(n):
        y = torch.arange(n_classes).repeat_interleave(n)
        k = y * protos_per_class + torch.randint(0, protos_per_class, (len(y),), generator=g)
        x = protos[k] * torch.exp(noise * torch.randn(len(y), n_glomeruli, generator=g))
        x = x + (add_noise * torch.randn(len(y), n_glomeruli, generator=g)).abs()
        return x, y

    Xtr, ytr = sample(n_train)
    Xte, yte = sample(n_test)
    return Xtr, ytr, Xte, yte
