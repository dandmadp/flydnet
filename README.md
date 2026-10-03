# flydnet

**Use the real *Drosophila* connectome (FlyWire v783) as neural-network layers, and ask whether the wiring
actually matters.** Pick any set of neurons by annotation (mushroom body, visual system, whole brain), wire a layer
exactly as the fly's synapse map, and train it with backprop or dopamine-like associative learning. `fd.compare`
trains the same model on the real connectome and on nested null models over paired seeds, runs exact sign-flip
tests, warns about pitfalls, and reads the nested controls to say *which* structure matters.
Own autograd engine on NumPy (CPU) / CuPy (GPU); PyTorch optional. Docs in Korean.

> **Alpha (0.2).** API가 바뀔 수 있다. 연구용 도구이며, 정확도 향상을 기대할 도구는 아니다 (아래 "결과 요약").

초파리 커넥톰의 실제 배선을 신경망 층으로 쓰고, **"이 배선이 정말 중요한가"**를 통계로 묻는 라이브러리.

## 설치

```bash
pip install flydnet                  # CPU (NumPy·SciPy). torch 없음
pip install "flydnet[gpu-cuda12]"    # + GPU (CuPy, CUDA 12.x 드라이버). CUDA 13.x면 [gpu-cuda13]
pip install "flydnet[torch]"         # + torch 연동 (flydnet.torch)

python -m flydnet download           # FlyWire v783 연결·주석 + DoOR 냄새 데이터 (약 130 MB, 버전 고정·SHA-256 확인)
python -m flydnet                    # 데이터 상태
```
CUDA 버전은 `nvidia-smi` 오른쪽 위에 나온다. 데이터 위치는 `~/.flydnet/data` (`fd.set_data_dir(...)`로 바꿈).
`FLYDNET_DEVICE=cpu`로 CPU를 강제할 수 있다.

## 빠른 시작

```python
import numpy as np, flydnet as fd

mb = fd.Circuit.from_flywire()                       # 오른쪽 버섯체: PN 344, KC 2597, APL 1, MBON 48
model = fd.Pathway(
    fd.Projection(784, 344),                         # 축삭 투사: 픽셀 → PN (모두 연결)
    fd.Neuropil(mb, "PN", "KC"),                     # 실제 PN→KC 배선 (연결 13,485개만, 부호 유지 학습)
    fd.LateralInhibition(frac=0.05),                 # APL 억제: KC 5%만
    fd.Projection(2597, 10),
)
rule = fd.AdaptivePlasticity(model.synapses(), rate=1e-3)
loss = fd.surprise(model(x), y)                      # 놀람 = 교차 엔트로피
rule.clear(); loss.retrograde(); rule.step()         # 역행성 신호(자동 미분) → 가소성
```
MNIST 2에폭 96.2% (GPU 약 6초, `examples/ganglion_mnist.py`).

## 대조 실험: fd.compare

같은 학습 절차를 실제 배선과 대조군 배선에 seed마다 짝지어 돌리고 비교한다.

```python
def run(circuit, seed):                    # 회로 하나로 모델을 만들고 학습해서 점수 하나 (seed를 학습 난수에)
    ...
    return accuracy

report = fd.compare(run, mb, controls=["shuffled", "randomized", "shuffled_weights"], seeds=6, chance=1/30)
print(report)
```
```
조건                               평균              95% CI      실제 - 대조      d       p    실제 우세
real (실제 배선)                 0.7806     [0.759, 0.8021]
shuffled                     0.7833    [0.7685, 0.7982]    -0.002778  -0.16   0.750      1/6
randomized                   0.6517     [0.6353, 0.668]      +0.1289   5.35   0.031      6/6
shuffled_weights             0.7736     [0.755, 0.7922]    +0.006944   0.73   0.188      4/6

해석 (대조군 포함 관계):
  → 실제 배선이 randomized는 이기고 shuffled와는 차이가 없음 → 중요한 구조: 뉴런별 연결 수 분포 (차수·허브)
```
(스파이킹 버섯체, 합성 냄새 30클래스, `examples/compare_odor.py --model lif`, 약 20초)

| 대조군 | 유지하는 구조 | 묻는 것 |
|---|---|---|
| `"randomized"` | 그룹 쌍별 연결 수 | 연결 수 분포(허브·차수)가 중요한가 |
| `"shuffled"` | + 뉴런별 연결 수 | 누가 누구와 연결되는가 |
| `fd.controls.Local(xy, r)` | + 시야 위치 대응 | 큰 공간 구조 말고 세부 배선까지 |
| `"shuffled_weights"` | 배선 그대로, 세기만 섞음 | 시냅스 세기 분포 |

- 같은 seed끼리 짝지어 학습 잡음을 상쇄하고, 부호 뒤집기 순열 검정(분포 가정 없음, seed ≤ 14면 정확한 p)을 쓴다.
- 대조군 포함 관계(randomized ⊂ shuffled ⊂ local ⊂ 실제)로 **어떤 구조가 중요한지** 해석한다.
- 함정을 경고한다: seed가 적어 p가 0.05 아래로 내려갈 수 없음, 상한 근처라 차이가 가려짐, 찍기 수준,
  같은 seed인데 결과가 다름, 큰 구조 때문에만 이김, 효과는 큰데 비유의.

## 구성 요소

| 무엇 | 이름 |
|---|---|
| 회로 고르기·대조군 | `Circuit.from_flywire(groups, side)`, `.shuffled()`, `.randomized()`, `.shuffled_weights()`, `.subset()` |
| 커넥톰 배선 층 (시간 없음) | `Neuropil` (학습: `"edge"` / `"pair"` / `"free"` / 고정), `LateralInhibition`, `AxonHillock` |
| 시간 시뮬레이션 | `ConnectomeLayer` — 스파이킹 LIF(Shiu et al. 2024 매개변수)·연속값 뉴런, 시간 역전파, 체크포인팅 |
| 역전파 없는 학습 | `MushroomBodyOutput`, `AssocReadout`, `DopamineReadout` (도파민 국소 규칙) |
| 인코더·리드아웃 | `RateEncoder`, `GlomerularEncoder`, `KCExpansion`, `extract`, `train_linear` |
| 데이터 | `synthetic_odors`, `door_odors` (DoOR 2.0), `biconditional_mixtures` |
| 시각계 | `visual_circuit`, `column_map`, `drifting_grating`, `direction_offsets`, `MOTION_PATHWAY` |
| 대조 실험 | `compare`, `controls.Shuffled / Randomized / ShuffledWeights / Local / Custom` |
| torch 연동 | `flydnet.torch.*` — 0.1의 torch판 전부 (같은 이름), `torch.nn`용 구조물 |

전체 뇌(13.9만 뉴런, 연결 1,509만)도 일반 GPU에서 학습된다 (`ConnectomeLayer` 학습 1스텝, RTX 5070:
배치 8에 0.58초, 배치 32에 1.3초).

### 자체 엔진 이름 (torch 대응)

| torch | flydnet | | torch | flydnet |
|---|---|---|---|---|
| `Tensor` | `Signal` | | `nn.Module` | `Tissue` |
| `requires_grad` | `plastic` | | `nn.Sequential` | `Pathway` |
| `backward()` / `.grad` | `retrograde()` / `.retro` | | `nn.Linear` | `Projection` |
| `no_grad()` | `quiescent()` | | `optim.SGD` / `Adam` | `Plasticity` / `AdaptivePlasticity` |
| `nn.Parameter` | `Synapse` | | `F.cross_entropy` | `surprise` |
| `utils.checkpoint` | `checkpoint` | | `"cuda"` | `"gpu"` |

이름은 같은 일을 하는 생물 구조에서 땄다 (역행성 신호, 시냅스, 조직, 신경 경로, 가소성…).
엔진 검증: 연산마다 수치 미분, torch와 출력·기울기·옵티마이저 비교, CPU↔GPU 비교 (`tests/`).

### torch 연동

0.1 코드는 `fd.이름`을 `fd.torch.이름`으로 바꾸면 그대로 돈다
(`fd.torch.ConnectomeLayer`, `fd.torch.AssocReadout`, `fd.torch.extract` 등).

## 결과 요약

12개 실험 (자세한 방법·수치·한계는 소스 저장소의 `REPORT.md`):

| 질문 | 답 |
|---|---|
| 실제 배선이 정확도를 높이나 | 대체로 아니다. 차수와 큰 구조(위치 대응)를 유지한 무작위 배선과는 MNIST·냄새·시각 과제에서 차이 없음 |
| 그럼 배선의 무엇이 중요했나 | 큰 구조: 시각계의 위치 대응(없으면 학습 불가), 스파이킹 버섯체의 KC 연결 수 분포(무작위면 −13%p) |
| 도파민 연합 학습은 | 연속 학습에서 쓸모 있음: CIFAR-100 10과제 57.7% (재생 버퍼 역전파 51.6%), 역전파 없이 한 번 보기 |
| KC 확장 층은 | 선형 리드아웃일 때만 도움 (XOR형 과제 +19%p) |

## 개발

```bash
python -m venv .venv
.venv\Scripts\pip install --no-cache-dir -e ".[examples,dev,gpu-cuda13]"
.venv\Scripts\python -m pytest -q                       # 테스트
.venv\Scripts\python scripts\release.py bump patch      # 버전 올리기 + CHANGELOG 틀
.venv\Scripts\python scripts\release.py check --upload  # 커밋·CHANGELOG·PyPI 중복·테스트·빌드·설치 확인 후 업로드
```

데이터 출처: FlyWire v783 연결(Shiu et al. 2024, MIT), 세포 주석(Schlegel et al. 2024), DoOR 2.0(CC BY-SA 4.0).
MIT 라이선스.
