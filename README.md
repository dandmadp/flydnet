# flydnet

**Use the real *Drosophila* connectome (FlyWire v783) as neural-network layers, and ask whether the wiring
actually matters.** Pick any set of neurons by annotation (mushroom body, visual system, whole brain), wire a layer
exactly as the fly's synapse map, and train it with backprop or dopamine-like associative learning. `fd.compare`
trains the same model on the real connectome and on nested null models over paired seeds, runs exact sign-flip
tests, warns about pitfalls, and reads the nested controls to say *which* structure matters.
Own autograd engine on NumPy (CPU) / CuPy (GPU); PyTorch optional. Docs in Korean.

> **Alpha (0.1).** API가 바뀔 수 있다. 연구용 도구이며, 정확도 향상을 기대할 도구는 아니다 (아래 "결과 요약").

초파리 커넥톰의 실제 배선을 신경망 층으로 쓰고, **"이 배선이 정말 중요한가"**를 통계로 묻는 라이브러리.

## 설치

```bash
pip install flydnet                  # CPU (NumPy·SciPy). torch 없음. CuPy가 이미 있으면(Colab 등) GPU도 자동으로 씀
python -m flydnet doctor             # 설치 진단: GPU 드라이버 CUDA·CuPy·torch, 어떤 옵션을 쓸지 알려 줌
python -m flydnet download           # FlyWire v783 연결·주석 + DoOR 냄새 데이터 (약 130 MB, 버전 고정·SHA-256 확인)
```

GPU: **CuPy가 없을 때만** 드라이버 CUDA 버전(`nvidia-smi` 오른쪽 위)에 맞는 옵션 하나를 쓴다.

```bash
pip install "flydnet[gpu-cuda13]"    # CUDA 13.x 드라이버 (CuPy + CUDA 런타임)
pip install "flydnet[gpu-cuda12]"    # CUDA 12.x 드라이버
pip install "flydnet[torch]"         # torch 연동 (flydnet.torch)이 필요할 때
```
Colab처럼 CuPy가 이미 깔린 곳에 GPU 옵션을 붙이면 CuPy가 두 개가 되고 CUDA 라이브러리가 바뀌어 torch까지
망가질 수 있다 (`doctor`가 잡아 줌. Colab이면 런타임을 삭제하고 옵션 없이 다시 설치).
드라이버를 업데이트해도 설치한 CuPy는 그대로 돈다 (드라이버는 하위 호환). CUDA·CuPy를 바꾼 뒤에는 `doctor`를 한 번 돌릴 것.
데이터 위치는 `~/.flydnet/data` (`fd.set_data_dir(...)`로 바꿈). `FLYDNET_DEVICE=cpu`로 CPU를 강제할 수 있다.

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
rule = fd.AdaptivePlasticity(model.named_synapses(), rate=1e-3, clip=1.0)   # clip: 기울기 크기 제한
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
real (실제 배선)                 0.7622    [0.7449, 0.7795]
shuffled                     0.7678    [0.7481, 0.7875]    -0.005556  -0.30   0.688      2/6
randomized                   0.6336    [0.6158, 0.6514]      +0.1286   5.36   0.031      6/6
shuffled_weights             0.7678    [0.7537, 0.7819]    -0.005556  -0.36   0.406      3/6

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

## 가상 유전학: fd.genetics

초파리 실험실의 방법 그대로: 드라이버로 뉴런 집단을 고르고, 효과기로 끄고·켜고·막는다.

```python
brain = fd.Circuit.whole_brain()                                  # 138,639개 뉴런 (Shiu et al. 2024 모델과 같은 순서)
G = fd.genetics
sugar = G.driver(brain, cell_sub_class="sugar")                   # GAL4: 주석 조건으로 (root_ids=, group=도 가능)
mn9 = G.driver(brain, root_ids=[720575940660219265])
layer = fd.ConnectomeLayer(brain, inputs=None, outputs="motor", t_ms=1000)
with G.activate(layer, sugar, hz=100):                            # CsChrimson: 포아송 자극
    r = layer(None, batch=30, return_all=True)                    # 30번 시행, 모든 뉴런 발화율
```

| 효과기 | 실험실 도구 | 효과 |
|---|---|---|
| `activate(layer, line, hz)` | CsChrimson, P2X2 | 포아송 자극 (연속값 뉴런은 `level`로 고정) |
| `silence(layer, line)` | Kir2.1 | 발화 없음 |
| `block(layer, line)` | Shibire-ts | 발화는 하지만 시냅스 전달 차단 |
| `ablate(circuit, line)` | 세포 제거 | 연결을 뺀 새 회로 (`fd.compare`로 학습 비교) |

- 드라이버 조합: `a & b` (split-GAL4), `a | b`, `a - b`. `lines(circuit, by="cell_type")`는 세포 유형마다 드라이버.
- `screen(measure, layer, lines)`: 집단마다 효과기를 발현해 측정값 변화와 짝지은 p값을 표로 (유전자 스크린).
- 효과기는 `with` 블록 안에서만 (또는 `.remove()`까지), 학습 중에도 쓸 수 있다 (역전파 됨).

**검증 - 원본 Shiu et al. 2024 Brian2 모델과 비교** (같은 뉴런 ID·매개변수, 1초 x 30시행, `validation/shiu2024/`):

| MN9 발화율 (Hz) | 무자극 | 단맛 25 | 단맛 50 | 단맛 100 | 단맛 200 | 쓴맛 100 | 단맛 + 쓴맛 |
|---|---|---|---|---|---|---|---|
| 원본 Brian2 | 0 | 0 | 12.6 | 61.8 | 88.8 | 0 | 2.2 |
| flydnet | 0 | 0.1 | 13.9 | 61.4 | 90.6 | 0 | 1.8 |

모든 조건에서 차이는 시행 간 잡음 안 (p > 0.17). 반응한 뉴런 329개의 발화율 상관 0.9985.
같은 입력 스파이크면 스파이크 시각까지 같다 (테스트로 고정). 시행당 약 17배 빠름 (GPU).

## 회로 기여도: fd.explain

학습된 모델이 답을 낼 때 **어떤 세포 유형·경로에 기대는지 실제 이름으로**, 그리고 **실제로 꺼서 확인**한다.

```python
def score(layer, seed):                                   # 설명할 값: 예) 정답 로짓의 합
    return readout(layer(enc(X), seed=seed))[np.arange(len(y)), y].sum()

rep = fd.explain(score, layer, by="cell_type", pathways=True, verify=5)
print(rep)          # 유형별·경로별 기여 + 실제로 끈 결과와의 순위 상관
```

- 방법: 뉴런·연결마다 배율(=1) 탐침을 두고 기울기를 구한다 = "끄면 얼마나 줄어드는가"의 1차 예측.
  `fd.genetics.silence`와 같은 조작이라, `verify`로 예측을 실제 손상과 바로 비교한다.
- 실제 냄새(DoOR) 8개를 구분하는 스파이킹 버섯체 모델 (`examples/explain_odor.py`, 정확도 98.8%):
  - 기여가 큰 유형: KCγ, KCαβ, 그다음 사구체별 PN. APL(억제)은 음수 (끄면 정답 로짓이 오름)
  - 실제로 꺼서 확인: 예측과 순위 상관 0.92 (크기는 1차 근사라 PN에서 2배 정도 과대)
  - 냄새마다 모델이 기대는 사구체와 그 냄새에 실제로 반응하는 사구체: 8개 모두 양의 상관 (평균 0.71)

### 세포 유형 드롭아웃: fd.genetics.mosaic

학습 중에만, 시료마다 세포 유형을 통째로 무작위로 끈다 (유전 모자이크). `by="neuron"`이면 보통 드롭아웃.

```python
fd.genetics.mosaic(layer, p=0.2, by="cell_type", within=fd.genetics.driver(mb, group="PN"))
```

실제 냄새 20개, 스파이킹 버섯체, seed 6개 짝지음 (`examples/mosaic_odor.py`, 약 6분). 평가 = 냄새 구분에 중요한
사구체 10개를 하나씩 없앴을 때 (후각 수용체 결손 모사):

| 학습 | 깨끗 | 사구체 결손 평균 | 최악의 결손 |
|---|---|---|---|
| 드롭아웃 없음 | 0.934 | 0.885 | 0.840 |
| PN 뉴런 드롭아웃 (p 0.2) | 0.973 | 0.932 | 0.867 |
| PN 세포 유형 드롭아웃 (p 0.2) | 0.971 | **0.950** | **0.923** |

세포 유형 드롭아웃은 보통 드롭아웃과 깨끗한 정확도는 같고 (p = 0.75), 결손에는 더 강하다
(+1.7%p, 6/6 seed, p = 0.031, 최악의 결손 +5.6%p). 학습 때 끄는 단위와 평가 때 잃는 단위가 같은 것(사구체)이
이유일 것이므로, 손상이 세포 유형 단위로 일어나는 상황에서 쓸모 있다.

## 역전파 없는 학습: fd.ThreeFactor

뇌에서 가능한 정보만으로 커넥톰 연결을 학습하는 3요소 규칙 (e-prop 방식): 시냅스 변화 = 학습 신호(오차) x 시냅스 후
민감도 x 시냅스 전 흔적. 시간을 거슬러 가지 않아 메모리가 시뮬레이션 길이와 무관하다.

```python
tf = fd.ThreeFactor(layer, feedback="connectome")       # 오차를 실제 연결을 따라 (또는 "random", "none")
out = tf(x, seed=s)
loss = fd.surprise(readout(out), y); rule.clear(); loss.retrograde()   # 리드아웃까지만
tf.assign(out); rule.step()                                             # 흔적 x 신호 → 커넥톰 연결
```

- 출력 뉴런으로 들어오는 연결의 기울기는 시간 역전파와 같다 (테스트로 고정). 숨은 연결은 피드백으로 받은 오차를 쓴다.
- GPU 메모리 (버섯체, 배치 32): 시간 역전파 50 ms 261 MB → 800 ms 4.1 GB, 3요소 규칙은 길이와 무관하게 41 MB.
- 실제 냄새 12개, 버섯체 + 도파민 뉴런, PN>KC·KC>MBON 학습, seed 6개 (`examples/threefactor_odor.py`, 약 9분):

| 학습 | 정확도 |
|---|---|
| 리드아웃만 (커넥톰 고정) | 0.302 |
| 시간 역전파 | 0.721 |
| 3요소, 피드백 없음 (KC>MBON만) | 0.545 |
| 3요소, 무작위 피드백 | 0.637 |
| 3요소, 실제 연결 피드백 (MBON → KC, MBON → DAN → KC) | 0.587 |

역전파 없이 리드아웃만보다 +33%p (p = 0.031, 6/6), 역전파와의 차이는 8%p. 숨은 연결 학습은 무작위 피드백으로
+9%p (p = 0.031). 실제 피드백 경로는 피드백 없음보다 나은 경향(+4%p, 4/6, p = 0.13)이지만 무작위 피드백보다는
못했다 (-5%p, 1/6, p = 0.13) - 이 설정에서는 실제 MBON → DAN → KC 경로가 오차를 그대로 전달하지 않는다.

## 구성 요소

| 무엇 | 이름 |
|---|---|
| 회로 고르기·대조군 | `Circuit.from_flywire(groups, side)`, `.shuffled()`, `.randomized()`, `.shuffled_weights()`, `.subset()` |
| 커넥톰 배선 층 (시간 없음) | `Neuropil` (학습: `"edge"` / `"pair"` / `"free"` / 고정), `LateralInhibition`, `AxonHillock` |
| 시간 시뮬레이션 | `ConnectomeLayer` — 스파이킹 LIF(Shiu et al. 2024 Brian2 모델과 같은 한 스텝)·연속값 뉴런, 시간 역전파, 체크포인팅 |
| 역전파 없는 학습 | `MushroomBodyOutput`, `AssocReadout`, `DopamineReadout` (도파민 국소 규칙) |
| 인코더·리드아웃 | `RateEncoder`, `GlomerularEncoder`, `KCExpansion`, `extract`, `train_linear` |
| 데이터 | `synthetic_odors`, `door_odors` (DoOR 2.0), `biconditional_mixtures` |
| 시각계 | `visual_circuit`, `column_map`, `drifting_grating`, `direction_offsets`, `MOTION_PATHWAY` |
| 대조 실험 | `compare`, `controls.Shuffled / Randomized / ShuffledWeights / Local / Custom` |
| torch 연동 | `flydnet.torch.*` — 0.1의 torch판 전부 (같은 이름), `torch.nn`용 구조물 |

전체 뇌(13.9만 뉴런, 연결 1,509만)도 일반 GPU에서 학습된다. `scripts/bench.py` (RTX 5070 12 GB, 감각 → 하행 뉴런,
연결 1,509만 개 모두 학습, 20 ms = 200스텝): 학습 1스텝 배치 8에 1.1초 (GPU 3.3 GB), 배치 32에 2.6초 (8.6 GB).
시뮬레이션만은 1초 시행 30개에 약 26초 (원본 Brian2 CPU는 1시행 약 15초).

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
