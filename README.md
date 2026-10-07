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
pip install flydnet                  # CPU. 필요한 것은 numpy·pandas뿐 (scipy·pyarrow·torch 없음). CuPy가 이미 있으면 GPU도 자동
python -m flydnet doctor             # 설치 진단: GPU 드라이버 CUDA·CuPy·torch, 어떤 옵션을 쓸지 알려 줌
python -m flydnet download           # FlyWire v783 연결·주석 + DoOR 냄새 데이터 (약 130 MB, 버전 고정·SHA-256 확인)
```

FlyWire 연결 파일(parquet)은 받은 뒤 한 번 numpy 형식(`Connectivity_783.npz`)으로 바꿔 두고 그것을 읽는다.
바꾸는 데만 pyarrow가 필요하다: `pip install "flydnet[parquet]"` 후 `python -m flydnet download flywire` (그 뒤에는 필요 없음,
다른 컴퓨터에서 만든 npz를 같은 폴더에 복사해도 됨). 이미 받아 두었고 pyarrow가 있으면 처음 회로를 만들 때 자동으로 바뀐다.

선택 설치 (필요한 기능만):

| 옵션 | 무엇 | 언제 |
|---|---|---|
| `[parquet]` | pyarrow | 연결 parquet → npz 변환 (한 번) |
| `[scipy]` | scipy | `Circuit.from_scipy`·`to_scipy`, C 커널이 없는 플랫폼의 CPU 희소 연산 |
| `[graph]` | networkx | `Circuit.to_networkx` |
| `[torch]` | torch | `fd.torch` 연동 |
| `[gpu-cuda13]`·`[gpu-cuda12]` | CuPy + CUDA 런타임 | GPU (아래) |

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

### CPU 희소 연산

CPU에서 커넥톰 전달(희소 행렬 x 배치)은 직접 작성한 C 커널이 한다 (`flydnet/ganglion/csr.py`, `_csr.c`).
Linux (x86_64·aarch64)·macOS (x86_64·arm64)·Windows (x86_64·arm64)용이 설치 파일 하나에 모두 들어 있고, 파이썬 스레드로
행을 나눠 계산한다 (OpenMP를 쓰지 않아 런타임 설치가 필요 없고, 스레드 수와 상관없이 결과 비트가 같음).
처음 열 때 작은 행렬로 numpy 결과와 비교해, 다르면 쓰지 않는다. 쓸 수 있는 경로를 이 순서로 고른다:

1. C 커널 (위 6개 플랫폼)
2. scipy (설치되어 있으면)
3. numpy (느림 - 처음 한 번 경고)

세 경로 모두 행마다 연결 순서대로 더해 **같은 입력이면 같은 비트**를 낸다 (`tests/test_golden.py`가 세 경로에서 확인).

| 시각계 전체 (연결 429만, 배치 32, float32) | 시간 |
|---|---|
| C 커널, 8 스레드 | 2.8 ms |
| C 커널, 2 스레드 | 6.6 ms |
| C 커널, 1 스레드 | 13.0 ms |
| scipy (1 스레드) | 17.5 ms |
| numpy | 489 ms |

`FLYDNET_SPARSE=c|scipy|numpy`로 경로를, `FLYDNET_THREADS=n`으로 스레드 수(기본 CPU 수, 최대 8)를 정한다.
비교: `python scripts/bench_sparse.py`. GPU는 이 경로를 쓰지 않는다 (전용 CUDA 커널).

## 빠른 시작

가장 짧게 (PyTorch의 `nn.Sequential`처럼) - 입력을 발화율로 바꾸고, 연결 세기를 데이터에 맞게 자동 보정하고,
출력 뉴런을 클래스로 바꾸는 층까지 붙인 모델:

```python
import flydnet as fd

Xtr, ytr, Xte, yte = fd.door_task(n_odors=12)              # DoOR 실제 냄새 (사구체 반응)
model = fd.MushroomBody(n_in=Xtr.shape[1], n_classes=12)   # 오른쪽 버섯체 PN → KC 2,597개
model.fit(Xtr, ytr, val=(Xte, yte))                        # 처음 데이터로 보정한 뒤 학습 (fd.train)
model.score(Xte, yte)                                      # 0.976 (찍기 0.083), GPU 약 25초
```

어떤 회로·그래프든 `fd.ConnectomeModel(circuit, "in", "out", n_in, n_classes)`. 세부는 `model.encoder`·`model.layer`·
`model.head`, 또는 아래처럼 직접 조립. KC 출력이 MBON 출력(48개뿐)보다 정확도가 훨씬 높음 (같은 과제 0.97 대 0.36).

직접 조립:

```python
import numpy as np, flydnet as fd

mb = fd.flywire()                       # 오른쪽 버섯체: PN 344, KC 2597, APL 1, MBON 48
model = fd.Pathway(
    fd.Projection(784, 344),                         # 축삭 투사: 픽셀 → PN (모두 연결)
    fd.Neuropil(mb, "PN", "KC"),                     # 실제 PN→KC 배선 (연결 13,485개만, 부호 유지 학습)
    fd.Inhibition(frac=0.05),                 # APL 억제: KC 5%만
    fd.Projection(2597, 10),
)
rule = fd.Adaptive(model.named_synapses(), rate=1e-3, clip=1.0)   # clip: 기울기 크기 제한
loss = fd.surprise(model(x), y)                      # 놀람 = 교차 엔트로피
rule.clear(); loss.retrograde(); rule.step()         # 역행성 신호(자동 미분) → 가소성
```
MNIST 2에폭 96.2% (GPU 약 6초, `examples/ganglion_mnist.py`).

학습 루프를 한 줄로 (`fd.train`: 배치·섞기·코사인 학습률·기울기 제한·평가), 실제 냄새 과제도 한 줄로 (`fd.door_task`),
가중치는 자동 보정 (`calibrate`: 그룹마다 목표 발화율이 되도록 들어오는 연결 배율을 맞춤):

```python
Xtr, ytr, Xte, yte = fd.door_task(n_odors=12)                       # DoOR 실제 냄새
enc = fd.Glomeruli(mb)
layer = fd.Connectome(mb, "PN", "KC", t_ms=50, dt=0.5, input_mode="regular", trainable=["PN>KC"])
layer.calibrate(enc(Xtr[::3]), {"KC": 5})                           # KC 평균 5 Hz가 되도록 배율 자동 조정
model = fd.Pathway(enc, layer, fd.Homeostasis(), fd.Projection(2597, 12))
hist = fd.train(model, Xtr, ytr, val=(Xte, yte), epochs=12)         # 평가 정확도 0.96 (찍기 0.083), examples/quickstart.py
```

출력이 0 Hz일 때 (신호가 출력까지 가지 않음 - 기본 세기로는 버섯체 MBON의 96%, 합성 그래프는 전부 조용함):

```python
R = fd.Glomeruli(mb)(Xtr[::3])               # 대표 입력 (Hz)
layer = fd.Connectome(mb, "PN", "MBON")
print(layer.reach(R))                      # 그룹마다 입력(·activate 자극)에서의 홉 수·발화율, 신호가 끊기는 곳
layer.calibrate(R, {"MBON": 20})           # 출력만 목표로 줘도 경로 위 중간 그룹(KC)까지 함께 깨움 (억제 뉴런 APL은 제외)
```

처음 순전파에서 입력이 있는데 출력이 모두 0이면 경고가 나온다 (입력 뉴런조차 발화하지 않았으면 입력 쪽 문제로 알려 줌).
층을 만들 때 입력에서 경로가 없는 출력 그룹, 시냅스 지연보다 짧은 `t_ms`도 경고한다.

| 같은 과제, seed 4개 | 정확도 |
|---|---|
| MBON 48개에서 읽기 (배율 3 손으로) | 0.57 |
| KC 2,597개에서 읽기, 배율 3 손으로, 학습률 1e-2 | 0.938 ± 0.013 |
| KC에서 읽기, 자동 보정, 학습률 3e-3 (`fd.train` 기본) | **0.962 ± 0.004** |

MBON은 48개뿐이라 병목. 자동 보정의 정확도 이득은 잡음 안 (+0.8%p) - 이점은 배율을 손으로 맞출 필요가 없다는 것.

## 한 줄 학습기: fd.Learner

커넥톰은 고정하고, 데이터는 한 번 훑어 배우고, 새 데이터는 이어서 배운다. 입력 변환·연결 세기 보정은 처음 데이터로 자동.
학습률·에폭을 고를 필요가 없고, 새 클래스를 배워도 앞에서 배운 것은 그대로다.

```python
learner = fd.Learner()                              # 오른쪽 버섯체 PN → KC (fd.Learner(circuit, "in", "out")도)
learner.learn(X_a, y_a)                             # 처음: 입력 변환·보정 자동, 한 번 훑기
learner.learn(X_b, y_b)                             # 새 클래스 이어서 - 앞의 클래스 평균은 그대로
learner.learn(X_img, y_img, source="image")         # 다른 종류(특징 수가 다른) 데이터: 입력 변환만 새로, 기억은 공유
learner.predict(X), learner.score(X, y)             # 라벨은 문자열 등 아무 값이나
learner.save("me.npz"); fd.Learner.load("me.npz")   # 불러온 뒤에도 계속 배움
```

| rule | 방식 | 냄새 24개, 6개씩 4번 차례로 | 한 번에 전부 |
|---|---|---|---|
| `"lda"` (기본) | 흐름 선형 판별 (Hayes & Kanan 2020): 클래스 평균 + 공유 공분산을 누적, 한 번 훑기 | **0.859** | **0.865** |
| `"assoc"` | 도파민 연합 학습: 정답 클래스 원형만 강화, 한 번 훑기 (생물에 가장 가까운 국소 규칙) | 0.744 | 0.712 |
| `"backprop"` | 망각을 줄인 역전파 (아래), 기본 5에폭 | 0.708 | 0.841 |

DoOR 냄새 24개 (잡음 1.2, 찍기 4.2%), seed 3개 평균. 커넥톰까지 학습하는 `fd.MushroomBody`·`fd.train`(0.89 수준)에 가깝고,
차례로 배워도 한 번에 배운 것과 거의 같다 (잊지 않음). `lda`는 KC끼리 겹치는 반응을 공분산으로 걸러 (백색화 - 측면 억제의
탈상관과 같은 효과) 평균만 쓰는 것보다 훨씬 잘 구별한다. 공분산 축소는 자동 (`shrink="auto"` = max(OAS 추정, 특징 수 / 본 시료 수)).
같은 특징에서 class-incremental, 재생 없음:

| 특징 | lda | assoc | 망각을 줄인 역전파 | 역전파 그대로 |
|---|---|---|---|---|
| MNIST KC (5과제) | **0.896** | 0.728 | 0.656 | 0.419 |
| CIFAR-100 resnet18 (10과제) | **0.640** | 0.539 | 0.585 | 0.247 |

CIFAR의 0.640은 한 번에 전부 학습한 선형 분류(0.658)에 가깝다. 단 `lda`는 출력 뉴런 수의 제곱만큼 메모리를 쓴다
(KC 2,597개 → 54 MB, 12,000개 넘으면 오류 - `rule="assoc"`).

**역전파의 망각 줄이기** (`rule="backprop"`): 세 가지를 함께 쓴다.
1. 새 클래스 가중치를 그 클래스 평균 특징으로 시작한다 (연합 규칙으로 초기화한 뒤 역전파로 다듬기).
2. 이번 `learn()`에 나온 클래스끼리만 경쟁한다 (도파민이 그 출력에만 오는 것과 같음 - 다른 클래스 가중치는 안 바뀜).
3. 코사인 점수를 쓴다 (특징·가중치 크기 정규화). 특징 중심은 처음 데이터로 정해 고정한다.

class-incremental, 과제당 1에폭, 재생 버퍼 없음, 본 클래스 전체 정확도 (seed 3개):

| 특징 | 역전파 그대로 | ②만 | ②+③ | ①+②+③ | 연합 (k=1) |
|---|---|---|---|---|---|
| CIFAR-100 resnet18 (10과제) | 24.7% | 47.0% | 54.7% | **58.5%** | 53.9% |
| MNIST 픽셀 (5과제) | 41.1% | 53.4% | 64.3% | 72.9% | **82.2%** |
| MNIST KC (5과제) | 41.9% | 56.9% | 63.7% | 65.6% | **72.8%** |

특징 중심을 빼면 효과가 사라진다 (①+②+③, CIFAR 26.3%). 중심을 첫 과제 데이터로만 정해도 거의 같다 (57.4%).

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
brain = fd.brain()                                  # 138,639개 뉴런 (Shiu et al. 2024 모델과 같은 순서)
G = fd.genetics
sugar = G.driver(brain, cell_sub_class="sugar")                   # GAL4: 주석 조건으로 (root_ids=, group=도 가능)
mn9 = G.driver(brain, root_ids=[720575940660219265])
layer = fd.Connectome(brain, inputs=None, outputs="motor", t_ms=1000)
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
| flydnet | 0 | 0 | 13.7 | 62.7 | 90.6 | 0 | 2.2 |

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
  - 실제로 꺼서 확인: KC 유형은 예측 크기까지 맞음 (5,960 대 5,730). 억제 뉴런 APL은 방향이 틀림
    (예측 -1,050, 실제 +6,500) → 순위 상관 0.44, APL은 "방향을 틀린 유형"으로 자동 표시. 되먹임 억제처럼 큰 손상의
    효과는 1차 예측으로 맞지 않으므로 결론은 실제로 끈 값으로
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
| 시간 역전파 | 0.741 |
| 3요소, 피드백 없음 (KC>MBON만) | 0.551 |
| 3요소, 무작위 피드백 | 0.649 |
| 3요소, 실제 연결 피드백 (MBON → KC, MBON → DAN → KC) | 0.575 |

역전파 없이 리드아웃만보다 +35%p (p = 0.031, 6/6), 역전파와의 차이는 9%p. 숨은 연결 학습은 무작위 피드백으로
+10%p (p = 0.031). 실제 피드백 경로는 피드백 없음과 차이가 뚜렷하지 않고(+2%p, 4/6, p = 0.34) 무작위 피드백보다
못했다 (-7%p, 0/6, p = 0.031) - 이 설정에서는 실제 MBON → DAN → KC 경로가 오차를 그대로 전달하지 않는다.

## 어떤 그래프든: 다른 커넥톰과 합성 그래프

초파리용으로 만든 도구(`ConnectomeLayer`, `compare`, `explain`, `genetics`, `ThreeFactor`)가 어떤 방향 그래프에서도 동작한다.

```python
c = fd.Circuit.from_edges(pre, post, weight, groups={"in": [...], "out": [...]})   # 번호 또는 이름
c = fd.Circuit.from_scipy(A)                     # A[i, j] = i → j  (또는 orientation="post_pre")
c = fd.Circuit.from_networkx(G)                  # 노드 속성 → 주석, "group" 속성 → 그룹
worm = fd.worm()                     # 예쁜꼬마선충 (Cook et al. 2019), python -m flydnet download worm
g = fd.graphs.watts_strogatz(400, 12, 0.1, groups={"in": range(20), "out": range(360, 400)})
#   erdos_renyi / watts_strogatz / barabasi_albert / stochastic_block / layered (억제 비율·데일의 법칙)
c.to_scipy(), c.to_networkx(), c.regroup({...}), c.check()   # scipy·networkx는 선택 설치 ([scipy]·[graph])
```

- 주석이 없는 그래프에서는 `explain`·`mosaic`·`genetics.lines`가 그룹 단위로 동작한다 (주석이 있으면 `cell_type`).
- 세기는 시냅스 수처럼 `w_syn`(0.275 mV)을 곱해 쓴다. 기본 세기로 출력이 조용하면 `layer.calibrate(X, {"out": 10})`
  (중간 층까지 자동으로 맞춤, 예: `layered([20, 100, 100, 10])` 출력 0 → 11 Hz).
- `examples/any_graph.py`:
  - 예쁜꼬마선충 감각 뉴런 24개 → 체벽 근육 95개로 패턴 구분 + `fd.compare`: 실제 배선의 이점은 보이지 않고,
    시냅스 세기만 섞은 대조군이 오히려 좋음 (+9.7%p, 6/6, p = 0.031). 임의의 패턴 구분은 이 회로가 하는 일이 아니므로
    흔한 결과
  - 그래프 종류 비교 (노드 400, 연결 약 4,700, 그래프마다 세기를 골라 **출력 그룹** 발화율 약 18 Hz로 맞춤 - 예전엔 전체 평균으로
    맞춰 블록 구조의 출력이 꺼진 채로 비교됐음): 무작위 0.64 > 척도 없음 0.60 > 작은 세상 0.37 > 블록 0.20
    (작은 세상·블록은 6/6, p = 0.031. 척도 없음은 무작위와 차이 없음 - 0.1.18에서 척도 없음 그래프 생성기가 대상 m개를
    서로 다르게 고르도록 고친 뒤, 예전 0.53). 단 입력·출력 노드의 위치
    (고리 위 거리, 다른 모듈)에 크게 좌우되므로, 그래프 구조 자체의 결론이 아니라 이런 질문을 하는 방법의 예

## 플러그인: 뉴런 모델·학습 규칙을 직접

**뉴런 모델** - 한 스텝만 정의하면 시뮬레이션(지연·입력·체크포인팅), 역전파(자동 미분), `genetics`, `explain`,
torch 연결 장치, 저장·불러오기가 따라온다.

```python
@fd.neurons.register
class MyNeuron(fd.neurons.NeuronModel):
    state = ("v",)
    def init(self, ctx): return {"v": ctx.full(-65.0)}
    def step(self, st, I_syn, I_ext, ctx):                  # Signal 연산만 → 역전파 자동
        v = st["v"] + ctx.dt * (-(st["v"] + 65.0) / 10.0) + I_syn + I_ext
        spk = fd.ganglion.fire(v, -50.0, slope=1.0)
        return {"v": fd.ganglion.where(spk.data > 0, -65.0, v)}, spk

layer = fd.Connectome(circuit, "in", "out", neuron=MyNeuron(), checkpoint_every=50)
```
- 내장: `fd.neurons.LIF` (내장 LIF와 스파이크가 스텝까지 같고 기울기 차이 4e-5 이내 - 플러그인 틀의 검증),
  `fd.neurons.Izhikevich`. 예제 `examples/custom_neuron.py`는 AdEx를 직접 정의한다
- 비용: 내장 LIF의 약 1.4배 시간, 중간값을 모두 저장하므로 학습할 때는 `checkpoint_every`

**학습 규칙** - 순전파를 스텝마다 지켜보는 관찰자 규격 (`begin(info)`, `step(s, spikes, **extra)`).
`fd.ThreeFactor`와 `fd.STDP`(쌍 기반 STDP, 곱셈형)가 이 규격을 따른다.

```python
stdp = fd.STDP(layer); stdp(x, seed=0); stdp.assign(); rule.step()    # 원인 → 결과 순서면 강화, 반대면 약화
```

## 스파이킹 역전파 기울기 확인: fd.gradcheck

대리 기울기는 되먹임 회로를 돌며 곱해져서, 감쇠 없이 길게 돌리면 방향이 무작위이거나 반대가 되고 크기가 수억 배로
부푼다 (0.1.15까지의 기울기: 초파리 버섯체 1,000스텝에서 cos -0.75, 16억 배). 0.1.16부터:

- `surrogate_damp="auto"` (기본): 시뮬레이션 동안 신호가 시냅스를 건널 수 있는 횟수(홉 수)로 감쇠를 정함 -
  짧으면 감쇠 없음, 길수록 강하게. 초파리 버섯체·전체 뇌에서 가장 잘 맞은 값을 맞추는 경험 규칙
- `noise=` 막전위 잡음 (선택): 대리 기울기가 '잡음 있는 뉴런의 발화 확률의 기울기'에 가까워짐
- `truncate=` 구간 절단 역전파 (시냅스 지연보다 길어야 함)

```python
print(fd.gradcheck(score, layer))            # 방향 일치(cos)·크기 비율·기준 신뢰도, 연결 종류별
print(fd.gradcheck(score, layer, seeds=16))  # 포아송 입력이면 여러 seed 평균 출력의 기울기로 (기준이 불안정할 때)
fd.tune(score, layer)              # 후보 중 자기 손실에 가장 잘 맞는 감쇠를 골라 적용
```

| 초파리, 방향 일치 cos | 감쇠 없음 | auto (기본) |
|---|---|---|
| 버섯체 100스텝 (dt 0.5) | 0.01 | ≈0.87 (0.29) |
| 버섯체 1,000스텝 | -0.75 (16억 배) | ≈0.91 (0.09, 크기 1.4배) |
| 전체 뇌 200스텝 | 0.985 | 0.985 (0.985) |

- `gradcheck`는 비교 기준(유한 차분)이 성립하는지도 잰다: 무작위성 없이 길게 돌린 회로는 출력이 연결 세기에 대해
  울퉁불퉁해 기준이 흔들린다 (전체 뇌 1,000스텝 규칙 입력: 판단 불가) → 포아송 입력과 `seeds=`로
- 대리 기울기가 부분적으로만 맞는 회로가 있다 (예쁜꼬마선충: 깨끗한 기준으로 cos 약 0.5).
  `fd.ThreeFactor`도 같은 대리 기울기라 이 부정확함은 그대로 (메모리·긴 시간 문제만 해결)
- `explain`은 기울기로 모든 유형을 한 번에 훑어 후보를 고르고, 결론은 실제로 끈 값(`verify`)으로.
  1차 예측이 방향까지 틀린 유형(되먹임 억제 뉴런 APL 등)은 확인 결과에 표시된다
- 표와 재현: `validation/gradients/`

## 실험적 기능: 추가 연결 (구조적 가소성, growth)

> **실험적 기능 (0.1.18)** - 인터페이스·기본값이 바뀌거나 없어질 수 있음. 효과는 조건부: 손상된 회로의 회복에서는 뚜렷하지만,
> 멀쩡한 회로와 MNIST에서는 이득이 없거나 무작위 연결보다 못한 경우가 있음 (아래 결과). 연구 용도로 쓰고, 결과는 늘
> 무작위 연결(`rule="random"`)과 짝지어 비교할 것

실제 배선은 그대로 두고, 학습으로 생기고 없어지는 시냅스를 따로 얹는다 (타고난 회로 + 경험으로 덧붙는 시냅스).
커넥톰 층의 "연결이 정해져 있다"는 한계를 풀면서, 실제 배선은 늘 기준으로 남는다.

```python
from flydnet import lab                           # 실험실 - 엔진에는 없고, 층에 붙는 부품 (layer.attach)
layer = fd.Connectome(mb, "PN", "KC", trainable=["PN>KC"])
g = lab.Growth(layer, allow=["PN>KC"], budget=5000)   # 층에 붙음 (layer.growth로도), 저장·불러오기 때 같이
g.grow(2000)                                      # 무작위 (허용된 그룹 쌍 안에서)
g.grow(2000, rule="coactive", rates=R)            # 헤브: 함께 많이 발화한 PN·KC 쌍
g.grow(2000, rule="gradient", loss=lambda L: fd.surprise(head(L(x)), y))   # 손실이 가장 줄 자리
lab.grow_contrastive(g, 2000, inputs=X, encoder=enc)                           # 정답 없이: 시료끼리 구별되게 (대조 학습)
g.prune(0.2)                                      # 추가 연결 중 약한 20% (실제 배선은 절대 안 건드림)
g.extra_edges()                                   # 표: pre, post, 경로, 세기(시냅스 수), 부호
```

- 생물학적 제약: 허용된 그룹 쌍 안에서만, 부호는 보내는 뉴런의 실제 부호 (데일의 법칙), 상한 `budget`·`per_neuron`
- **받는 뉴런마다 상한 `per_neuron`** (기본 `"auto"` = 올림(budget ÷ 받을 수 있는 뉴런 수), 버섯체 KC 2,597개에 5,400개면
  3개): 기울기·대조 학습 규칙이 효과가 커 보이는 소수 뉴런에 연결을 몰아, 그 뉴런들이 반응을 독차지하는 것을 막음
  (MNIST에서 상한이 없으면 KC 상위 10%가 새 연결의 51%를 가져감). 실제 KC마다 PN 입력 수가 비슷한 것, SRigL(뉴런마다
  입력 수 일정)과 같은 생각
  - **최고 성능이 필요하고 데이터가 고르게 퍼진다면 `per_neuron=None`**: 냄새처럼 상한이 없어도 연결이 여러 뉴런에 퍼지는
    데이터에서는 상한이 약간 손해 (냄새 손상 회복 5,400개: 상한 0.894, 없음 0.901). 퍼지는지는
    `np.bincount(g.extra_edges().post)`로 볼 수 있음 - 소수 뉴런에 몰려 있으면 상한을 켤 것
- 새 연결은 그 경로 실제 연결 하나 크기(시냅스 수 중앙값, 버섯체 PN→KC 10개, `init_syn`)로 생겨 역전파로 커지거나 작아짐
  (그 경로의 배율 `gains`도 같이). prune·grow 뒤에도 살아남은 연결의 학습 상태(Adam 관성)는 유지
- 추가 연결이 없으면 계산이 전과 같다. 같은 지연·효과기(genetics)·저장·GPU. 예: `lab/growth_odor.py`
### 결과 (0.1.18 기본값: 받는 뉴런 상한 auto, 처음 세기 = 경로 중앙값, seed 6개, 짝지은 부호 뒤집기 검정)

**멀쩡한 회로에서는 이득 없음** (`lab/growth_odor.py`: 냄새 24개, PN→KC 4,000개 추가 = 실제의 30%, 추가 뒤 KC 5 Hz로
다시 보정): 실제 배선만 0.881, 무작위 0.883, 헤브 0.874, 기울기 0.890, 대조 학습 0.884 - 모두 차이 없음 (p ≥ 0.16).
상한이 없을 때 헤브 규칙은 해로웠음 (0.803, 6/6, p = 0.031 - 활발한 쌍만 이어 소수 KC가 반응을 독차지) → 상한으로 사라짐.

**손상 회복 - 냄새에서는 정답 없이도 됨** (`lab/growth_lesion.py`): PN→KC 연결 13,485개 중 80%(10,788개)를 끊고, 남은
회로에 추가 연결을 얹음. 냄새 48개를 번갈아 두 묶음으로 나눠 B(24개)로 학습·평가. 모든 조건은 연결을 정한 뒤 KC 5 Hz로 보정

| 조건 (추가 연결 수) | 고르는 기준 | 정확도 | 회복 | 무작위 대비 |
|---|---|---|---|---|
| 손상 없음 | | 0.877 | 100% | |
| 80% 손상 | | 0.823 | 0% | |
| 무작위 1,350 / 5,400 | 없음 | 0.841 / 0.856 | 33% / 61% | |
| 항상성 1,350 / 5,400 | 활동 (정답 없음) | 0.833 / 0.853 | 17% / 55% | 못함 |
| 기울기 1,350 | 과제 B 정답 | 0.865 | 76% | |
| 기울기 1,350 / 5,400 | 다른 과제 A 정답 | 0.870 / 0.891 | 86% / 126% | |
| **대조 학습 1,350 / 5,400** | **정답 없음** | **0.873 / 0.894** | **91% / 130%** | **+0.032 / +0.037 (6/6, p = 0.031)** |
| 대조 학습, SCARF 보기 1,350 / 5,400 | 정답 없음 | 0.867 / 0.893 | 81% / 128% | +0.026 (6/6, p = 0.031) / +0.036 |
| 대조 학습, 상한 없음 5,400 | 정답 없음 | 0.901 | 144% | |

대조 학습 = 같은 시료에 잡음을 두 번 따로 넣어 KC 반응 두 벌을 만들고, 각 시료가 자기 짝을 다른 시료들 사이에서 찾도록
하는 목표(InfoNCE, SimCLR과 같은 생각)의 기울기로 연결 자리를 고름. 라벨을 전혀 쓰지 않음.
`lab.grow_contrastive(g, n, inputs=X, encoder=enc, augment=0.3)` - `augment`는 곱 잡음 세기(숫자) 또는 함수
(`lab.corrupt(0.3)` = SCARF: 특징 30%를 다른 시료의 값으로). 효과가 조건부라 핵심 엔진이 아니라 `flydnet.lab`에 둠

- **정답 없이 회복**: 대조 학습 1,350개(잃은 연결의 1/8)로 손상 전과 차이 없음 (−0.005, p = 0.47)
- **순환 아님**: 데이터를 만든 잡음(곱 로그 정규)과 같은 보기 만들기를 쓰면 '답을 아는' 셈이 될 수 있어, 무관한 SCARF로도
  확인 - 같은 결과
- 손상 없는 회로에 대조 학습 5,400개를 얹으면 0.899 - 손상 회로 + 5,400 (PN→KC 8,097개)이 손상 없는 회로 + 5,400 (18,885개)과
  비슷 (0.894 대 0.899)
- 정답 없이 활동만 보는 항상성 규칙(조용해진 KC에 입력)은 무작위보다 못함 - 희소 부호에서는 대부분의 KC가 원래 조용해서,
  조용함이 입력 부족을 뜻하지 않음

**MNIST에서는 대조 학습의 이득 없음** (`lab/growth_lesion_mnist.py`, 픽셀 → 무작위 투영 → PN → KC, 학습 3,000장,
보기 만들기 = 최대 2픽셀 이동 + 밝기 잡음): 손상 없음 0.862, 80% 손상 0.812

| 조건 | 1,350 | 5,400 |
|---|---|---|
| 무작위 | 0.847 | **0.862** (회복 100%) |
| 대조 학습 (상한 auto) | 0.849 | 0.851 (무작위 대비 −0.011, 1/6, p = 0.062) |
| 대조 학습, SCARF 보기 | | 0.853 |
| 대조 학습, 상한 없음 | | 0.825 (무작위보다 확실히 못함, 0/6, p = 0.031) |
| 손상 없는 회로 + 대조 학습 | | 0.871 |

여기서는 무작위 연결만으로 회복되고 (배선이 병목이 아님), 대조 학습은 무작위를 넘지 못함. 상한이 없으면 대조 학습이 KC
상위 10%에 새 연결의 51%를 몰아 크게 나빠졌는데, 자동 상한으로 그 손해는 사라짐. 진단 (`--conds`): 이동을 빼고 밝기 잡음만,
온도 0.5, 시료 256개, 숫자마다 한 장(같은 숫자끼리 밀어내지 않게)은 모두 몰림을 고치지 못했고, 받는 뉴런 상한만 효과가 있었음.
→ **대조 학습 규칙은 입력 채널에 뜻이 있는 데이터(냄새의 사구체)에서 통했고, 무작위로 섞인 입력(MNIST 픽셀 → 무작위 투영)
에서는 통하지 않음**. 80% 손상 회로는 반응이 가팔라 보정이 진동하기 쉬움 → `calibrate`가 진동하는 그룹의 보폭을 줄여 맞춤

## 짧은 이름

긴 이름도 그대로 동작한다.

| 짧은 이름 | 원래 이름 |
|---|---|
| `fd.flywire()`, `fd.brain()`, `fd.worm()` | `Circuit.from_flywire()`, `Circuit.whole_brain()`, `Circuit.celegans()` |
| `fd.Connectome` | `fd.ConnectomeLayer` (인자 `damp` = `surrogate_damp`, `ckpt` = `checkpoint_every`) |
| `fd.Adaptive`, `fd.Glomeruli`, `fd.Inhibition`, `fd.MBON` | `AdaptivePlasticity`, `GlomerularEncoder`, `LateralInhibition`, `MushroomBodyOutput` |
| `fd.tune` | `fd.tune_surrogate` |

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
| torch 연동 | `fd.torch.bridge` — 자체 엔진 구조물을 `torch.nn` 모델 안에서 (계산은 자체 엔진) |

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
엔진 검증 (`python scripts/verify.py`): 연산 55종의 값·기울기를 torch autograd와 (꺾이는 점·동점 포함), 최적화기를
torch.optim과 한 스텝씩, 직접 작성한 LIF CUDA 커널을 일반 연산으로 짠 LIF와 (스파이크 같음, 기울기 float64 1e-7),
원본 Brian2와, CPU↔GPU를 비교 (`tests/ref_*.py`, 약 1분) + 테스트·예제 전체.

### torch 연동: fd.torch.bridge

torch 모델 안에서 flydnet 구조물을 쓴다. **계산은 자체 엔진** (원본 모델과 같은 계산, 직접 작성한 CUDA 커널),
겉은 `nn.Module`. 복사 없이 메모리를 공유하고(GPU는 DLPack), 역전파가 torch 쪽으로 이어진다.

```python
layer = fd.Connectome(mb, "PN", "MBON", trainable=True)
model = torch.nn.Sequential(encoder, fd.torch.bridge(layer, seed=0), torch.nn.Linear(48, 10))
opt = torch.optim.Adam(model.parameters())      # 커넥톰 학습 값도 torch가 갱신 (자체 엔진과 같은 메모리)
```

- 자체 엔진만 쓸 때와 출력·기울기가 같다 (테스트). 연결 장치 자체의 비용은 거의 없음 (전체 뇌 학습 1스텝 배치 8에 1.1초)
- `examples/torch_bridge.py`, 시각계 예제 (`examples/visual_motion.py`: torch 학습 루프 + 자체 엔진 층)

0.1.17까지 있던 0.1의 torch판 복사본(`fd.torch.ConnectomeLayer` 등)은 0.1.18에서 뺐다. 같은 이름이 `fd.`에 있다
(`fd.torch.RateEncoder` → `fd.RateEncoder`). torch 학습 루프를 그대로 쓰려면 층을 연결 장치로 감쌀 것.

## 결과 요약

12개 실험 (재현: `examples/`의 각 스크립트):

| 질문 | 답 |
|---|---|
| 실제 배선이 정확도를 높이나 | 대체로 아니다. 차수와 큰 구조(위치 대응)를 유지한 무작위 배선과는 MNIST·냄새·시각 과제에서 차이 없음 |
| 그럼 배선의 무엇이 중요했나 | 큰 구조: 시각계의 위치 대응(없으면 학습 불가), 스파이킹 버섯체의 KC 연결 수 분포(무작위면 −13%p) |
| 도파민 연합 학습은 | 연속 학습에서 쓸모 있음: CIFAR-100 10과제 57.5% (재생 버퍼 역전파 51.4%), 역전파 없이 한 번 보기. 이득은 KC 배선이 아니라 학습 규칙에서 나옴 (같은 규칙을 원래 특징에 쓰면 57.5%, KC 실제 배선 위에서는 48.9%) |
| 역전파도 덜 잊게 할 수 있나 | 연합 규칙의 생각 세 가지(새 클래스는 평균에서 시작, 나온 클래스끼리만 경쟁, 코사인 점수)로 CIFAR-100 24.7% → 58.5% (재생 없음). `fd.Learner(rule="backprop")` |
| KC 확장 층은 | 선형 리드아웃일 때만 도움 (XOR형 과제 +19%p) |

## 개발

```bash
python -m venv .venv
.venv\Scripts\pip install --no-cache-dir -e ".[examples,dev,gpu-cuda13]"
.venv\Scripts\python scripts\build_csr.py --all         # _csr.c를 고쳤을 때만: 6개 플랫폼 C 커널 (zig, 시험용)
.venv\Scripts\python -m pytest -q                       # 테스트 (tests/test_golden.py = 엔진 정리 전 결과와 비교)
.venv\Scripts\python scripts\release.py bump patch      # 버전 올리기 + CHANGELOG 틀
.venv\Scripts\python scripts\release.py check --upload  # 커밋·CHANGELOG·PyPI 중복·테스트·빌드·설치 확인 후 업로드
```

배포는 순수 파이썬 휠 하나 (6개 플랫폼 C 커널 포함). `release.py check`가 커널이 지금 `_csr.c`로 빌드된 것인지 확인한다.
배포 전 플랫폼 확인: GitHub Actions → wheels → Run workflow (5~10분, Linux·macOS·Windows에서 골든 시험).

데이터 출처: FlyWire v783 연결(Shiu et al. 2024, MIT), 세포 주석(Schlegel et al. 2024), DoOR 2.0(CC BY-SA 4.0).
MIT 라이선스.
