# 변경 기록

## [0.1.18] - 미배포

### 엔진 정리: scipy·pyarrow 없이 (필수 의존성을 줄임)
- `import flydnet`이 scipy·CuPy를 불러오지 않음 (처음 쓸 때 불러옴 - CuPy의 느린 초기화도 GPU를 쓸 때만).
  `Circuit.to_networkx`의 networkx는 선택 설치 `pip install "flydnet[graph]"` (없으면 안내)
- FlyWire 연결 parquet를 numpy 형식 `Connectivity_783.npz`(pre, post, weight)로 한 번 바꿔 두고 그것을 읽음 → 회로를 만드는 데
  pyarrow가 필요 없음. `download`가 만들고, 예전에 받아 둔 사람은 parquet를 처음 읽을 때 만들어짐 (약 10초, 52 MB).
  배열 내용의 SHA-256으로 확인 (`python -m flydnet verify`). npz만 있어도 됨 (parquet 없이 복사해 와도).
  둘 다 읽을 수 없으면 무엇을 하면 되는지 알림
- scipy를 쓰던 주변 기능을 numpy로: `fd.compare`의 95% 신뢰구간 (t 분위수 표 + 전개, scipy와 차이 2e-6 이하),
  `layer.reach`의 홉 수, `fd.column_map`의 이웃 평균 (결과 비트까지 같음), `python -m flydnet doctor`의 커널 확인.
  `Circuit.from_scipy`·`to_scipy`는 scipy를 그대로 씀 (쓸 때만 불러옴)
- **CPU 희소 행렬 곱을 직접 작성한 C 커널로** (`flydnet/ganglion/csr.py`, `_csr.c`): 엔진의 CPU 계산이 scipy 희소 행렬 대신
  자체 CSR (indptr int64, indices int32)을 씀. 행을 파이썬 스레드로 나눠 C 함수를 부름 (OpenMP 없음 → 플랫폼마다 런타임을
  배포할 필요 없음, 스레드 수와 상관없이 같은 비트). 시각계 전체(연결 429만, 배치 32): scipy 17.5 ms → 2스레드 6.6 ms,
  8스레드 2.8 ms. 경로 순서: C 커널 → scipy (있으면) → numpy (느림, 한 번 경고). `FLYDNET_SPARSE=c|scipy|numpy`,
  `FLYDNET_THREADS=n`. 세 경로 모두 행마다 연결 순서대로 더해 같은 비트 (골든 시험이 세 경로에서 비트까지 같음).
  역전파용 전치 구조는 배선마다 한 번만 만듦

### 버그 수정
- **`fd.gradcheck`가 학습값 배열을 사본으로 바꿔 끼우던 것**: 되돌릴 때 원래 배열이 아니라 사본을 넣어, 같은 메모리를 쓰던
  `fd.torch.bridge`(torch 옵티마이저)와의 연결이 끊김 → 제자리로 바꿨다 되돌림
- **`fd.torch.bridge`로 감싼 뒤 학습값이 생기면 (확장을 붙이는 등) torch가 '기울기 개수가 틀림' 오류**: 이름 목록을 다시 만들고
  순전파 맨 앞에서 맞춤 (옵티마이저를 다시 만들라는 경고는 그대로)
- **`fd.train`에 손실을 직접 주고 (B, T) 라벨이면 모든 에폭을 마친 뒤 오류로 멈춰 결과를 잃던 것** (0.1.18의 라벨 검사) →
  train_acc는 라벨이 시료마다 하나일 때만 (아니면 None)
- `fd.graphs.watts_strogatz`: 다시 잇기에서 자기 자신·이미 있는 상대를 뽑으면 나중에 버려 연결이 n·k개보다 적었음
  (beta 1에서 1.3% 적음) → 피해서 다시 뽑음 (networkx와 같음, 같은 seed의 그래프가 바뀜 - `examples/any_graph.py` 수치도)
- `fd.STDP`·`fd.Monitor`를 연속값 뉴런 층에: 그 순전파는 관찰자를 부르지 않아 STDP는 알기 어려운 오류, 직접 만든 관찰자는
  조용히 아무것도 안 함 → 만들 때 알림
- `fd.compare`: 점수가 0~1이 아니면 (발화율 Hz 등) "과제가 너무 쉬움" 상한 경고를 내지 않음 (예전: 늘 붙었음)
- `fd.genetics.screen`: 빈 집단 목록·조작 뒤 측정값 NaN을 어느 집단인지 알림, `driver(missing=)` 오타는 오류
- `fd.KCExpansion`: 빈 입력에 빈 출력 (예전: concatenate 오류), `fd.neurons.Izhikevich`: a·b·c·d NaN 확인
- 데이터 받기: 받는 도중 연결이 끊기는 IncompleteRead도 안내 오류로 (예전: 원래 오류 그대로)
- `Circuit.from_flywire`: 주석의 같은 뉴런이 두 번이면 버림 (whole_brain과 같게 - 지금 데이터에는 없음)

- **그룹끼리 뉴런이 겹치는 회로로 만든 층·Neuropil이 조용히 틀린 계산을 하던 것**: 겹친 입력 뉴런이 입력 자리에 두 번 들어가
  순전파는 마지막 값만, 역전파는 두 자리 모두에 기울기 (Neuropil은 첫 자리 값을 버림), 연결 종류 번호도 마지막 그룹으로.
  `fd.Circuit`은 겹침을 허용하므로 (확인은 `circuit.check()`) 층·Neuropil이 만들 때 알림. 제공하는 회로(버섯체·전체 뇌·
  시각계·선충·합성 그래프)는 겹치지 않음
- **`ConnectomeLayer.load`가 저장 파일에 적힌 모듈·클래스를 그대로 불러 실행하던 것** (0.1.18의 확장 자리): 남이 만든 파일로
  코드가 돌 수 있었음 (pickle과 같은 위험) → flydnet 안의 확장(Tissue + extension_config)만 다시 붙임
- `where(조건, a, b)`: GPU 조건 + CPU 신호에서 TypeError → 조건을 신호의 장치로
- `ConnectomeLayer(trainable="PN>KC")` (문자열 하나): 글자마다 연결 종류로 봐서 엉뚱한 오류 → 연결 종류 하나로
- `fd.flywire(side="both")` 등 없는 side가 "그룹이 모두 비었음"으로 보이던 것 → 쓸 수 있는 값 안내 (양쪽은 side=None)

- **`fd.sign_flip_p`에 NaN이 섞이면 p = 0 (유의)으로 나오던 것**: 평균이 NaN이 되어 모든 비교가 거짓 → 실패한 seed 하나가
  "6/6 유의"처럼 보였음. 이제 NaN·무한대는 오류, (n, 1)은 폄 (`compare`·`genetics.screen`도 이 함수를 씀)
- **`Learner.learn`이 배우다 실패하면 새 라벨만 남던 것**: 나중에 그중 일부만 배우면 backprop의 빈 클래스 가중치가 빈 평균(NaN)
  → 실패하면 새 라벨을 되돌림
- `fd.evaluate`: 출력이 (시료, 클래스)가 아니면 오류 - (배치, 시간, 클래스)면 argmax가 (배치, 시간)이 되어 시간 길이 = 배치일 때
  라벨과 퍼져 조용히 틀린 정확도
- `Learner(rule="backprop")` 저장에 학습 관성(Adam)도 - 불러온 뒤 이어 배우면 저장 안 한 것과 같은 결과 (예전: 점수가 0.08 다름)
- `AssocReadout`·`DopamineReadout.fit`에 pandas: 행 이름으로 골라 KeyError → numpy로
- `lab.grow_contrastive`: 상한이 차서 아무것도 안 만들 때도 인자(tau·samples·candidates·augment)를 확인
- `python -m flydnet --help`가 실패(종료 코드 1)로 끝나던 것 → 0. 모르는 명령은 무엇이 틀렸는지 알림
- 알기 어려운 오류 대신 안내: `graphs.stochastic_block`에 사전이 아닌 sizes, `Circuit.regroup`에 노드 번호가 아닌 값

- **(n, 1) 모양 라벨에서 정확도가 조용히 틀리던 것**: scikit-learn 습관대로 라벨을 열 벡터로 주면 예측 (n,)과 비교할 때
  (n, n)으로 퍼져 `fd.evaluate`가 23.0 같은 값을, `AssocReadout.accuracy`가 다른 정확도를 냄 (`DopamineReadout.fit`·`fd.train`·
  `train_linear`은 알기 어려운 오류). 이제 분류용 함수 모두 (n, 1)을 펴고, 그 밖의 2차원 라벨은 알기 쉬운 오류
  (`fd.train`에 손실을 직접 주면 (B, T) 같은 라벨은 그대로)

- **`grow`·`prune`이 잘못된 인자를 조용히 넘기던 것**: 추가 연결 상한이 찼거나 추가 연결이 없으면, 필요한 인자(`rates`·`loss`·
  `inputs`)가 빠졌거나 `prune(frac=5)`처럼 범위 밖이어도 0개를 돌려주며 넘어갔음 → 인자 확인을 먼저

- **`calibrate`가 진동하다 목표에 못 닿던 것**: 반응이 가파른 회로(예: PN→KC 연결 80%를 끊은 버섯체)에서 보폭 0.5로 두 배율
  사이를 오가다 끝나 KC 8.1 Hz (목표 5)로 남았음. 지금은 목표를 넘었다 못 미쳤다를 연속 두 번 오가는 그룹만 보폭을 반으로 줄이고,
  그래도 못 닿으면 반복 중 가장 가까웠던 배율로 되돌림. 진동 없이 수렴하던 보정은 예전과 같은 배율 (손상 없는 버섯체 KC·MBON 확인)

- **한 줄 학습기 `fd.Learner`**: 커넥톰 고정, 데이터 한 번 훑기로 학습, 새 데이터는 이어서 (앞에서 배운 것 유지).
  입력 변환·연결 세기 보정은 처음 데이터로 자동 (보정은 한 번만 - 나중에 바꾸면 기억과 특징이 어긋남)
  - `learn(X, y)`, `predict`, `score`, `features`, `save`·`Learner.load` (불러온 뒤에도 계속 배움). 라벨은 문자열 등 아무 값
  - `source="이름"`: 특징 수가 다른 데이터는 입력 변환만 따로, 커넥톰·기억은 공유
  - `rule="lda"` (기본): 흐름 선형 판별 (streaming LDA, Hayes & Kanan 2020) - 클래스 평균 + 공유 공분산을 정확히 누적
    (묶음·순서와 무관), 공분산 축소 자동 (max(OAS, 특징 수 / 시료 수)). 차례로 배우기: 냄새 24개 0.859 (assoc 0.744,
    한 번에 0.865), MNIST KC 특징 0.896 (assoc 0.728), CIFAR-100 resnet18 특징 0.640 (assoc 0.539, 선형 상한 0.658)
  - `rule="assoc"`: AssocReadout과 같은 도파민 연합 학습 - 다른 클래스 기억은 한 비트도 안 바뀜
  - `rule="backprop"`: 망각을 줄인 역전파 - 새 클래스 가중치를 그 클래스 평균 특징으로 시작, 이번 learn()에 나온 클래스끼리만
    경쟁, 코사인 점수 (특징 중심은 처음 데이터로 고정). class-incremental CIFAR-100 (resnet18 특징, 과제당 1에폭, 재생 없음)
    24.7% → 58.5% (연합 53.9%), MNIST KC 특징 41.9% → 65.6% (연합 72.8%)
  - DoOR 냄새 24개를 6개씩 차례로: lda 85.9%, 연합 74.4%, 역전파 70.8%. 한 번에 전부: lda 86.5%, 연합 71.2%,
    역전파 84.1% (seed 3개)

- **GPU 연결 전달이 크게 빨라짐 (결정론 유지)**: 긴 행(입력이 많은 뉴런)을 연결 64개씩 조각내 동시에 계산하고 정해진
  순서로 더하는 커널. 예전 커널은 행 하나를 스레드 한 묶음이 끝까지 맡아, 버섯체 APL(입력 2,597개)·MBON처럼 긴 행 하나가
  끝날 때까지 나머지가 기다렸음 (배치가 클수록 심함 - torch판보다 2배 느렸던 원인)
  - 연결 전달 1번: 버섯체 배치 256 970 → 102 µs, 배치 1 82 → 22 µs, 전체 뇌 배치 8 1.2 → 0.4 ms
  - 버섯체 순전파 (100 ms, 배치 256): 1,048 → 148 ms (예전 torch판 553 ms)
  - 결정론적: 같은 입력이면 늘 같은 비트 (cuSPARSE는 실행마다 반올림이 달라 쓰지 않음). 짧은 행(조각 하나)은 예전과
    비트 단위로 같고, 긴 행은 덧셈 순서가 바뀌어 반올림 수준(1e-4)으로 다름
- **GPU에서 결과가 실행마다 미세하게 달라지던 곳** (cuSPARSE를 그대로 쓰던 곳): `Neuropil`의 전달·입력 기울기,
  `ThreeFactor(feedback="connectome")`의 오차 전파 → 위 결정론적 커널
- **[실험적] 추가 연결 (구조적 가소성, `growth`)** - 인터페이스·기본값이 바뀌거나 없어질 수 있음, 효과는 조건부: 실제 배선(고정)은 그대로 두고, 학습으로 생기고 없어지는 시냅스를 따로 얹음.
  타고난 회로 위에 경험으로 시냅스가 덧붙고 없어지는 뇌의 방식 - 크기·연결이 고정이던 커넥톰 층의 한계를 풀면서 실제 배선은
  늘 기준으로 남음. 엔진에는 들어 있지 않고 `flydnet.lab`의 부품: `g = lab.Growth(layer, allow=["PN>KC"], budget=5000)`
    (층에 붙어 `layer.growth`, 저장·불러오기 때 같이 다시 붙음)
  - `g.grow(n, rule=...)`: `"random"` (허용된 그룹 쌍 안에서), `"coactive"` (헤브 - 함께 많이 발화한 뉴런 쌍),
    `"homeostatic"` (항상성, 정답 없이 - 그룹 평균보다 덜 발화하는 받는 뉴런이 모자란 만큼 새 입력을 받음, 보내는 뉴런은 무작위),
    (정답 없는 대조 학습 성장은 엔진이 아니라 `flydnet.lab.grow_contrastive` - 라벨 없는 데이터 `inputs`를 `augment`로 두 번
    따로 흔들어(`encoder`가 있으면 그다음 변환) 각 시료의 출력이 자기 짝을 찾도록 하는 대조 손실(InfoNCE, `lab.info_nce`)의
    기울기로 `rule="gradient"`처럼 고름),
    `"gradient"` (세기 0으로 넣었을 때 손실이 가장 줄 자리, RigL과 같은 생각). `g.prune(frac)`·`prune(below=)`는 추가
    연결만, `g.extra_edges()`는 표
  - 생물학적 제약: 허용된 그룹 쌍(기본 = 실제 배선에 있는 쌍) 안에서만, 이미 있는 연결·자기 연결 없음, 부호 = 보내는 뉴런의
    실제 부호 (데일의 법칙), 전체 상한 `budget`, 받는 뉴런마다 상한 `per_neuron`
  - 받는 뉴런마다 상한 `per_neuron` 기본 `"auto"` = ceil(budget / 받을 수 있는 뉴런 수) (SRigL의 일정한 fan-in). 상한이
    없으면 기울기·헤브 규칙이 소수 뉴런에 연결을 몰아 그 뉴런들이 반응을 독차지 - MNIST 손상 회복에서 대조 학습이 KC 상위
    10%에 51%를 몰아 0.825, 상한 3이면 0.851 (무작위 0.861). `per_neuron=None`이면 상한 없음
  - 세기 = 시냅스 `init_syn`개 x 그 경로의 배율(gains) x exp(학습값) - 추가 시냅스 1개 = 같은 경로의 실제 시냅스 1개.
    `init_syn` 기본 `"median"` = 그 그룹 쌍 실제 연결의 시냅스 수 중앙값 (버섯체 PN>KC 10개). 처음엔 시냅스 1개였는데,
    실제 연결의 1/10이라 추가 연결 4,000개를 합쳐도 PN>KC 입력의 2.7%뿐이고 학습으로도 거의 안 커져 (중앙값 1.00~1.02개)
    결과에 영향이 없었음
  - 같은 지연, 같은 효과기(silence·block·activate·mosaic), 체크포인팅, 저장·불러오기(연결 수가 달라도), CPU·GPU, torch 연결 장치.
    추가 연결이 없으면 계산이 전과 똑같음 (엔진 비트 비교). ThreeFactor·STDP는 고정 배선만 학습 (알림)
  - 검증: 추가 연결을 얹은 층 = 그 연결을 처음부터 회로에 넣어 만든 층 (출력·입력 기울기·연결 기울기, LIF·graded, CPU·GPU)
  - 가소성 규칙: prune·grow 뒤에도 살아남은 추가 연결의 관성·적응 상태(Adam 단계 수 포함)는 그 연결을 따라가고 새 연결만
    0에서 시작 (연결 수가 같아도 자리가 바뀌면 옮김). 새 연결의 첫 변화량은 다른 연결과 같은 크기 (단계 수가 연결마다)
  - 예제 `lab/growth_odor.py` (멀쩡한 회로): 실제 배선만 대 무작위·헤브·기울기·대조 학습 추가 연결 (에폭마다 약한 20%
    없애고 다시 채움, 처음 채운 뒤 KC 5 Hz로 다시 보정, seed 6개). 이득 없음 (실제 배선만 0.881, 무작위 0.883, 헤브 0.874,
    기울기 0.890, 대조 학습 0.884 - 모두 p ≥ 0.16). 상한이 없으면 헤브는 해로웠음 (0.803, 6/6, p = 0.031)
  - 예제 `lab/growth_lesion.py` (손상 회복): PN→KC 연결 80%를 끊으면 0.877 → 0.823 (냄새 48개를 A·B로 나눠 B로 평가,
    seed 6개). **정답 없이** 대조 학습(`lab.grow_contrastive`)으로 고른 추가 연결 1,350개(잃은 연결의 1/8)로 0.873 - 손상 전과
    차이 없음, 무작위 1,350개 0.841보다 +0.032 (6/6, p = 0.031). 데이터를 만든 잡음과 무관한 SCARF 보기로도 같음 (순환 아님).
    다른 과제 정답으로 고른 기울기 연결도 비슷 (0.870 / 5,400개 0.891), 활동만 보는 항상성 규칙은 무작위보다 못함
  - 예제 `lab/growth_lesion_mnist.py` (MNIST 재현): **대조 학습의 이득 없음** - 80% 손상 0.862 → 0.812, 무작위 5,400개로
    0.862 (회복), 대조 학습 0.851. 상한이 없으면 소수 KC에 몰려 0.825로 무작위보다 확실히 못했음 → 받는 뉴런 상한 auto로
    손해는 사라짐. 입력 채널에 뜻이 있는 냄새에서는 통하고, 무작위로 섞인 MNIST 입력에서는 통하지 않음
- **`flydnet.lab` (실험실)과 `lab/` 폴더**: 효과가 조건부이거나 증명되지 않은 것을 핵심에서 분리. `from flydnet import lab`로
  따로 불러야 씀 (최상위 `fd.`에는 없음), 인터페이스가 바뀌거나 없어질 수 있음. 엔진에는 확장 자리 `ConnectomeLayer.attach(name,
  부품)`만 있음 - 부품이 순전파마다 추가 경로를 내고(`paths()`), 배율 변화(`on_build()`)·학습 신호 필요 여부·표시·저장 인자를
  알려 줌. 붙인 것이 없으면 엔진 계산은 전과 같음 (엔진 비트 비교). 지금: 추가 연결 `lab.Growth`, 대조 학습 성장 `lab.grow_contrastive`,
  `lab.info_nce`, `lab.contrastive_loss`, 보기 만들기 `lab.views`·`lab.corrupt` (SCARF). 실험적 기능의 연구 예제는 `lab/`
  (`growth_odor`·`growth_lesion`·`growth_lesion_mnist`) - 전체 검증에 기본으로 들어가지 않음 (`verify.py --lab`로 함께).
  결론이 안정된 연구 재현 예제 `door_assoc`·`odor_ablation`도 `lab/`으로 - 다른 예제가 이미 실행하는 코드만 쓰고(고유 함수
  0~2개, 단위 테스트가 확인) 버그를 잡은 기록이 없어 배포 검증에서 뺌 (전체 검증 순차 기준 약 67분 줄어듦)
- **torch판 복사본 제거 (`fd.torch.ConnectomeLayer`, `fd.torch.RateEncoder` 등 0.1의 torch판 전부)**: 같은 기능이
  flydnet 최상위에 있고, 계산 방식이 둘(legacy·brian)이라 버그가 쌍둥이로 나고(이번에도 RateEncoder·KCExpansion) 관리
  부담이 컸음. **`fd.torch.bridge`는 그대로** - torch 모델 안에서 flydnet을 쓰는 방법은 이것 하나 (계산은 자체 엔진).
  - 옮기는 법: `fd.torch.이름` → `fd.이름` (결과는 numpy·Signal). torch 학습 루프를 그대로 쓰려면
    `fd.torch.bridge(fd.ConnectomeLayer(...))` (예: `examples/visual_motion.py`, `examples/torch_bridge.py`)
  - 예제 13개를 자체 엔진으로 옮김. 계산이 legacy → brian(원본 Brian2와 같은 계산)으로 바뀌어 숫자가 조금 바뀜
    (README 결과 갱신). 같은 배율에서 KC 활성 비율이 조금 높음 (PN>KC 3.0: 약 6% → 8%)
  - MNIST 특징 캐시는 `.npz` (`feat_..._brian_*.npz`), CIFAR-100 특징 캐시도 `.npz` (예전 `.pt`가 있으면 한 번 옮김)
  - `Circuit.to_dict`·`from_dict` 제거 (torch판 저장에만 쓰던 것. 저장은 `layer.save`의 np.savez)
  - 테스트: torch판과 값을 비교하던 검사를 독립 기준으로 바꿈 - legacy LIF·graded 뉴런은 torch 기본 연산으로 다시 짠
    참조 구현(`tests/_ref_models.py`)과 값·기울기 비교, 인코더·KC 확장·시각 도구는 수식으로 직접 계산한 값과
- **입력 → 출력 경로를 찾을 때 입력으로 되돌아가는 우회와 출력을 지나 도는 고리를 경로로 세던 것** (세 곳이 같은 원인).
  버섯체에는 PN → MBON → PN 흥분성 고리가 있어서, PN → KC 층에서 MBON이 "KC로 가는 중계"로 잡혔음:
  - `layer.calibrate(R, {"KC": 5})`가 목표에 없는 MBON의 들어오는 연결까지 2배로 키움 (MBON 0 → 20.6 Hz) - 회로가 의도와
    다르게 바뀜. `fd.MushroomBody`·`fd.ConnectomeModel`의 자동 보정도 같음
  - `trainable="path"` (`fd.ConnectomeModel` 기본)가 KC 출력 모델에서 출력과 상관없는 `PN>MBON`·`KC>MBON`까지 학습
    (이제 `PN>KC`·`KC>KC`만)
  - `layer.reach`가 출력 KC가 거의 0 Hz인데 "신호가 MBON에서 끊김"으로 엉뚱한 그룹을 지목
- `Reach.break_at`: 출력이 발화하면 꺼진 중계가 있어도 None (다른 경로로 신호를 받음 - 끊긴 곳이 아님). 꺼진 중계는 따로 알림
- **`fd.compare`의 해석이 비교하지 않은 단계를 건너뛰고 원인을 단정하던 것**: `controls=["shuffled"]`만 비교해 실제 배선이
  이기면 위치 구조일 수도 있는데 "중요한 구조: 세부 배선"으로 말했음 → 사이 단계 구조를 모두 후보로 ("다음 중 하나 이상")
- `fd.sign_flip_p`: seed 15개 이상(표본 순열)에서 p = 0이 나올 수 있던 것 → (맞은 수 + 1) / (표본 수 + 1)
- `fd.graphs.barabasi_albert`: 새 노드마다 m개라고 했지만 중복을 버려 연결이 약 3% 적었음 → 서로 다른 m개
  (같은 seed의 그래프가 바뀜, `examples/any_graph.py` 수치도)
- `fd.explain(verify=...)`: `fd.genetics.training()` 안에서 부르면 확인(실제로 끄기)할 때만 mosaic 드롭아웃이 켜져
  예측과 다른 모델을 비교했음
- **`fd.KCExpansion(n_in=...)`이 특징 20개 이하에서 입력 정보를 전부 잃던 것**: PN마다 특징 k_in(20)개
  평균인데 k_in이 특징 수 이상이면 모든 PN이 같은 값 → 평균 빼기 뒤 0 → 입력과 상관없이 같은 KC 코드 (특징 16개, 시료 6개에
  코드 2가지, 값 1e-5). 0.1.17의 RateEncoder 수정과 같은 규칙: k_in은 특징 수의 절반 이하 (특징 40개 이상이면 예전과 같음)
- `load_state`가 버퍼 모양을 확인하지 않던 것: 다른 크기로 만든 구조물(RateEncoder의 투영 P, Neuropil 등)의 상태를 불러오면
  조용히 바꿔 끼워 출력 크기가 달라졌음 → 학습 값처럼 먼저 확인하고 오류
- `python -m flydnet verify`: 묶음을 안 주면 기본으로 받지 않는 worm까지 확인해 정상 설치에서도 missing·종료 코드 1
  → 기본 묶음과 받아 둔 묶음만
- **시료 하나 (n,)를 넣으면 오류로 죽던 곳** (층·Projection은 받는데): `model.predict`, `layer.reach`, `layer.calibrate`,
  `Neuropil`, `DopamineReadout.predict`·`fit(리스트)` → 시료 하나는 (1, n) 묶음으로, predict는 클래스 번호 하나
- `layer.gains["PN>KC"] = 2`처럼 직접 바꾸면 저장 파일에만 들어가고 순전파는 예전 세기 그대로였던 것 → 다음 순전파에 반영
  (저장 전후 동작이 달랐음)
- `fd.compare`·`genetics.screen`: 같은 seed를 두 번 주면 같은 짝을 두 번 세어 p가 작아짐 (유사 반복) → 오류
- `fd.ConnectomeModel(target_hz="auto")`: 시행당 출력 스파이크 수를 t_ms 전체로 계산 → 스파이크를 세는 시간
  (t_ms - count_from_ms)으로 (count_from_ms를 쓸 때만 달라짐)
- **`Signal(불리언) + 숫자`가 논리합이 되던 것**: 숫자를 신호의 불리언형으로 바꿔 `[True, False] + 1`이 `[True, True]`
  (numpy는 `[2, 1]`). 비교 결과(`s > 0`)를 세거나 더할 때 조용히 틀렸음
- `MushroomBodyOutput`·`kenyon_code`: 시료 하나 (n,)를 (n, 1)로 봄 (위 DopamineReadout과 같은 패턴). predict는 클래스 번호 하나
- 예제: `genetics_sugar.py`의 `rate_100Hz` 열에 발화율 대신 무자극 대비 증가량이 들어가던 것 (따로 `rise_100Hz`),
  `any_graph.py` 설명문 (0.1.17부터 출력 그룹 평균 발화율로 맞춤)
- **검증 도구** (`scripts/verify.py`, `tests/ref_*.py`):
  - 참조 검사 10종 (약 1분): Signal 연산 55종의 값·기울기 대 torch autograd (꺾이는 점·동점·경계 포함), 역전파 엔진
    구조, 최적화기 대 torch.optim (단계별), 내장 LIF 커널 대 일반 Signal 연산으로 짠 LIF (출력 스파이크 같음, 기울기
    float64로 1e-7), legacy·graded 대 참조 구현, 층 대 torch nn, ThreeFactor 출력 쪽 = 역전파 (31조합), STDP = 직접 센
    쌍 기반, 무작위 대조군 불변량, 회로·데이터 대 원본 파일 직접 집계, CPU = GPU·torch 연결 장치
  - `scripts/verify.py`: 빠른 것부터 실패하면 멈춤, 테스트를 둘로 나눠 동시에, 예제를 GPU 메모리를 보며 병렬로.
    `--changed`는 실제로 실행되는 flydnet 함수의 소스가 바뀐 예제만 다시 (함수 단위 지문, 주석만 바꾸면 그대로)
  - `tests/snapshot.py`: 엔진 비트 단위 비교 (기준 `.verify/snapshot_base.npz`)
- 테스트 강화: graded 음수 입력이 실제로 전달되는지, `fd.train`이 연결을 바꾸고 손실을 줄이는지 (예전엔 오류가 없는지만)
- 테스트: 자식 프로세스의 한글 출력을 콘솔 인코딩(cp949)으로 읽다 실패하던 것 (환경 변수에 따라. 0.1.17에서 원인을 못 찾은 간헐 실패도 이것으로 보임)

## [0.1.17] - 2026-10-05

- **전체 코드 검토 (3회 + 5회, 회차마다 다른 관점)**. 고친 뒤마다 이전 커밋과 82개 배열 비트 단위 비교·옵션 조합 시험·
  전체 테스트, 마지막에 예제 22개·원본 Brian2 검증:
  - **합친 GPU 커널이 다른 집단을 자극하던 것**: "뉴런 → 입력 행" 역표를 GPU 주소로 캐시해서, 해제 뒤 같은 주소에 만든 새
    목록이 옛 역표를 받음 - 크기가 같은 집단을 차례로 activate하면 두 번째 A가 B를 자극 (이 버전에서 생겼다가 고침, 0.1.16엔 없음).
    옵션 조합 시험(`tests/combo_check.py`)을 테스트에 넣음 - 하나씩만 시험하면 안 보이는 버그용
  - torch 연결 장치: x + y처럼 같은 배열이 두 입력의 기울기가 되어 .grad가 메모리를 공유 (두 번 누적하면 2 대신 3).
    load_state가 학습 값 배열을 바꿔 끼워 이미 만든 torch 옵티마이저가 쓰이지 않는 값을 갱신 (학습이 반영 안 됨) → 제자리에 덮어씀
  - **만든 뒤 바꾼 속성이 저장에서 빠지던 것** (`layer.t_ms = 80`, `noise`, `slope`, `count_from_ms`, `p[...]`): save가 만들 때의
    config를 써서 불러온 층이 조용히 다르게 동작. `p["w_syn"]`을 바꾸면 입력 자극에만 반영되고 연결 세기는 그대로 (55 대 5 Hz)
  - STDP가 Shibire(block)된 뉴런의 발화를 못 봐서 그 연결이 학습되지 않던 것 → 관찰자에 실제 발화(`fired`)도 넘김
  - ThreeFactor를 만든 뒤 `layer.to(...)`로 옮기면 실패하던 것
  - 실패·중단 시 상태가 반쯤 바뀐 채 남던 것: `tune`(감쇠 값), `calibrate`(배율), `load_state`(모양이 틀린 항목이 뒤에 있으면 앞 값만 바뀜)
  - `reach`: 경로가 없어 늘 0인 출력을 빼고 "출력까지 신호가 감"으로 판정하던 것, activate로만 자극하는 층에서 출발점이
    없던 것 (`calibrate`의 중계 그룹도 자극한 그룹에서 찾음)
  - 0.1.16 이전에 Pathway 등에 넣어 저장한 파일은 보정 배율이 없어 보정 전 세기로 돎 → 불러올 때 알림
  - `release.py`: CHANGELOG 날짜가 "미배포"면 멈춤, 배포 전 테스트도 경고를 실패로. `python -m flydnet download 오타`가
    트레이스백 대신 안내
  - 확인만 함 (맞음): graded 층 역전파 = float64 수치 미분 6e-8, 입력 자료형 5가지 CPU·GPU 같음, 저장 가능한 객체 37가지 x 장치
    조합 왕복, 0.1.16 ConnectomeLayer 파일 → 같은 출력, ConnectomeModel이 compare에서 GPU로도 재현됨

- **연산 버그** (회귀 테스트 `tests/test_017.py`):
  - **`transpose`에 음수 축을 주면 역전파가 틀린 모양의 기울기를 쌓던 것** (`(2, 3)` 신호에 `(3, 2)` 기울기) - 역순열을
    음수 축 그대로 계산했음. `transpose((1, 0))`처럼 튜플 하나도 받음, 같은 축 두 번은 오류
  - **`s == 0`·`s != 0`이 원소별 비교가 아니라 객체 비교라 늘 `False` 하나**이던 것 → numpy처럼 원소별
    (`(spk == 1).sum()` 같은 코드가 조용히 틀렸음). 딕셔너리·집합 키로는 예전처럼 신호 객체 그대로
  - `x ** 0`의 기울기가 x = 0에서 NaN → 0. `숫자 ** 신호` 지원 (예전: TypeError)
  - `@`가 1차원도 numpy처럼: `(n,) @ (n, k)`, `(m, n) @ (n,)`, `(n,) @ (n,)` (예전: 오류)
  - **GPU 신호의 `clip`에 numpy 배열 경계를 주면 TypeError** (CPU에서는 됨) → 장치로 옮김. Signal 경계도 받음
  - `concat`에 numpy·cupy 배열을 섞으면 실패하던 것, 장치가 다르면 알기 쉬운 오류
- **GPU 속도: 한 스텝을 합친 CUDA 커널** - 시간을 재 보니 작은 회로는 계산이 아니라 스텝마다 작은 GPU 연산 20~30개를
  띄우는 비용이 대부분이었음 (버섯체 1,000스텝 중 입력 난수 166 ms, LIF 원소별 295 ms). 커널 세 개로 합침:
  포아송 입력 스파이크 (난수·비교, 예전 연산 약 12개), LIF(`timing="brian"`) 한 스텝 순전파, 그 역전파.
  모든 실수 연산을 따로 반올림(`__fmul_rn` 등, FMA로 합치지 않음)해서 **기존 결과와 비트 단위로 같음** (출력·연결 기울기·
  입력 기울기, 활성화·mosaic·잡음·체크포인팅·절단·ThreeFactor 14가지 설정에서 확인, 원본 Brian2 검증 그대로)

  | GPU, 배치 8 | 예전 | 지금 |
  |---|---|---|
  | 버섯체 순전파 1,000스텝 | 600 ms | 164 ms (3.7배) |
  | 버섯체 학습 1스텝 | 1,148 ms | 508 ms (2.3배) |
  | 전체 뇌 순전파 200스텝 | 326 ms | 274 ms (1.2배, 시냅스 전달이 대부분) |
  | 전체 뇌 학습 1스텝 | 1,376 ms | 1,179 ms (1.2배) |

  - 연결별 기울기 커널의 전환 기준을 배치 16 → 48: 배치 16·32에서는 워프 합산보다 연결마다 스레드 하나가 4배·1.4배 빠름
    (전체 뇌, 배치 32에서 스텝당 5.3 → 3.8 ms)
  - 세포 유형별 매개변수(`bias`·`t_mbr`·`train_neurons`), CPU, float64는 예전 경로 그대로
- **역전파 메모리 3.8배 감소·속도 개선** (결과는 비트 단위로 같음 - 14가지 설정 x CPU·GPU의 출력·연결 기울기·입력 기울기
  82개를 변경 전 커밋과 대조). 시간을 재 보니 전체 뇌 역전파는 계산보다 메모리가 문제였음: 순전파가 역전파용으로
  원소당 약 34바이트를 붙잡아 200스텝에 6.9 GB (필요한 것은 약 7바이트) → GPU 메모리가 차면 할당을 되풀이하며 느려지고
  배치를 키울 수 없었음
  - 소비가 끝난 중간 신호(막전위·시냅스 전류·스파이크·도착 입력·발화 수 누적)의 값을 놓음 - 역전파 엔진은 중간 노드의 값을
    쓰지 않음 (`signal.release`)
  - 상수배(대리 기울기 감쇠·Kir2.1·드롭아웃·Shibire)를 값을 붙잡지 않는 단일 연산으로 (`signal.scaled`, 감쇠는 노드 3개 → 1개)
  - 연결별 기울기용 스파이크를 1바이트로 저장하고 그대로 읽는 커널 (스파이크 0이면 기울기를 읽지 않음)
  - 합친 LIF 역전파가 출력을 붙잡던 것

  | GPU | 처음 (0.1.16) | 지금 |
  |---|---|---|
  | 전체 뇌 배치 8: 역전파용 메모리 | 6.88 GB | 1.82 GB |
  | 전체 뇌 배치 8: 학습 1스텝 (200스텝) | 1,376 ms | 635 ms (2.2배) |
  | 전체 뇌 배치 32: 학습 1스텝 | 메모리 부족 (75% 상한) | 1,092 ms (6 GB) |
  | 버섯체 배치 8: 순전파 (1,000스텝) | 600 ms | 132 ms (4.5배) |
  | 버섯체 배치 8: 학습 1스텝 (1,000스텝) | 1,148 ms | 300 ms (3.8배) |
  | 버섯체 배치 8: 역전파용 메모리 | - | 0.16 GB |

  이어서 스텝당 비용: 감쇠를 LIF 역전파 커널 안에서, 불응기 갱신·다음 스텝 act를 LIF 커널에서 (효과기 마스크가 없을 때),
  입력 프레임을 순전파당 한 번 연속 배열로. 예제 22개 모두 다시 끝까지 (any_graph 465 → 117초 등)
- **한 줄 모델 `fd.MushroomBody(n_in, n_classes)`, `fd.ConnectomeModel(circuit, inputs, outputs, n_in, n_classes)`**:
  커넥톰 층을 쓸 때 매번 손으로 하던 일을 대신 함 - 입력을 발화율(Hz)로 (버섯체에 사구체 반응이 오면 실제 사구체 → PN 구조),
  처음 데이터로 연결 세기 자동 보정, 출력 뉴런을 클래스로 (Homeostasis + Projection). `fit`·`predict`·`score`,
  fd.train·save·load 그대로. 세 줄로 DoOR 12가지 냄새 0.962 (손으로 배율을 맞춘 예제와 같음).
  기본값: 학습 = 입력 → 출력 흥분성 경로 위 연결 종류 (`trainable="path"`), 목표 발화율 = 출력 뉴런 수에 맞춰
  (KC 5 Hz, 뉴런이 적으면 높게, 5~30 Hz). 보정은 데이터 전체에서 고르게 뽑은 시료로 (앞에서 자르면 클래스 순 데이터의
  뒤 클래스가 빠짐)
- **Pathway 등에 넣은 커넥톰 층을 저장하면 보정 배율(gains)이 사라지던 것**: gains가 딕셔너리라 상태(state)에 안 들어가,
  불러오면 보정 전 세기로 조용히 다른 출력 → 배열로 상태에 넣음 (예전 파일도 읽힘)
- **RateEncoder: 특징 수가 적으면 (k=20 이상) 입력 정보가 전부 사라지던 것** - 모든 입력 뉴런이 같은 평균을 받아 시료마다
  최댓값 정규화 뒤 전부 max_rate (특징 8개 → 20개 뉴런 모두 100 Hz, 어떤 시료든). 뉴런마다 특징 수의 절반 이하만 고름
  (특징 40개 이상이면 예전과 같음)
- **0~1 입력 (Colab에서 실제로 겪은 것)**: `torch.rand(32, 2)`처럼 정규화한 값을 넣으면 Hz로 해석돼 100 ms에 입력 뉴런당
  스파이크가 평균 0.1번뿐인데,
  - 경고는 "연결이 약함"으로 원인을 잘못 짚었고 (입력 뉴런이 우연히 한두 번 발화했으므로)
  - **`calibrate`가 배율을 43억 배(2^32)까지 올리고도 목표 20 Hz 대신 0.6 Hz로 아무 경고 없이 끝남** (경고는 정확히 0 Hz일
    때만) - 남은 출력 10 Hz는 32시료 중 하나의 우연한 스파이크 한 개 (1000/t_ms 단위)
  → 입력 최댓값이 0~1이면 "입력은 Hz" 안내 (한 번). 입력 판정은 가장 활발한 입력 뉴런이 시행당 1번도 발화하지 않는지로
  (버섯체 PN처럼 일부러 0인 입력이 섞여도 오판 안 함). `calibrate`는 입력이 그렇게 조용하면 배율을 바꾸지 않고 오류,
  `max_gain`(기본 1000배) 상한, 목표의 ±tol에 못 닿으면 0 Hz가 아니어도 경고. 같은 코드를 `x * 100`으로 고치면 배율 81배에서
  20 Hz (32시료 중 26개 발화)
- **입력 신호**: 입력 뉴런은 두 스텝에 한 번까지만 발화 (dt 0.1 ms면 최대 5 kHz)라 큰 입력 발화율은 조용히 잘리고
  역전파는 그것을 모름 → 스텝당 확률 0.2를 넘으면 층마다 한 번 경고. `calibrate`·`reach`는 진단 중이라
  "출력이 모두 0" 경고를 내지 않음
- **Linear 계열 점검**: `tests/fuzz_layers.py` - Projection(1·2·3차원 입력, bias), Neuropil(edge·pair·free, bias,
  fan_in·counts), Homeostasis, 측억제, 활성화, Pathway 조합의 기울기를 float64 수치 미분과 대조 (CPU·GPU 각 300번,
  문제 0). 일부러 틀린 역전파를 넣으면 잡아내는 것도 확인. 학습 뒤 Neuropil = `x @ dense().T + b`, 그룹 순서,
  `edge_chunk`도 확인. 연산 퍼저에 음수 축 transpose·1차원 @·지수·숫자 ** 신호·squeeze·flatten·배열 경계 clip 추가
  (GPU clip 버그를 이것이 찾음)
- **테스트가 출력이 0인 채로 검사하던 것**: ThreeFactor의 "역전파와 같은 기울기" 비교 4개, 1차원 입력, brian 뉴런 매개변수
  학습, 그래프 compare가 발화하지 않는 회로로 돌아 0 = 0 비교였음 → 발화하는 회로로 바꾸고 발화를 확인
  (ThreeFactor 비교는 발화하는 상태에서도 통과). 의도된 경고는 테스트마다 이유와 함께 표시 → 테스트 경고 0개

## [0.1.16] - 2026-10-04

- **LIF 한 스텝을 원본 Shiu et al. 2024 Brian2 모델과 같게 (`timing="brian"`, 새 기본값)**: 원본을 직접 돌려 비교해서
  찾은 차이 - 불응기 중 도착한 시냅스 입력은 버림(예전: 쌓아 두었다 반영), 적분 → 발화 → 입력 순서, 불응기 21스텝,
  정확한 선형 적분. 같은 입력이면 스파이크 시각까지 같고, 전체 뇌 MN9 실험이 원본과 잡음 안에서 일치
  (예전 방식은 다단계에서 최대 24% 과대). `validation/shiu2024/`
  - **동작이 바뀜**: 같은 설정의 스파이킹 층 결과가 0.1.15와 다르다. 예전 방식은 `timing="legacy"` (torch판과 같음).
    0.1.15에서 저장한 층 파일은 자동으로 legacy로 읽혀 예전과 같은 결과. REPORT의 실험 ①~⑫는 예전 방식으로 한 것
- **함수·인자 점검**: 공개 함수 58개 인자에 잘못된 값(음수·0·NaN·무한대·정수 자리에 실수·문자열)을 넣어 보니 107건이
  조용히 통과 → 모두 알기 쉬운 오류로 (`flydnet/_check.py` 공통 규칙, 회귀 테스트 `tests/test_arguments.py` 199가지).
  위험했던 것:
  - 뉴런 매개변수 `params`: **이름 오타가 조용히 무시**되던 것 (`{"tmbr": 10}` → "혹시 't_mbr'?"), `t_mbr`·`tau` 0·음수
    (0으로 나누기), **문턱 `v_th` ≤ 리셋 `v_rst`** (발화 판정·기울기 부호가 뒤집힘)
  - 옵티마이저: `betas`가 [0, 1) 밖 (수렴 안 함), `eps` ≤ 0, `momentum` ≥ 1, 음수 `decay`
  - `checkpoint_every` 0·실수·문자열, 음수 발화율·학습률·잡음, 범위 밖 비율(`k_frac`, `inhibitory`, `beta`, `p`)
  - 정적 분석(받기만 하고 안 쓰는 인자): `Signal.mean(dtype=)`이 무시되던 것을 적용. 234개 함수·메서드의 문서와 실제 인자 일치 확인
  - 예제 22개 다시 끝까지 실행 (검증이 정상 사용을 막지 않음)
- **출력까지 신호가 가지 않는 경우**: 기본 세기로는 버섯체 PN→MBON에서 MBON의 96%가 0 Hz(평균 0.03 Hz),
  `fd.graphs.layered`·`erdos_renyi` 기본값은 출력이 전부 0 Hz였고, 출력이 0이면 역전파 기울기도 거의 0이라 학습이 멈춤.
  예제들은 손으로 맞춘 배율(`gains={"PN>KC": 3.0}`)로 피해 왔음
  - **`calibrate`가 출력만 목표로 주면 실패하던 것**: 목표 그룹으로 들어오는 연결만 키워서 중간 층이 꺼져 있으면 배율
    256배에도 0 Hz로 끝나고 아무 경고도 없었음 → 입력 → 목표의 흥분성 경로 위 중간 그룹(중계)을 `relay_hz`(기본 5 Hz)까지
    함께 올림. 억제를 내보내는 그룹(APL 등)은 제외. 목표에 못 닿으면 경고. `target` 기본 20 Hz, `iters` 기본 20.
    버섯체 MBON만 목표 → MBON 0.03 → 20 Hz (KC 0.13 → 4.7 Hz 자동), layered 출력 0 → 11 Hz
  - **`layer.reach(rates)`**: 그룹마다 입력에서의 최소 홉 수·발화율·활동 비율, 신호가 처음 끊기는 곳(`break_at`)과
    약한 출력(평균 1 Hz 미만)을 알려 줌
  - 경고: 처음 순전파에서 입력이 있는데 출력이 모두 0 (입력 뉴런도 발화하지 않았으면 입력 쪽 문제로 구분), 층을 만들 때
    입력에서 경로가 없는 출력 그룹·시냅스 지연보다 짧은 `t_ms`
- **포아송 난수 버그**: splitmix64는 0 → 0이라, seed 0·스텝 0·칸 0의 난수가 늘 0.0 → **seed=0이면 첫 시료의 첫 입력 뉴런이
  발화율과 상관없이 시작하자마자 발화**했음 (0.01 Hz 입력에도). 섞기 전에 상수를 더함 (표준 splitmix64). 난수열이
  바뀌어 포아송 입력 결과가 시드별로 조금 달라짐. 원본 Brian2와 다시 비교해 모든 조건이 잡음 안 (p > 0.2,
  `validation/shiu2024/`)
- **라이브러리 전체 코드 검토** (자체 엔진 → 회로 → 기능 모듈 → torch판, 약 8,000줄. 역전파 공식은 손으로 다시 유도해 대조,
  torch판은 같은 입력으로 자체 엔진과 18항목 대조 - 모두 일치). 고친 것 (회귀 테스트 `tests/test_review.py`):
  - 자동 미분: **두 손실이 중간값을 공유하면 두 번째 retrograde가 기울기를 조용히 버리던 것** → 오류 (한 번에 더하거나
    keep=True). `retrograde(retro=)` 모양이 다르면 오류 (예전: 브로드캐스트된 틀린 기울기). `quiescent()` 안에서
    역전파해도 체크포인트 구간 기울기가 0이 되지 않게. `max`의 역전파가 float64로 바뀌던 것. `clip`에 배열 경계
  - **그룹별 `t_mbr`가 `tau`(5 ms)와 같으면 출력이 0 Hz** (정확한 적분 계수가 0/0) → 안정적인 급수로. 5.001 ms면 9% 틀렸음
  - **RateEncoder에 1차원 입력(시료 하나)을 주면 모두 max_rate** (특징 1개짜리 시료 n개로 봄) → 시료 하나로.
    GlomerularEncoder·KCExpansion도 1차원·리스트·pandas를 받고 특징 수를 확인
  - **explain·gradcheck가 학습 기울기(.retro)를 바꾸던 것**: explain 뒤 rule.step()이 explain의 기울기로 가중치를 바꿈 /
    gradcheck는 쌓아 둔 기울기를 지움 → 둘 다 원래대로 둠
  - 회로: `subset`이 오타 난 그룹을 조용히 빼던 것 → 오류. 그룹 밖 뉴런이 있으면 shuffled·randomized가 실패하고
    summary에서 빠지던 것 → "?" 그룹. **저장하면 주석(meta)이 모두 문자열이 되어 `driver(gaba=True)`가 안 맞던 것** →
    숫자·참거짓 유지, driver가 값 하나·숫자도 받음. `normalized`가 입력 0인 뉴런에서 NaN
  - genetics: `activate(hz=NaN·inf)`가 자극을 조용히 없애던 것 → 오류. `screen`이 효과기를 붙일 수 없는 집단을
    오래 돈 뒤에야 알리던 것 → 시작 전에. Holm 보정이 NaN p를 1로 바꾸던 것
  - `sign_flip_p`가 1e-8보다 작은 단위의 점수를 모두 "차이 없음"(p = 1)으로 보던 것 → 단위와 무관
  - `fd.graphs.erdos_renyi`가 노드가 적으면 연결을 덜 만들 수 있던 것 → 정확히 (자기 연결을 뺀 칸에서 바로 뽑음).
    `layered([n])`·확률 범위 밖·없는 그룹 쌍 → 오류
  - `forward(batch=0)`, `calibrate(iters·step·tol)`, 들어오는 연결이 없는 그룹의 calibrate, `train(val=)` 모양(첫 에폭
    뒤에야 실패), 빈 데이터의 `extract`·`train_linear`, `Projection`이 1차원·3차원 입력(nn.Linear처럼)
  - `fd.download(path=...)`가 묶음이 여러 개면 path를 조용히 무시하고 기본 위치(C 드라이브)에 받던 것 → path/<묶음>
  - torch판 `gains`의 오타가 조용히 무시되던 것 → 오류
  - **테스트 자체의 버그**: 인자 점검 표 10줄이 테스트 줄의 호출 실수(인자 중복)로 TypeError가 나서 검사하려던 값은
    한 번도 시험되지 않았음 → 고치고, 이런 실수는 테스트 실패로 잡히게 함. 그 줄들이 실제 빈틈 8개를 찾음
- **전체 점검 3차** (정적 분석, 저장·불러오기 왕복 13설정 x CPU/GPU 4쌍, CPU·GPU 일치 14기능, 경계 입력 31가지, 회로 변환 성질):
  - **Izhikevich 기울기 폭발**: v^2 항의 기울기가 문턱 근처에서 스텝마다 1보다 커서 30 ms에 1e10, 방향도 틀림
    (gradcheck cos -0.23) → 역전파에서만 그 기울기를 0 이하로 자름 (순전파 값은 그대로). cos +0.90, 크기 비율 0.59
  - **LIF 플러그인을 여러 층이 함께 쓰면 기울기가 틀림** (최대 3%): init에서 계산한 상수를 모델 객체(self)에 둬서,
    체크포인팅이 역전파 중에 step을 다시 부를 때 다른 층의 불응기를 씀 → 실행마다 새로 만드는 `ctx.cache`에.
    사용자 정의 모델도 실행별 값은 `ctx.cache`에 (`fd.neurons` 설명)
  - **ThreeFactor·STDP로 학습하면 mosaic이 조용히 꺼지던 것**: 역행성 경로 없이 돌아서 "학습 중이 아님"으로 판단됨 →
    켜짐 (평가·`quiescent()`에서는 예전처럼 꺼짐). gradcheck의 수치 미분도 역전파 쪽과 같은 마스크로 비교
  - `t_ms`가 `dt`의 배수가 아닐 때 발화율을 실제로 돈 시간으로 환산 (배수면 예전과 같은 값).
    `count_from_ms`가 반올림으로 `t_ms`와 같아지면 오류 (0으로 나누기)
  - `fd.train`·`fd.evaluate`에 빈 데이터 → 오류 (예전: ZeroDivisionError / 조용히 0.0), `record=[]` → 빈 기록,
    `gradcheck(seeds=0)`은 계산 전에 오류
- **연산·인자 점검 2차**: 1차 표에 없던 공개 함수 49개 인자 (`argfuzz` 2차)에서 35건이 조용히 통과 → 33건 오류로
  (`compare(chance=)` 범위 밖 2건은 손실 기준값도 되므로 의도적으로 허용). 조용히 틀린 값을 내던 연산:
  - `Signal.var(ddof)`: ddof ≥ 원소 수이면 1로 나눠 틀린 값 → 오류. `clip(lo > hi)` → 오류
  - `surprise`(교차 엔트로피): **라벨 수 ≠ 시료 수여도 조용히 0.693**을 돌려주던 것, 빈 배치(NaN), 1차원 로짓 → 오류
  - `fire(slope ≤ 0)` (기울기 부호 뒤집힘), `inhibit(k=0, frac>1)`, 음수 forward `seed` → 오류
  - `Pathway`에 부를 수 없는 것을 넣으면 만들 때 바로 오류. `Activation` 종류 오류 메시지
  - 그 밖: `screen(seeds)`·`lines(min_size)`·`activate(hz)`·`tune(candidates)`·`Local(radius)`·`drifting_grating`·
    `column_map(smooth)`·`door_odors`·`biconditional_mixtures`·`evaluate(batch)` 등 검증
  - `compare`: 대조군을 만들다 난 인자 오류는 종류(ValueError 등)를 유지 (예전: 모두 RuntimeError)
- **수학 연산·자료구조 반복 검토** (`tests/fuzz_ops.py`, `tests/fuzz_sparse.py`): 연산 35가지를 무작위 모양·축·브로드캐스팅으로
  값(numpy float64 기준)과 기울기(수치 미분) 대조 - CPU 약 12,000번·GPU 약 2,600번, 희소 배선·전용 GPU 커널을 밀집 행렬과
  대조 - 약 500번. 찾아서 고친 것:
  - **합을 float64로 누적** (결과 자료형은 그대로): float32 합의 반올림 오차로 1e7 근처 값의 분산이 1.667로 나오던 것 →
    0.667 (numpy float32 자체도 1.667). 평균은 x(1/n) 대신 ÷n
  - `sigmoid`: 큰 음수에서 오버플로 경고 없이 (안정적인 공식)
  - `std`: 값이 모두 같을 때 기울기가 무한대(NaN)이던 것 → 0
  - `where`: numpy 조건 + GPU 신호에서 실패하던 것
  - 학습 배율의 지수를 ±20으로 제한 (Connectome·Neuropil): 학습률이 커도 무한대로 넘치지 않고 부호 유지
  - 회귀 테스트 `tests/test_ops_fuzz.py`
- 오류 점검 5차:
  - **같은 손실로 retrograde를 두 번 하면 오류** (전에는 두 번째가 조용히 아무것도 안 함 - 기울기가 쌓였다고 착각하기 쉬움).
    다시 보내려면 첫 번째에 `keep=True` (torch와 같은 규칙)
  - 조용히 틀리던 것을 오류로: `calibrate` 목표 음수(배율이 NaN), 음수 `gains`·`set_gain`(흥분·억제가 뒤집힘 -
    부호는 `with_sign`), 음수 `dt`, `explain(seeds=0)`(모두 NaN), 대조군 없는 `compare`, `Inhibition(frac>1)`
  - 알기 쉬운 메시지: `fd.train(batch=0, epochs<0)`, `door_task`의 정수 아닌 개수, `Inhibition(k<1)`
  - 확인만 함 (맞음): 브로드캐스팅·불리언 마스크·음수 인덱스·간격 슬라이스·중복 고급 인덱싱·같은 층 재사용의 기울기
- **입력 자료형 자유롭게**: 층(Projection, Connectome, Homeostasis, Neuropil, 인코더 등)·`fd.train`이 정수·불리언·uint8·
  float16·torch bfloat16·리스트·pandas·torch·CuPy 모두 받고 float32로 계산 (numpy·torch·CuPy float64를 넣으면 정밀도를
  위해 그대로). 전에는 bfloat16이 모든 곳에서 실패하고, 정수 입력은 float64로 승격되어 느리고 메모리를 두 배로 씀.
  torch 연결 장치도 bfloat16·float16을 받고 기울기를 입력과 같은 자료형으로 돌려줌
- **라벨**: 정수가 아닌 실수 라벨(1.7)은 오류 (전에는 조용히 1로 잘림), 문자열 라벨은 `np.unique(..., return_inverse=True)`
  안내
- 오류 점검 3차 (실패 상황·입력 형식·설정 실수):
  - 데이터를 불러올 때 파일 크기 확인 (받다가 끊겼거나 다른 버전이면 알기 어려운 읽기 오류 대신 다시 받기 안내)
  - 다운로드 중 네트워크 오류에 안내 (전에는 URLError 그대로), 실패한 임시 파일 정리
  - 같은 그룹을 입력·출력에 두 번 넣으면 오류 (전에는 뉴런이 중복되어 자극 하나가 무시되고 출력도 중복)
  - `fd.train`이 파이썬 리스트·pandas 입력을 받음 (`evaluate`·`train_linear`와 같게)
  - `door_task`: 측정된 사구체가 적어 냄새가 없을 때 원인 안내, `samples` 확인
  - 확인만 함 (문제 없음): with 블록·explain·calibrate가 도중에 실패해도 층 상태가 원래대로
- 기능 점검 2차 (대조군 성질: shuffled 연결 수 유지·중복 없음, randomized 종류별 연결 수, shuffled_weights, local 칸 유지,
  subset·normalized·with_sign·저장 왕복, 인코더·KC 확장 희소성 5%, 도파민 리드아웃, Neuropil 4방식과 edge 부호 유지,
  시각계 도구, 명령줄, 예전 예제 13개 끝까지 실행): `controls.Custom(..., question=)` (결과표의 '묻는 것' 설명),
  이름 안내 `num_parameters` → `.n_synapses()` (괄호 빠졌던 것)
- 오류 점검 (정적 분석, CPU·GPU 17개 기능 일치, torch·CuPy 없는 환경, 새 예제 11개 끝까지 실행, CPU 전용 전체 테스트):
  - 역전파 핵심 반복문이 연산이 돌려준 기울기 개수를 확인 (전에는 개수가 다르면 그 부모의 기울기가 조용히 사라짐)
  - `gradcheck`: 연결을 바꿔도 출력이 변하지 않으면 (출력 뉴런이 발화하지 않음) "cos 0" 대신 그렇다고 알려 줌
  - `doctor`: `FLYDNET_DEVICE=cpu`로 일부러 CPU를 고르면 "GPU 계산이 안 됨"을 문제로 세지 않음
- **맞는 이름을 알려 주는 오류**: torch·numpy에서 쓰던 이름을 쓰면 flydnet 이름을 안내 (`x.grad` → `.retro`,
  `loss.backward()` → `.retrograde()`, `syn.w` → 이 신호 자체가 값 `.numpy()`, `model.parameters()` → `.synapses()`,
  `rule.zero_grad()` → `.clear()`, `fd.Linear` → `fd.Projection`, `fd.no_grad` → `fd.quiescent` 등), 오타는 비슷한 이름 제안.
  Signal·Tissue·가소성 규칙·Circuit·`fd.` 최상위
- **Signal에 numpy·torch에서 기대하는 것들**: `min`, `var`, `std`, `sqrt`, `square`, `softmax`, `squeeze`, `astype`, `copy`
  (모두 역전파 됨, 수치 미분 확인), `argmax`/`argmin` (numpy), `tolist`, `any`, `all`, `size`, `float()`·`int()`·`bool()`
  (값 하나일 때, 여러 개면 알기 쉬운 오류 - `if loss > 0:`이 깨지던 것), `abs()`, `np.mean(x)`
- 저장·불러오기·데이터 폴더에서 `~`(홈 폴더)를 풀고, 저장할 때 없는 폴더는 만듦 (전에는 내부 임시 파일 이름이 보이는 오류)
- **짧은 이름** (긴 이름도 그대로): `fd.Connectome`, `fd.Adaptive`, `fd.Glomeruli`, `fd.Inhibition`, `fd.MBON`, `fd.tune`,
  `fd.flywire()`, `fd.brain()`, `fd.worm()`, 인자 `damp`·`ckpt`
- **가중치 자동 보정 `Connectome.calibrate(x, {"KC": 5, ...})`**: 그룹마다 평균 발화율이 목표가 되도록 들어오는 연결
  종류의 배율을 반복 조정 (흥분·억제 비율 유지, 저장됨). 손으로 맞추던 gains를 대신함
- **정확도**: 실제 냄새 12개 과제에서 MBON(48개) 대신 KC(2,597개)에서 읽고 학습률 3e-3 → 0.57 → 0.962 ± 0.004 (seed 4).
  `fd.train` 기본 학습률 1e-2 → 3e-3. `examples/quickstart.py`가 이 설정
- **편의**: `fd.train` (학습 루프 한 줄: 배치·섞기·코사인 학습률·clip·평가, seed를 받는 층에 자동으로),
  `fd.evaluate`, `fd.door_task` (DoOR 실제 냄새 구분 과제), `fd.Homeostasis` (시료마다 평균 0·표준편차 1, 발화율을
  리드아웃에 넣기 전에), `Pathway(x, seed=)`가 seed를 받는 자식에게 전달, `ConnectomeLayer`에 시료 하나 `(n_in,)` 입력.
  예제 `examples/quickstart.py`
- **조용히 틀리던 것을 오류로**: 음수 발화율 입력(스파이크가 안 생겨 출력 0), 실수 seed, 없는 주석 값으로 만든 빈 그룹
  (`from_flywire`, 비슷한 값 제안), seed 1개짜리 `compare`, 음수 학습률. 입력·출력 그룹이 겹치면 경고.
  알기 쉬운 메시지: 잘못된 share·input_mode·v_init, t_ms ≤ 0, 없는 그룹 이름, record 범위 밖, Projection 크기,
  train_linear 특징·라벨 개수
- **스파이킹 역전파 기울기 수정**: 연결 종류별 유한 차분과 비교해 보니 0.1.15까지의 대리 기울기는 긴 시뮬레이션에서
  방향이 무작위이거나 반대였고 크기가 수억 배로 부풀었음 (초파리 버섯체 1,000스텝 cos -0.75, 16억 배 - 되먹임 회로를
  돌며 곱해짐).
  - `ConnectomeLayer(surrogate_damp=...)` 대리 기울기 감쇠 (값은 그대로, 역전파만). 기본 `"auto"` = min(1, (11/홉 수)^1.5),
    홉 수 = 스텝 / 시냅스 지연 스텝: 초파리 버섯체·전체 뇌에서 가장 잘 맞은 값(홉 11 → 1, 25 → 0.3, 55 → 0.1)을 맞추는
    경험 규칙 (버섯체 1,000스텝 cos 0.91, 전체 뇌 200스텝 0.985). `timing="legacy"`와 0.1.15 파일은 1
  - `truncate=` 구간 절단 역전파 (시냅스 지연보다 짧으면 오류), `noise=` 막전위 잡음 (대리 기울기가 잡음 있는 뉴런의
    발화 확률 기울기에 가까워짐, 버섯체 0.91 → 0.98)
  - `fd.gradcheck` (방향·크기·**기준 신뢰도**, `seeds=`로 평균 출력의 기울기), `fd.tune_surrogate` (기준이 불안정하면
    고르지 않음). `layer.damp_value()`. `fd.ThreeFactor`도 같은 감쇠
  - `explain`: 1차 예측과 실제 손상의 방향이 다른 유형을 표시 (되먹임 억제 뉴런 APL 등)
  - 역전파 그래프가 상수 입력의 배열을 붙잡아 두지 않음 (메모리)
  - `validation/gradients/`. 예쁜꼬마선충은 깨끗한 기준으로도 cos 약 0.5 (대리 기울기가 부분적으로만 맞는 회로)
  - **동작이 바뀜**: 스파이킹 층의 기울기 크기·방향이 0.1.15와 다름 (값은 같음). 예전 기울기는 `surrogate_damp=1`
- **플러그인: 사용자 정의 뉴런 모델 `fd.neurons`**: `NeuronModel`(state, init, step)을 정의하면 `ConnectomeLayer(neuron=...)`
  에서 시뮬레이션·자동 미분 역전파·체크포인팅·genetics·explain·mosaic·torch 연결 장치·저장(`@register`)이 동작.
  내장 `LIF`(내장 고속 LIF와 스파이크 동일, 기울기 4e-5 이내), `Izhikevich`. 예제 `examples/custom_neuron.py` (AdEx)
- **관찰자 규격과 `fd.STDP`**: `begin(info)` / `step(s, spikes, **extra)`로 사용자 정의 학습 규칙. 쌍 기반 STDP
  (곱셈형, 데일의 법칙 유지), 사용자 정의 뉴런에서도 동작
- **torch 연결 장치 `fd.torch.bridge`**: 자체 엔진 구조물을 torch 모델 안에서 `nn.Module`로. 계산은 자체 엔진,
  메모리 공유(GPU DLPack·CPU numpy, 복사 없음), torch backward → 자체 엔진 retrograde, 학습 값은 같은 메모리의 torch
  Parameter (torch 옵티마이저가 갱신). no_grad 지원, 자체 엔진이 배열을 바꿔 끼우면 다시 묶음. 출력·기울기가 자체
  엔진과 같음 (테스트). 전체 뇌에서 0.1 torch판 복사본보다 약 4배 빠름. 예제 `examples/torch_bridge.py`
- **어떤 그래프든 회로로**: `Circuit.from_edges` (번호·이름, 그룹 dict·노드별 이름, 남는 노드는 rest),
  `from_scipy`, `from_networkx`, `to_scipy`, `to_networkx`, `regroup`, `check` (번호 범위·그룹 겹침·NaN 검증).
  예쁜꼬마선충 커넥톰 `Circuit.celegans()` (Cook et al. 2019, 고정 커밋·SHA-256, GABA 뉴런 26개는 억제,
  `python -m flydnet download worm`). 합성 그래프 `fd.graphs` (Erdos-Renyi, Watts-Strogatz, Barabasi-Albert,
  블록 구조, 앞먹임 층; 억제 비율·데일의 법칙). 주석이 없는 그래프에서 `explain`·`mosaic`·`lines`는 그룹 단위
  (`by=None` 자동). 예제 `examples/any_graph.py`
- `fd.compare` 해석: 대조군이 실제 배선보다 유의하게 좋으면 그렇다고 말함 (전에는 "차이를 확인하지 못함"으로 잘못 나옴)
- **역전파 없는 학습 `fd.ThreeFactor`** (e-prop 방식 3요소 규칙): 시냅스마다 적격 흔적(리셋·불응기 반영) x 시냅스 후
  대리 기울기 x 학습 신호. 피드백 `"random"` / `"connectome"`(실제 연결을 따라 출력에서 퍼지는 오차) / `"none"`.
  출력으로 들어오는 연결은 시간 역전파와 같은 기울기 (테스트). 메모리가 시뮬레이션 길이와 무관 (800 ms에서 41 MB 대
  역전파 4.1 GB). 실제 냄새 과제에서 리드아웃만보다 +33%p, 역전파보다 -8%p. `examples/threefactor_odor.py`
  - `ConnectomeLayer`에 스텝마다 상태를 받는 관찰자 연결점 (`timing="brian"`)
- **세포 유형 드롭아웃 `fd.genetics.mosaic`**: 학습 중에만 시료마다 세포 유형을 통째로 끔 (`by="neuron"`이면 보통
  드롭아웃). seed로 정해짐, 체크포인팅·역전파 확인, `explain`할 때는 자동으로 꺼짐. 실제 냄새 과제에서 보통 드롭아웃과
  깨끗한 정확도는 같고 사구체 결손에 더 강함 (+1.7%p, p = 0.031, 최악 +5.6%p) - `examples/mosaic_odor.py`
- **회로 기여도 `fd.explain`**: 학습된 모델이 어떤 세포 유형·경로(pre > post)에 기대는지. 뉴런·연결 배율 탐침의
  기울기 = 가상 손상의 1차 예측, `verify=k`로 실제로 꺼서(silence) 비교하고 순위 상관을 보고. 스파이킹·연속값,
  체크포인팅, CPU=GPU 확인. 예제 `examples/explain_odor.py` (DoOR 실제 냄새, 기대는 사구체 = 반응하는 사구체)
- **가상 유전학 `fd.genetics`**: 초파리 실험실의 방법 그대로 뉴런 집단을 고르고(`driver`, split-GAL4처럼 `&`)
  끄고(`silence` = Kir2.1)·막고(`block` = Shibire-ts)·켜고(`activate` = CsChrimson, Shiu et al.과 같은 포아송 자극)·
  없앤다(`ablate`). `lines`(세포 유형별 드라이버 모음), `screen`(유전자 스크린: 효과·짝지은 p값 표).
  효과기는 학습 중에도 쓸 수 있음 (역전파·체크포인팅·CPU=GPU 확인). `active`/`clear`, 층 출력에 켜진 효과기 표시,
  `screen`은 seed가 모자라면 경고하고 여러 집단 보정 p(`p_holm`)도 냄. Shiu 모델의 "silence"는 `block`에 해당
- `Circuit.whole_brain()`: 전체 뇌 138,639개 뉴런 (Shiu et al. 2024 모델과 같은 뉴런·순서, 주석 없는 뉴런 포함)
- `ConnectomeLayer(inputs=None)`: 입력 그룹 없이 (`layer(None, batch=시행 수)`), 효과기로만 자극
- 회로 주석에 `side` 추가
- 예제 `examples/genetics_sugar.py`: Shiu et al. 2024 재현 (당 GRN → MN9)
- `python -m flydnet doctor`: 설치 진단 (드라이버 CUDA, 설치된 CuPy와 충돌, torch, flydnet이 쓸 장치, 권장 설치 옵션)
- README GPU 설치 안내 정정: CuPy가 이미 있으면(Colab 등) 옵션 없이 설치. GPU 옵션은 CuPy가 없을 때만, 드라이버 CUDA에 맞춰.
  (Colab의 CUDA 13 환경에 `[gpu-cuda12]`를 붙이면 CuPy가 두 개가 되고 CUDA 라이브러리가 12로 내려가 torch까지 영향)
- README의 "Alpha (0.2)" 표기를 0.1로
- CUDA·드라이버 업데이트 대비:
  - 전용 GPU 커널이 컴파일되지 않으면 (새 GPU를 모르는 NVRTC, CuPy 변경 등) 경고 후 CuPy 기본 연산으로 계속 (결과 같음, 더 느림)
  - `doctor`가 전용 커널을 실제로 컴파일·실행해 결과 확인
  - `doctor` 버전 판단 정정: 드라이버는 하위 호환이라 새 드라이버 + 예전 CuPy(예: CUDA 13 + cupy-cuda12x)는 문제 아님.
    CuPy가 드라이버보다 새것일 때만 문제. 미래 드라이버(CUDA 14 등)에는 없는 옵션 대신 지원하는 최신 옵션을 권함
- 저장 파일 호환:
  - 모든 저장 파일에 형식 버전·종류·저장한 flydnet 버전을 기록. 더 새 형식이면 "flydnet을 업데이트" 오류,
    다른 종류의 파일이나 손상·잘린 파일이면 무엇이 문제인지 알려 주는 오류
  - 저장은 임시 파일에 쓴 뒤 바꿔치기 → 저장 도중 멈춰도 예전 파일이 깨지지 않음
  - 0.1.15에서 저장한 파일을 고정해 두고 계속 같은 결과로 읽히는지 테스트 (`tests/data/`)
- 학습 안정성:
  - 가소성 규칙에 `clip=` (기울기 전체 크기 제한, clip_grad_norm_), `rule.last_norm`
  - 기울기에 NaN·무한대가 있으면 시냅스를 바꾸기 전에 멈추고 어느 시냅스인지 알려 줌 (`guard=True` 기본,
    `named_synapses()`를 주면 이름으로). 손실이 NaN·무한대면 역전파 전에 멈춤. 비용은 스텝당 약 1 ms
  - GPU 메모리 부족이면 사용량과 해결 방법을 붙인 `GPUMemoryError`

## [0.1.15] - 2026-10-03

**기준 엔진이 torch에서 자체 엔진 `flydnet.ganglion`(NumPy = CPU, CuPy = GPU)으로 바뀜. torch는 선택.**

- **자체 엔진**: 자동 미분(`Signal`, `retrograde`), 조직(`Tissue`, `Pathway`, `Projection`, `Neuropil`, …),
  가소성 규칙(`Plasticity`, `AdaptivePlasticity`), 체크포인팅. 이름은 생물 구조에서 땀
- **모든 기능이 torch 없이**: `ConnectomeLayer`(스파이킹 LIF·연속값, 시간 역전파)와 0.1의 나머지 기능 전부를 같은 이름으로.
  torch판과 출력·기울기가 같음. torch판은 `fd.torch.*` (0.1 코드는 `fd.` → `fd.torch.`)
- **대조 실험 `fd.compare`**: 실제 배선 대 겹겹이 놓인 대조군(`randomized` ⊂ `shuffled` ⊂ `Local`, `shuffled_weights`),
  짝지은 seed, 부호 뒤집기 순열 검정, 함정 경고, 어떤 구조가 중요한지 해석
- **성능**: LIF 한 스텝을 하나의 연산으로 합치고 CUDA 커널 직접 작성. 전체 뇌 학습 1스텝 배치 8에 0.58초 (torch판 2.2초)
- **데이터**: 원본 저장소의 커밋으로 주소 고정 + SHA-256 확인, `python -m flydnet download / verify`
- **새 기능**: `KCExpansion`, `Circuit.randomized()`, `Circuit.shuffled_weights()`, `limit_gpu_memory`, `FLYDNET_DEVICE`
- **설치**: 기본은 numpy·scipy·pandas·pyarrow. `[torch]`, `[gpu-cuda12]`, `[gpu-cuda13]`
- **버그 수정**: 음수 라벨로 손실이 조용히 틀림, NaN 입력이면 출력이 조용히 0, 정수 신호 × 실수가 0으로 잘림,
  연결 종류 순서가 torch판과 달라 학습된 배율이 엉뚱한 종류에 붙음, 한국어 윈도우에서 GPU 커널 컴파일·출력 실패,
  예제 `--help`가 죽음, `extract`가 층 안의 오류를 삼킴, 장치를 옮기면 옵티마이저 오류

## [0.1.0] - 2026-10-03

- 첫 배포: Circuit (FlyWire 회로 선택, 무작위·국소 무작위 대조군), ConnectomeLayer (스파이킹 LIF·연속값 뉴런,
  역전파 학습, 체크포인팅), 인코더, 도파민 연합 학습 리드아웃, DoOR 냄새 데이터, 시각계 도구, 데이터 다운로드
