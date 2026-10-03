# 변경 기록

## [0.2.0] - 2026-10-03

**기준 엔진이 바뀜: torch 대신 flydnet 자체 엔진 `flydnet.ganglion` (NumPy = CPU, CuPy = GPU).**
기본 설치에 torch가 들어가지 않음. torch 기능은 `pip install flydnet[torch]`로 그대로 쓸 수 있음.

- `flydnet.ganglion`: 자체 자동 미분과 신경망 구성 요소 (이름은 생물 구조에서 땀)
  - `Signal` (≈ Tensor), `plastic` (≈ requires_grad), `retrograde()`/`.retro` (≈ backward/grad), `quiescent()` (≈ no_grad)
  - `Synapse`, `Tissue`, `Pathway`, `Projection` (≈ Parameter, Module, Sequential, Linear)
  - `Neuropil` (실제 커넥톰 배선 희소 층), `LateralInhibition`, `AxonHillock`, `Activation`, `MushroomBodyOutput`
  - `Plasticity`, `AdaptivePlasticity` (≈ SGD, Adam), `surprise` (≈ cross_entropy), `transmit`, `fire`, `inhibit`
  - 수치 미분·torch·CPU↔GPU 비교 테스트 36개. MNIST (Neuropil) 96.2%, torch 없이 GPU 2에폭 약 6초
- `import flydnet`이 torch 없이 됨. torch가 필요한 0.1.0 기능 (`ConnectomeLayer`, `AssocReadout`, `KCExpansion`,
  `RateEncoder`, 시각계 도구 등)은 처음 쓸 때 불러오고, torch가 없으면 설치 안내
- 선택 설치: `flydnet[torch]`, `flydnet[gpu-cuda12]`, `flydnet[gpu-cuda13]`
- `fd.Neuropil` 등 최상위 이름은 자체 엔진 것. torch판은 `flydnet.torch.anatomy`, `flydnet.torch.physiology`
- **`fd.ConnectomeLayer`도 자체 엔진** (`flydnet.ganglion.ConnectomeLayer`): 스파이킹 LIF·연속값, 시간 역전파,
  체크포인팅(`fd.checkpoint`), 세포 유형 매개변수, 연결 종류 공유, 시간 변화 입력, 기록, 저장(np.savez).
  torch판과 출력이 같고(버섯체 스파이크 100% 일치) 기울기도 같음. 포아송 난수는 상태 없는 해시 난수 (CPU·GPU 같음).
  **0.1의 torch판은 `fd.torch.ConnectomeLayer`** (이전 코드는 이 이름으로 바꾸면 됨)
- `fd.ganglion.limit_gpu_memory(fraction)`: GPU 메모리 상한, `Circuit.to_arrays()`/`from_arrays()`: torch 없는 저장
- **0.1의 나머지 기능도 전부 자체 엔진판** (같은 이름, 결과는 numpy): `RateEncoder`, `GlomerularEncoder`, `extract`,
  `train_linear`, `DopamineReadout`, `AssocReadout`, `KCExpansion`, `synthetic_odors`, `door_odors`,
  `biconditional_mixtures`, `visual_circuit`, `column_map`, `drifting_grating`, `direction_offsets`.
  난수 없는 계산은 torch판과 값이 같음. torch판은 모두 `fd.torch.*` (0.1 코드는 `fd.` → `fd.torch.`)
- **성능**: LIF·연속값 한 스텝을 하나의 연산으로 합치고(역전파 직접 유도), 행 우선 SpMM·연결별 내적 CUDA 커널.
  전체 뇌 학습 1스텝 배치 8: 0.58초 (torch판 2.2초), 배치 32: 1.3초 (torch판 2.6초)
- 버그 수정: 정수 신호 × 실수가 0으로 잘림 / GPU에서 `MushroomBodyOutput.predict`가 cupy 배열이라 numpy 라벨과 비교 불가 /
  가소성 규칙을 만든 뒤 모델을 다른 장치로 옮기면 오류 / CUDA 소스의 한국어 주석 때문에 한국어 윈도우에서 컴파일 실패 /
  예제 4개의 `--help`가 도움말의 `%` 때문에 죽음 / 연결 종류 순서가 torch판과 달라 학습된 배율이 엉뚱한 종류에 붙음

아래는 배포하지 않은 0.1.1의 변경 (모두 0.2.0에 포함):

- 데이터 주소를 원본 저장소의 특정 커밋으로 고정하고, 받은 파일을 크기 + SHA-256으로 확인.
  원본이 나중에 바뀌거나 옮겨져도 항상 같은 데이터. 예전 버전 파일이 있으면 다시 받음 (내용은 0.1.0과 같음)
- `python -m flydnet verify`: 받은 데이터가 기대한 버전인지 확인
- 다운로드 진행률을 한 줄로 표시 (터미널·Colab)
- README 역전파 예제에 정규화 층 추가 (없으면 손실이 커짐)
- `KCExpansion`: 실제 PN→KC 배선을 스파이크 시뮬레이션 없이 한 번에 계산하는 확장 층
  (고정 투영 `sparse`/`gaussian` → PN 평균 빼기 → 시냅스 수 → 상위 `k_frac`만 남김). 1만 샘플 0.14초
- README에 연속 학습 사용법: 사전학습 특징 + `AssocReadout` (CIFAR-100 10과제 57.7%, 재생 버퍼 51.6%)
- `flydnet.anatomy` (`torch.nn`에 해당, 0.2.0에서 `flydnet.torch.anatomy`로 옮김): `Neuropil` (커넥톰 배선 희소 층, 학습 방식 edge/pair/free, 배선 지문으로
  다른 회로의 state_dict 거부), `LateralInhibition`, `AxonHillock`, `MushroomBodyOutput`
- `flydnet.physiology` (`torch.nn.functional`에 해당, 0.2.0에서 `flydnet.torch.physiology`로 옮김): `transmit`, `wiring`, `fire`, `inhibit`, `transduce`,
  `kenyon_code`, `recall`, `reinforce_`. `AssocReadout`과 `KCExpansion`도 이 함수들을 씀 (결과 동일)
- 희소 전파(`SparsePropagate`)가 직사각 행렬도 지원. 스파이크 출력이 입력과 같은 자료형

## [0.1.0] - 2026-10-03

- 첫 배포: Circuit (FlyWire 회로 선택, 무작위·국소 무작위 대조군), ConnectomeLayer (스파이킹 LIF·연속값 뉴런,
  역전파 학습, 체크포인팅), 인코더, 도파민 연합 학습 리드아웃, DoOR 냄새 데이터, 시각계 도구, 데이터 다운로드
