# 변경 기록

## [0.1.16] - 2026-10-04

- `python -m flydnet doctor`: 설치 진단 (드라이버 CUDA, 설치된 CuPy와 충돌, torch, flydnet이 쓸 장치, 권장 설치 옵션)
- README GPU 설치 안내 정정: CuPy가 이미 있으면(Colab 등) 옵션 없이 설치. GPU 옵션은 CuPy가 없을 때만, 드라이버 CUDA에 맞춰.
  (Colab의 CUDA 13 환경에 `[gpu-cuda12]`를 붙이면 CuPy가 두 개가 되고 CUDA 라이브러리가 12로 내려가 torch까지 영향)
- README의 "Alpha (0.2)" 표기를 0.1로

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
