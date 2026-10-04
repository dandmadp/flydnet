# 변경 기록

## [0.1.16] - 2026-10-04

- **LIF 한 스텝을 원본 Shiu et al. 2024 Brian2 모델과 같게 (`timing="brian"`, 새 기본값)**: 원본을 직접 돌려 비교해서
  찾은 차이 - 불응기 중 도착한 시냅스 입력은 버림(예전: 쌓아 두었다 반영), 적분 → 발화 → 입력 순서, 불응기 21스텝,
  정확한 선형 적분. 같은 입력이면 스파이크 시각까지 같고, 전체 뇌 MN9 실험이 원본과 잡음 안에서 일치
  (예전 방식은 다단계에서 최대 24% 과대). `validation/shiu2024/`
  - **동작이 바뀜**: 같은 설정의 스파이킹 층 결과가 0.1.15와 다르다. 예전 방식은 `timing="legacy"` (torch판과 같음).
    0.1.15에서 저장한 층 파일은 자동으로 legacy로 읽혀 예전과 같은 결과. REPORT의 실험 ①~⑫는 예전 방식으로 한 것
- **스파이킹 역전파 기울기 수정**: 연결 종류별 유한 차분과 비교해 보니 0.1.15까지의 대리 기울기는 방향이 무작위이거나
  반대였고(버섯체 100스텝 cos 0.01, 1,000스텝 -0.38) 긴 시뮬레이션에서 크기가 수억 배로 부풀었음 (되먹임 회로를 돌며
  곱해짐). `ConnectomeLayer(surrogate_damp=...)` 대리 기울기 감쇠 (기본 0.1, `timing="legacy"`와 0.1.15 파일은 1),
  `truncate=` 구간 절단 역전파 (시냅스 지연보다 짧으면 오류). `fd.gradcheck` (방향·크기 확인), `fd.tune_surrogate`
  (감쇠 고르기). `fd.ThreeFactor`도 같은 감쇠. 재실험: 3요소 규칙 비교 결론 같음 (무작위 피드백 0.637 → 0.661),
  AdEx 기울기 확인 -0.42 → +0.98, explain 예제는 KC 크기가 맞게 됐고 APL은 여전히 틀림. `validation/gradients/`
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
