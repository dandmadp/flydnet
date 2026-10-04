# 변경 기록

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
