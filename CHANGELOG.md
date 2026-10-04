# 변경 기록

## [0.1.17] - 미배포

- **연산 버그** (회귀 테스트 `tests/test_017.py`):
  - **`transpose`에 음수 축을 주면 역전파가 틀린 모양의 기울기를 쌓던 것** (`(2, 3)` 신호에 `(3, 2)` 기울기) - 역순열을
    음수 축 그대로 계산했음. `transpose((1, 0))`처럼 튜플 하나도 받음, 같은 축 두 번은 오류
  - **`s == 0`·`s != 0`이 원소별 비교가 아니라 객체 비교라 늘 `False` 하나**이던 것 → numpy처럼 원소별
    (`(spk == 1).sum()` 같은 코드가 조용히 틀렸음). 딕셔너리·집합 키로는 예전처럼 신호 객체 그대로
  - `x ** 0`의 기울기가 x = 0에서 NaN → 0. `숫자 ** 신호` 지원 (예전: TypeError)
  - `@`가 1차원도 numpy처럼: `(n,) @ (n, k)`, `(m, n) @ (n,)`, `(n,) @ (n,)` (예전: 오류)
  - **GPU 신호의 `clip`에 numpy 배열 경계를 주면 TypeError** (CPU에서는 됨) → 장치로 옮김. Signal 경계도 받음
  - `concat`에 numpy·cupy 배열을 섞으면 실패하던 것, 장치가 다르면 알기 쉬운 오류
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
