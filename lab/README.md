# lab - 연구 예제

두 종류를 `examples/`와 분리해 둔다.

- flydnet의 [실험적] 기능(`flydnet.lab`: 추가 연결 `Growth`, 대조 학습 등)을 시험한 예제. 효과가 조건부
- 결론이 안정된 연구 재현 예제. 다른 예제가 이미 실행하는 코드만 쓰고 버그를 잡은 기록이 없어, 배포 검증에서는 뺌
  (0.1.18, `examples.json`의 호출 기록으로 판단). REPORT의 해당 숫자는 이 예제들로 잰 값

- `scripts/verify.py`의 기본 전체 검증에는 들어가지 않음 (오래 걸리고, 정식 기능의 회귀 검사가 아니므로).
  growth·보정(calibrate)·학습 규칙·연합 학습 리드아웃 코드를 고쳤을 때는 `python scripts/verify.py --only examples --lab`로 함께 돌릴 것
- README의 "실험적 기능" 결과는 0.1.18에서 이 예제들로 잰 값

| 예제 | 내용 | 시간 (GPU) |
|---|---|---|
| `growth_odor.py` | 멀쩡한 버섯체에 추가 연결 - 무작위·헤브·기울기·대조 학습 | 약 5분 |
| `growth_lesion.py` | PN→KC 80% 손상 뒤 회복 - 정답 없는 대조 학습, 순환 확인(SCARF), 진단 조건 | 약 20분 |
| `growth_lesion_mnist.py` | 같은 손상 회복을 MNIST로 재현 (대조 학습 이득 없음) | 약 30분 |
| `door_assoc.py` | REPORT ⑦: 조건부 구별(냄새 섞음) - 로지스틱·MLP·연합 학습, 사구체·KC 실제·KC 무작위 | 약 39분 |
| `odor_ablation.py` | REPORT ③-b: 버섯체 부분 연결을 무작위로 바꿨을 때의 손실 (PN→KC·KC→KC·APL) | 약 28분 |
