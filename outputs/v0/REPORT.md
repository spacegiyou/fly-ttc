# Fly-TTC v0 Report

이 점수는 초파리 LPLC2/Giant Fiber에서 영감을 받은 고정 팽창 검출기다. 전체 초파리 뇌를 시뮬레이션한 것이 아니다.

## Setup

모델: looming proxy inspired by LPLC2/GF, v0 Farneback expansion. 출력은 arbitrary unit이다.
분석 fps=15, short side=256px, EMA=0.3.
평가 성공 클립의 실제 입력 fps 범위=23.600–30.600. 시간축은 실제 프레임 인덱스/fps에 config의 offset만 적용한다.
theta=6.790890, validation negative FPR=0.1000, target FPR=0.1000.
표준화와 임계값은 음성 validation에서만 보정한다. 양성 통계 또는 evaluation 통계로 보정하지 않는다.

## Data used (ids, seed, n)

Source: nexar-ai/nexar_collision_prediction
Source revision: aa97deda5a59f00bb7187739053b7c72e14374df
seed=0, 요청 n=100, 평가 성공 n=100, 제외 n=0.
Status counts: {"ok": 100}
IDs: 00019, 00077, 00078, 00081, 00093, 00108, 00143, 00235, 00254, 00300, 00418, 00447, 00463, 00469, 00501, 00510, 00528, 00533, 00544, 00588, 00617, 00623, 00629, 00639, 00675, 00705, 00742, 00750, 00751, 00769, 00773, 00776, 00823, 00856, 00877, 00887, 00900, 00905, 00911, 00927, 00929, 00935, 00937, 00957, 00971, 00979, 01000, 01024, 01031, 01033, 01062, 01075, 01088, 01116, 01121, 01177, 01209, 01210, 01213, 01217, 01229, 01240, 01242, 01250, 01260, 01268, 01318, 01365, 01366, 01399, 01408, 01429, 01448, 01459, 01503, 01510, 01530, 01538, 01593, 01606, 01618, 01632, 01640, 01663, 01705, 01711, 01754, 01812, 01878, 01900, 01912, 01926, 01968, 01988, 02019, 02024, 02049, 02085, 02102, 02129
이 소량 서브셋은 scene/light 다양성을 고려한 편의 표본이며 전체 Nexar train의 무작위 대표 표본 또는 배포 환경의 사고 발생률 추정치가 아니다.
양성 trace 12개, 음성 trace 12개를 같은 y축 범위로 그렸다. 최대 요청은 각 12개다.

## Metrics

| Split | n | pos | neg | AUROC | AUPRC (AP) | FPR | Median lead→alert (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Evaluation (headline) | 80 | 40 | 40 | 0.5463 | 0.5235 | 0.2000 | 0.3440 |
| Validation (calibration) | 20 | 10 | 10 | 0.6000 | 0.6318 | 0.1000 | -0.4830 |
| All (descriptive) | 100 | 50 | 50 | 0.5592 | 0.5317 | 0.1800 | -0.0016 |

Evaluation AUROC=0.5463로 0.55 미만이다. 이 서브셋에서 유용한 양성/음성 분리는 확인되지 않았다. 파이프라인 실행 성공은 검출 성능 성공을 뜻하지 않는다. Flow 부호·시간축·사건 이후 제외를 별도로 검증해야 한다.

핵심 성능은 Evaluation 행이다. Validation은 보정에 사용했고 All에는 그 클립이 포함되어 독립 성능 추정치가 아니다. N/A는 평가 가능한 표본 또는 양쪽 클래스가 부족함을 뜻하며 0으로 대체하지 않았다.
양성 점수 최대값과 모든 사건 메트릭은 t < time_of_event만 사용한다. 그래프의 회색 사건 이후 구간은 집계에서 제외한다. 음성은 분석 구간 전체를 사용한다.
임계값 판정은 S > theta (엄격한 초과), FPR 단위는 프레임이 아닌 클립이다. s_at_alert는 alert 이하 마지막 샘플, s_at_event는 event 미만 마지막 샘플이다.
lead는 첫 발화가 아니라 사건 이전 최대 점수 시각 기준이다. Evaluation median lead→event=1.1997s; lead→alert IQR=[-0.8751, 2.4286]s, n=40.
이 peak lead 통계에는 임계값을 넘지 못한 양성도 포함된다. 따라서 median lead→alert=0.3440s를 실제 조기 경보의 중앙값으로 해석하면 안 된다.
hit_alert=0.1500, n=40. 창은 [alert−1.5s, event)다.
Early hit은 event−window 이하의 관측 프레임에서 한 번이라도 초과했는지다. 관측 구간이 cutoff 이후 시작하면 N/A다.

| Early window (ms) | Evaluation hit rate | Observable positives |
| --- | --- | --- |
| 500 | 0.1250 | 40 |
| 1000 | 0.1000 | 40 |
| 1500 | 0.1000 | 40 |

scene/light별 수치와 분모는 metrics_by_scene.csv, metrics_by_light.csv에 저장했다.

## What worked

Evaluation에서 양성 40개와 음성 40개가 집계되었다. 양성 pre-event threshold hit=7/40, rate=0.1750.
진단 패널: 성공 클립 00300, 전체 프레임 블러 적용.
합성 불변식 테스트 통과 여부는 실행 시 pytest 결과로 확인한다. 이 리포트 자체가 합성 테스트 통과를 추정하지 않는다.

기록된 검증 결과:

| 검증 항목 | 결과 |
| --- | --- |
| pytest | 통과 51개 / 실패 0개 |
| 영상 SHA256 검증 | 100/100개 일치 |
| 양성 사건 이전 peak·hit·시간축 | 50개 점검 |
| 검증 오류 | 0개 |
| Validation 임계값 재계산 | theta=6.790890, FPR=0.1000 |
| 표준화 입력 확인 | 10 validation negatives only |

[전체 검증 기록](validation_checks.json)을 함께 보존했다.

합성 자극 결과 (단위 통계로 계산한 테스트 점수이며 실제 영상 점수·임계값과 직접 비교하지 않는다):

| Stimulus | Frames | Late mean S | Peak S |
| --- | --- | --- | --- |
| loom | 60 | 0.324407 | 0.376493 |
| recede | 60 | 0.001433 | 0.011029 |
| translate | 60 | 0.170572 | 0.176521 |
| flicker | 60 | 0.000000 | 0.000000 |
| edge | 60 | 0.021517 | 0.074588 |

추가 감사에서 실제 영상 4개의 흐름 방향·입력 스케일·PTS를 대조했다. 양성 50개의 집계에서 사건 이후 756프레임이 제외됨을 확인했다. 이 검사 범위에서는 구체적인 구현 오류를 발견하지 못했으며 평가 결과를 보고 설정을 바꾸지 않았다. [흐름·시간축 감사 기록](scorer_clock_audit.json).

## What failed (and why that is expected)

파리가 측면 충돌을 못 잡는 것은 버그가 아니라 회로의 범위.
실패 태그는 영상 사고 유형의 정답이 아닌 휴리스틱이다. 태그는 중복 가능하고, 비율 분모는 해당 split의 평가 성공 클립 전체다. 필요한 flow 진단 또는 validation 기준이 없으면 해당 태그를 추측하지 않는다.
규칙은 src/fly_ttc/eval/failure_tags.py와 config.yaml의 failure_tags에 공개되어 있다. good_loom은 마지막 1초 radial 평균이 음성 validation radial 95퍼센타일을 초과함을 뜻한다.
Blob의 최초 출현은 이전 물체 지름이 없으므로 증가 점수를 0으로 둔다. 이는 출현 자체를 팽창으로 세지 않는 구현 선택이다. 단순 전역 밝기 깜빡임 합성 검사는 전체 조명 변화 불변성을 보장하지 않는다. 실제 텍스처, 노출 변화, 와이퍼는 여전히 흐름과 blob 항에 영향을 줄 수 있다.
AUROC가 낮으면 flow 부호, 시간축, 사건 이후 제외를 먼저 점검해야 하며, 이 보고서는 낮은 값을 숨기지 않는다.

- 초파리 looming 회로는 **앞으로 커지는 물체**에 선택적이다. 측면 스침, 후방 추돌, 이미 지나간 사고, 타 차량끼리의 원거리 사고는 설계상 못 잡는다.
- Nexar 양성은 collision과 near-miss를 구분하지 않는다.
- 광학 흐름은 야간·비·와이퍼·큰 카메라 흔들림에 깨진다.
- 점수는 TTC의 물리적 초가 아니다. 상대적 looming proxy다.
- 임계값은 이 서브셋 음성 FPR에 맞춰져 있어 다른 도메인에 그대로 못 옮긴다.
- 이 파이프라인은 자율주행 스택을 대체하지 않는다. 희귀 이벤트 마이닝/라벨 제안용이다.

## How to reproduce

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m fly_ttc.cli run --model v0 --config "[local checkout]/outputs/v0/config.yaml"
python -m fly_ttc.cli report --run "[local checkout]/outputs/v0"
```

clip_scores.csv, threshold.yaml, config.yaml, metrics.json 및 매니페스트를 함께 보존한다. 모든 양성과 위양성 음성의 parquet를 보존하며, 정상 음성 trace는 보고서 생성 후 그림으로만 보존한다. 재생성 시 기존 정상 음성 그림을 유지한다.

## Next (v1)

v1은 구현하지 않았다. v0 리포트 검토 이후 논문 STAR Methods에 근거한 4방향 모델을 검토한다. 스케치 대체 구현이라면 proxy_not_zhao_star로 명시해야 한다. BADAS-Open: optional, skipped.

![positive_traces](figures/positive_traces.png)

![negative_traces](figures/negative_traces.png)

![lead_to_alert](figures/lead_to_alert.png)

![roc_pr](figures/roc_pr.png)

![loom_diagnostic](figures/loom_diagnostic.png)

![failure_modes](figures/failure_modes.png)
