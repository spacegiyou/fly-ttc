# v0 고정 결과 재감사 및 하위집단 탐색

원본 `clip_scores.csv`와 `threshold.yaml`만 읽었다. 원래 분할과 theta=6.790889832241를 고정했고 재보정·재실행·평가셋 튜닝을 하지 않았다. 원본 파일 SHA256은 `analysis.json`에 기록했다.

## 독립 평가 분할 재집계

아래 대괄호는 95% 신뢰구간이다. AUROC는 라벨별 클립 복원추출 bootstrap, TPR/FPR은 Wilson 구간이다. TP/FN은 사건 이전 클립 최대값이 기존 임계값을 넘었는지로 판정했다.

| 하위집단 | 양성/음성 | TP/FP/TN/FN | AUROC [95% CI] | AP | TPR [95% CI] | FPR [95% CI] | alert hit |
|---|---:|---:|---|---:|---|---|---:|
| All clips | 40/40 | 7/8/32/33 | 0.546 [0.426, 0.673] | 0.524 | 17.5% [8.7, 31.9] | 20.0% [10.5, 34.8] | 6/40 |
| scene=Highway | 11/12 | 4/1/11/7 | 0.629 [0.379, 0.856] | 0.668 | 36.4% [15.2, 64.6] | 8.3% [1.5, 35.4] | 4/11 |
| scene=Industrial | 2/5 | 0/0/5/2 | 1.000 [1.000, 1.000] | 1.000 | 0.0% [0.0, 65.8] | 0.0% [0.0, 43.4] | 0/2 |
| scene=Nature | 1/0 | 0/0/0/1 | NA | NA | 0.0% [0.0, 79.3] | NA | 0/1 |
| scene=Other | 3/3 | 0/0/3/3 | 0.111 [0.000, 0.444] | 0.411 | 0.0% [0.0, 56.1] | 0.0% [0.0, 56.1] | 0/3 |
| scene=Rural | 3/4 | 0/1/3/3 | 0.750 [0.250, 1.000] | 0.639 | 0.0% [0.0, 56.1] | 25.0% [4.6, 69.9] | 0/3 |
| scene=Sub-urban | 11/7 | 1/3/4/10 | 0.403 [0.104, 0.714] | 0.561 | 9.1% [1.6, 37.7] | 42.9% [15.8, 75.0] | 0/11 |
| scene=Urban | 9/9 | 2/3/6/7 | 0.444 [0.173, 0.729] | 0.518 | 22.2% [6.3, 54.7] | 33.3% [12.1, 64.6] | 2/9 |
| scene=Highway + light=Normal | 3/4 | 1/1/3/2 | 0.583 [0.000, 1.000] | 0.698 | 33.3% [6.1, 79.2] | 25.0% [4.6, 69.9] | 1/3 |

요청된 Highway + Normal light는 7개(양성 3, 음성 4)뿐이다. AUROC 0.583 [0.000, 1.000], FPR 25.0% [4.6, 69.9]이므로 이 조건을 운용 대상으로 확정할 근거가 부족하다.

![고정 임계값 하위집단 구간](subgroup_intervals.png)

## 실패 태그 해석 교정

`low_visibility` 34개는 양성 미검출 17개와 올바르게 거부한 음성 17개를 포함한다. 이 태그는 `(Dark 또는 Twilight 또는 Rain) AND 임계값 미통과` 규칙이며 양성과 음성 모두에 적용된다. 따라서 전체 태그 비율을 실제 시야 불량의 비율이나 미검출 원인 비율로 해석하면 안 된다.
점수와 독립적인 Dark/Twilight/Rain 메타데이터 조건은 39개(양성 20, 음성 19)이며 TP/FP/TN/FN=3/2/17/17이다. 이것도 실제 영상 가시성의 수동 검증 결과는 아니다.
원본 low_visibility 규칙 불일치 ID: 없음. `ego_shake`도 원시 흐름 요약값 기반 휴리스틱이므로 카메라 흔들림의 수동 확정 라벨은 아니다.

## 인용 사례 재확인

| video_id | label | split | scene | light | s_peak | predicted_positive | hit_alert |
|---|---:|---|---|---|---:|---|---|
| 01538 | 0 | evaluation | Sub-urban | Normal | 15.674886 | True | None |
| 01510 | 0 | evaluation | Rural | Normal | 10.852738 | True | None |
| 01250 | 0 | evaluation | Urban | Normal | 9.986769 | True | None |
| 00300 | 1 | evaluation | Highway | Normal | 14.240634 | True | True |
| 00510 | 1 | evaluation | Urban | Normal | 9.983522 | True | True |
| 00773 | 1 | evaluation | Highway | Twilight | 8.541714 | True | True |
| 00877 | 1 | evaluation | Highway | Twilight | 7.964100 | True | True |
| 01024 | 1 | evaluation | Highway | Bright | 6.809974 | True | True |

## 해석 범위와 다음 실험

- 모든 하위집단은 v0 결과를 본 뒤 선택한 사후 탐색이다. 다중비교 보정이나 독립 확인을 하지 않았으므로 가장 높은 행을 골라 성능 개선으로 주장할 수 없다. 점수 순위가 완전히 분리된 작은 표본에서는 bootstrap AUROC 구간이 [1, 1]이 될 수 있지만, 이는 모집단에서 완벽함을 뜻하지 않는다. 고정 임계값의 TP/FP도 함께 봐야 한다.
- `subgroups.csv`에는 evaluation, validation, all을 분리하고 scene, light, 관측된 모든 scene×light 조합을 기록했다. validation과 all은 보정 표본을 포함하는 참고 집계다. 한 라벨만 있는 행의 AUROC/AP는 NA다.
- Bootstrap은 seed=0, 반복 2000회, 라벨별 원래 표본 수를 유지했다. 고정 검출기 및 관측 클립 집합의 불확실성만 다루며 임계값 보정, 같은 출처 영상의 상관, 사후 선택에 따른 불확실성을 포함하지 않는다. 작은 하위집단의 bootstrap 구간은 불안정하거나 퇴화할 수 있다.
- AP의 무작위 참고선은 각 하위집단의 양성 비율이다. 서로 다른 양성 비율을 가진 행의 AP를 바로 비교하면 안 된다. `positive_fraction`에 분모를 기록했다.
- v1 확대 전에 자기운동 보정의 제한된 가설을 합성 자극으로 검증하는 순서가 합리적이다. 합성 평행 이동 억제와 국소 팽창 보존을 먼저 잠그고 기존 음성 validation만으로 새 점수의 통계와 임계값을 보정한다. 이미 확인한 80개 결과를 기준으로 설정을 반복 선택하면 안 된다.
- 같은 80개를 새 모델로 다시 평가하면 비교 실험은 가능하지만, 이번 결과를 보고 가설을 선택했으므로 새로운 독립 최종 검증은 아니다. 다음 확증에는 별도로 고정한 미사용 데이터가 필요하다.

## 재현

```bash
python scripts/analyze_subgroups.py --run outputs/v0 --out outputs/v0_review --seed 0 --bootstrap-iterations 2000
```
