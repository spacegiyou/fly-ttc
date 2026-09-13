# Local visual reflex: model sources and deviations

검토일: 2026-09-13. 구현 이름은 **`proxy_not_zhao_star`**다. 공개 저자 코드의 EMD 계산과 STAR Methods의 국소 방향 대립을 참고한 작은 영상 특징 모듈이다. 논문의 모든 결과를 재현한 모델, 실제 커넥톰, GF 뉴런, 사고확률 모델로 부르지 않는다.

## 확인한 1차 자료

- [Klapoetke et al., Nature 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC7457385/), DOI [10.1038/nature24626](https://doi.org/10.1038/nature24626): 국소 수용장 중심의 바깥 방향 흥분과 안쪽 방향 억제에 관한 생물학적 근거. 본 구현은 그 실험 자료나 개별 시냅스 가중치를 가져오지 않는다.
- [Zhao et al., iScience 2023, STAR Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10073940/), DOI [10.1016/j.isci.2023.106337](https://doi.org/10.1016/j.isci.2023.106337): EMD 절, Eq. 1–5를 확인했다. Eq. 6–10의 GF 모델은 구현하지 않았다. 브라우저 도구에서는 CAPTCHA가 표시되었지만 같은 공식 PMC URL을 `curl`로 가져온 전체 HTML에서 Methods와 수식을 직접 읽었다.
- [논문에서 연결한 저자 MATLAB 저장소](https://github.com/zhaojunyu01/EMD-LPLC2-GF-model)는 실제로 접근 가능했다. 읽기 전용으로 복제하여 아래 commit과 파일을 확인했다. MATLAB을 설치하거나 실행하지 않았다. 저자 파일은 이 저장소나 공유 ZIP에 복사하지 않았다.

고정한 upstream commit: `50c7c8f3f74052b2562ccba967c9f1032f37405b`.

파일 비교의 기준 디렉터리는 `code/2_Response-Characteristics-LPLC2-Arrays-And-GF-Unit-To-Looming/`이다. 저자 저장소에는 실험별로 서로 다른 파일 사본이 있으므로 임의 사본을 섞어 하나의 정확한 원본이라고 취급하지 않는다.

| 이 저장소의 구현 | 직접 확인한 자료 | 일치하는 부분 / 의도적 차이 |
|---|---|---|
| `TimestampEMD.update` in `src/fly_ttc/models/local_reflex_proxy_not_zhao_star.py` | [emd.m](https://github.com/zhaojunyu01/EMD-LPLC2-GF-model/blob/50c7c8f3f74052b2562ccba967c9f1032f37405b/code/2_Response-Characteristics-LPLC2-Arrays-And-GF-Unit-To-Looming/emd.m), [loom.m](https://github.com/zhaojunyu01/EMD-LPLC2-GF-model/blob/50c7c8f3f74052b2562ccba967c9f1032f37405b/code/2_Response-Characteristics-LPLC2-Arrays-And-GF-Unit-To-Looming/loom.m) | 고역통과 → ON/OFF → 저역통과 → 이웃 곱. 고정 10 ms recurrence를 실제 `dt`로 확장. |
| ON/OFF rectification | [OnRect.m](https://github.com/zhaojunyu01/EMD-LPLC2-GF-model/blob/50c7c8f3f74052b2562ccba967c9f1032f37405b/code/2_Response-Characteristics-LPLC2-Arrays-And-GF-Unit-To-Looming/OnRect.m), [OffRect.m](https://github.com/zhaojunyu01/EMD-LPLC2-GF-model/blob/50c7c8f3f74052b2562ccba967c9f1032f37405b/code/2_Response-Characteristics-LPLC2-Arrays-And-GF-Unit-To-Looming/OffRect.m) | `ON=max(hp,0)`, `OFF=max(0.05−hp,0)`를 따름. OFF는 음수 변화에 0.05 deadband를 적용하는 식이 아니다. |
| `pool_directional_motion` | STAR Eq. 1–4, [lplc2conv2.m](https://github.com/zhaojunyu01/EMD-LPLC2-GF-model/blob/50c7c8f3f74052b2562ccba967c9f1032f37405b/code/2_Response-Characteristics-LPLC2-Arrays-And-GF-Unit-To-Looming/lplc2conv2.m) | 바깥−안쪽 방향 대립. 이 구현은 명시적 사각 영역 **평균**, 저자는 convolution **합**. |
| 동일 함수의 네 팔 결합 | STAR Eq. 5 | `max(arm−threshold,0)`의 곱. 수치 단위를 안정적으로 표시하기 위해 곱의 네제곱근을 출력. |
| `_rf_layout` | STAR의 cross-shaped RF 설명 | 32 px RF, 5×5 완전 지지 RF 위치는 새 공학 설계. 방향 좌표는 화면의 오른쪽/왼쪽/아래/위. |
| `LocalReflexScorer.update` output EMA | 본 프로젝트의 명시적 설계 | 초 단위 exponential smoothing. 논문의 GF 또는 spike dynamics를 대체한다고 주장하지 않음. |
| `tests/test_local_reflex_proxy_not_zhao_star.py` | 위 저자 EMD 함수와 독립적 계산/합성 자극 | scalar recurrence 비교, 4방향 실제 이동 부호, 불규칙 시간 간격, 경계 연결 금지, 국소 수용장, 수축/밝기 변화 검사. |

## 실제 계산과 단위

입력은 원본 분석 좌표계의 `gray∈[0,1]`, 실제 시각 `t_s`다. 모듈은 이미지 크기를 바꾸지 않는다. 입력 크기나 시간 순서가 바뀌면 오류를 내며 새 클립에는 `reset()`이 필요하다. 첫 프레임은 기준 상태를 초기화하고 0을 반환한다.

저자 코드와 동일한 backward-Euler 시간 필터를 사용한다. 여기서 `dt=t_s−previous_t_s`, `τH=0.250 s`, `τL=0.050 s`다.

```text
βH = τH / (τH + dt)
hp_t = βH × (gray_t − gray_previous + hp_previous)
ON_t = max(hp_t, 0)
OFF_t = max(0.05 − hp_t, 0)
αL = dt / (τL + dt)
delayed_t = αL × channel_t + (1−αL) × delayed_previous
R = delayed(x,y) × channel(x+s,y)
L = channel(x,y) × delayed(x+s,y)
D = delayed(x,y) × channel(x,y+s)
U = channel(x,y) × delayed(x,y+s)
```

ON과 OFF의 대응 곱을 더한다. 이는 **EMD 임의 단위**이며 px/s나 TTC 초가 아니다. `dt`를 전달한다고 큰 시간 간격에서 사라진 공간·시간 정보를 복구하거나 샘플링률 불변성을 보장하지 않는다. 해당 한계는 합성 샘플링 조건별로 측정해야 한다.

샘플 간격 `s=sample_spacing_px`는 이웃 픽셀의 거리이며 다운샘플링 비율이 아니다. 공통 EMD 셀 격자는 `(H−s,W−s)`이고 모든 방향 채널을 `(x+s/2,y+s/2)` 셀 중심에 표시한다. 수평·수직 이웃 쌍 자체의 중심은 반 셀만큼 다르므로 이 공통 표시 좌표는 집계 규약이다. 경계를 넘는 이웃 연결과 wrap-around를 만들지 않는다.

각 팔에서 지정된 바깥 방향 채널에서 반대 채널을 빼고 평균한다. 기본 값은 동일한 0의 팔 임계값, 네 팔 곱의 네제곱근이다. 임계값 0은 생물학적으로 추정한 값이 아니며 STAR의 양수 `L0`와 다르다. 하나라도 양수가 아니면 그 RF의 곱은 0이다. `S=max(activation_map)`이며 맵 좌표는 입력 픽셀 단위다. 위치는 실제 물체 경계, 검출 박스, 추적 ID가 아니다.

출력 평활화는 `αout=1−exp(−dt/0.050)`로 한다. 순수 pooling 함수는 평활화하지 않은 맵을 반환하므로 같은 EMD 배열과 동일한 평활화로 여러 대조군을 비교할 수 있다.

## 논문과 저자 코드의 차이도 보존해 기록

STAR Eq. 5는 각 팔에서 임계값을 **빼고** 양수 부분을 곱한다. 확인한 MATLAB 파일은 `L×1[L>2]`를 곱하며 임계값을 빼지 않는다. STAR는 cross 폭을 RF 한 변의 1/3로 설명하지만 해당 코드의 폭은 40/100이다. 이 구현은 임계값 빼기와 1/3 폭을 채택한다. 숫자 결과가 동일한 완전한 Python 포트라고 부를 수 없는 이유다.

저자 코드는 왼쪽/오른쪽 채널 순서와 `conv2` 커널 반전을 함께 사용한다. 여기서는 직접 오른쪽 팔에서 오른쪽−왼쪽을 평균한다. 주어진 실행 지시서의 짧은 EMD 식에 채널 이름만 붙이면 방향 부호가 뒤집힐 수 있으므로, 실제 오른쪽/왼쪽/아래/위 이동 영상으로 채널 부호를 검증한다.

## 공정한 비교 범위

`direction_permutation`은 같은 EMD 필터 상태, 같은 RF 위치·크기, 같은 네 채널과 같은 네 팔에서 방향 할당만 바꾼다. 기본 순서는 `(right,left,down,up)`이다. 이는 **공학적 방향 매핑 대조군**이며 생물학적 커넥톰을 섞은 실험이 아니다. 실제 신경 연결표도, 학습할 시냅스 가중치도 없다.

`simple_energy_map`은 같은 네 팔의 지지 영역에서 `|R−L|+|D−U|`를 평균한다. 팔마다 평균한 뒤 네 팔을 평균하므로 팔 교차 영역은 각각의 팔에서 중복 집계된다. 위치 지원과 EMD 입력은 같지만 계산 연산 수가 완전히 같다는 주장은 하지 않는다. `aggregation='mean'`도 선택할 수 있으나 네 팔 동시 활성 요구를 없애는 별도 구조다.

각 모델은 동일한 합성 calibration 자극, 동일한 임계값 선택 규칙과 예산을 받아야 한다. 교란 모델에 원래 모델의 임계값을 강제로 적용하면 불공정하다. 학습 가중치가 없어 재학습이라는 이름으로 할 작업은 없으며, 출력 임계값 보정은 각 모델에 별도로 허용한다. 제한된 합성 결과만으로 생물 회로의 장점이나 실도로 효과를 주장하지 않는다.

## 확인한 upstream SHA256

| 파일 | SHA256 |
|---|---|
| emd.m | `7d9844a578c2c79bd0fc96b85301486d77f4a4013e3da2cb027faf32ceefdc0f` |
| loom.m | `96acab266a3306ecbda79019b4e034b88b827da536a05efeaeca1e96ae8ac7e8` |
| lplc2conv2.m | `26d878284f7c03b6035ddadbd8e1cc9cb393908588a11ff2c94349e172dba386` |
| OnRect.m | `7b505b01503124db139a40ca5951cf8297ba496a58c1f05ad70cf65201ad6f2f` |
| OffRect.m | `956348ec5f5eea6dcd5de3fb77f4ba335bc662bc485e813597c573f38ae8854c` |

고정 10 ms에서 독립 scalar Python reference와 vectorized EMD가 일치하는 것을 테스트한다. 이것은 MATLAB 실행과의 교차 검증이나 논문 그림의 수치 재현과 구분한다. upstream 저장소의 HEAD가 나중에 바뀌더라도 위 commit 경로와 해시가 본 검토의 기준이다.
