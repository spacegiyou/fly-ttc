# Fly-TTC v0

초파리 LPLC2/Giant Fiber에서 영감을 받은 **고정 팽창 검출기**를 Nexar train 영상에서 평가한다. `looming proxy inspired by LPLC2/GF`이며, 전체 초파리 뇌나 생물 뉴런을 재현하지 않는다. 기본 경로는 CPU OpenCV Farneback이며, 모델 가중치를 학습하지 않는다.

## Scope

v0의 divergence, radial outflow, blob expansion, 음성 validation 표준화, 임계값 선택, 평가, Markdown/HTML 리포트를 구현한다. v1은 후속 단계이며 현재 CLI에서 명시적으로 거부한다. RAFT와 BADAS-Open 비교는 **optional, skipped**이다. Torch·MATLAB·전체 커넥톰이 필요하지 않다.

## Current research: local visual reflex

2026-09-13 추가 검토 이후 범용 사고예측 튜닝을 멈추고, 별도 브랜치 `research/local-visual-reflex`에서 작은 국소 팽창 모듈을 구현·평가했다. 기존 v0/affine 코드·설정·결과는 보존했다. 원본 검토 문서 두 사본·독립 감사 JSON·pytest 로그를 모두 확인했고 [검토 대응과 판단](docs/REVIEW_RESPONSE_2026-09-13.md)에 반영 내역을 기록했다.

Zhao 저자 공개 EMD와 STAR 방향 대립 수식을 확인했지만 RF·정규화·출력 등이 달라 모델 이름은 **`proxy_not_zhao_star`**다. `update(gray, t_s)`가 국소 활성 맵과 입력 픽셀 좌표를 반환한다. 실제 연결지도·GF·논문 그림의 재현은 아니다. [출처와 구현 대응](docs/LOCAL_REFLEX_MODEL_SOURCES.md)에 일치점과 변경점을 명시했다.

고정한 **354개 합성 조건 × 9모델**을 실행했다. 18개 calibration 중 같은 음성 12개에서 모델별 임계값을 정하고, 나머지 336개는 팽창 100개/비팽창 236개다. 7개 EMD readout은 같은 상태와 RF를 공유하며, 2개 흐름 모델은 계산량을 맞추지 않은 별도 대조다. 결정적이며 상관된 합성 진단 결과이므로 아래 숫자는 Nexar 성능이 아니다.

| 합성 challenge | AUROC | 팽창 검출 | 비팽창 오반응 |
|---|---:|---:|---:|
| 국소 방향 대립 proxy | 0.86568 | 84/100 | 61/236 (25.85%) |
| 같은 RF의 단순 EMD 움직임 에너지 | 0.64877 | 41/100 | 59/236 (25.00%) |
| 시간 정규화 전역 흐름 | 0.95237 | 100/100 | 36/236 (15.25%) |

중심 밖 어두운 원판은 16/16 검출했지만 밝은 원판은 0/16이었다. 60Hz 대비 활성 조건의 최고 응답 상대차 중앙값은 29.85%였다. **선택성·위치·샘플링 안정성 기준을 통과하지 못해 정답 ROI와 자동 탐지 단계로 확대하지 않았다.** 방향 순열 5종까지 포함한 전체 결과와 실패 조건은 [합성 실험 보고서](outputs/synthetic/local_reflex/REPORT.md)에 있다. 실제 생물 회로의 구조적 장점은 검증하지 않았다.

전체 테스트 **179개 통과**. 현재 환경의 정식 pyarrow로 기존 534개 시계열 파일/71,353행, 사건 전 peak 474개와 v0 점수 59개를 다시 확인했다. 저장 raw trace 118개 중 44개는 기록된 관측 간격이 일정하지 않았다. 기존 프레임 시각은 frame_index/reported fps이며 VFR PTS 검증은 아니다. 기존 동결 보고서의 실제 파리 능력에 대한 과도한 표현은 [재감사 보고서](outputs/review_response/AUDIT.md)와 현재 문서에서 정정한다.

```bash
python scripts/run_local_reflex.py --phase prepare
python scripts/run_local_reflex.py --phase run
python scripts/run_local_reflex.py --phase report
python scripts/export_review.py --output exports/fly_ttc_local_reflex_review.zip
```

이미 시작한 행렬의 `run`은 덮어쓰지 않는다. 보존된 결과는 `--phase report`로 보고서만 다시 생성한다. 새 연구에는 별도 출력 경로·프로토콜을 사용한다. 실제 신경 연결표 및 공정한 재학습을 포함한 구조 비교, 사람 정답 ROI 검증, 최종 배포 평가를 완료한 것으로 해석하지 않는다.

## First v0 result (2026-09-13)

seed 0에서 공식 Nexar train **100/100개**를 평가했고, 100개 파일의 SHA256이 원본과 일치했다. 소스 revision은 `aa97deda5a59f00bb7187739053b7c72e14374df`, 다운로드 크기는 1,424,019,152 bytes다. 테스트는 **51개 통과**했다.

| Split | n (pos/neg) | AUROC | AUPRC | FPR | 사건 전 양성 hit |
|---|---:|---:|---:|---:|---:|
| Evaluation | 80 (40/40) | 0.54625 | 0.52355 | 0.20 (8/40) | 0.175 (7/40) |
| Validation | 20 (10/10) | 0.60000 | 0.63178 | 0.10 (1/10) | 0.200 (2/10) |
| 전체, 보정 표본 포함 | 100 (50/50) | 0.55920 | 0.53168 | 0.18 (9/50) | 0.180 (9/50) |

theta는 **6.7908898322**다. Evaluation의 최대점 기준 `lead_to_alert` 중앙값은 **+0.34396초**, IQR 구간은 **[-0.87508, +2.42861]초**다. 이 통계에는 미검출 양성도 포함되므로 조기 경고 성공률로 해석하면 안 된다. **독립 평가의 판별 성능은 낮으며 v0 성공 신호 기준 AUROC 0.60을 넘지 못했다.** 평가 표본을 보고 임계값이나 하이퍼파라미터를 바꾸지 않았다.

AUROC 0.55 미만 점검으로 50개 양성의 저장 시계열에서 사건 이전 최대점·히트·실제 FPS 시간축을 재계산했고 일치했다. 음성 validation ID와 임계값도 재검증했다. 실제 FPS 범위는 23.6–30.6이며 30 fps로 고정 계산하지 않았다. 상세 결과와 그림은 [v0 보고서](outputs/v0/REPORT.md), 감사 기록은 `outputs/v0/validation_checks.json`에 있다. 합성 마지막 15프레임 평균 raw 점수는 loom 0.3244, recede 0.0014, translate 0.1706, 단색 flicker 0.0000이다.

## Follow-up: global motion ablation

다음 단계로 v1을 추가하기 전에 **전역 affine 흐름 보정의 효과를 분리하는 대조 실험**을 구현·실행했다. 기존 100개는 보존하고, 추가 Nexar train 100개(양성 50/음성 50, seed 1)를 먼저 고정했다. 기존 validation 음성 10개에서만 모델별 통계와 임계값을 정한 뒤 추가 표본에 적용했다. 코드·설정·ID·라벨·시각·원본 SHA256을 동결하고 **총 200개 × 4모델 = 800행**을 평가했다.

| 추가 100개 확인 평가 | AUROC | AP | FPR | 사건 전 양성 hit |
|---|---:|---:|---:|---:|
| 원본 v0 | 0.4904 | 0.5380 | 20% (10/50) | 18% (9/50) |
| Raw flow, blob 제외 | 0.4872 | 0.5299 | 20% (10/50) | 18% (9/50) |
| Affine residual, blob 제외 | 0.4876 | 0.5282 | 20% (10/50) | 16% (8/50) |
| 프레임 차분 | 0.5252 | 0.5479 | 20% (10/50) | 18% (9/50) |

사전 지정한 주 비교는 가운데 두 모델이다. **ΔAUROC = +0.0004, paired bootstrap 95% 구간 [-0.0136, +0.0160]**로, 이번 보정의 개선 근거를 얻지 못했다. 보정 후 양성 1개를 더 놓쳤으며, 기존 위양성 01538·01510·01250도 유지됐다. 이 affine 근사는 기본 검출기에 채택하지 않는다. 모든 물리적 자기운동 보정이 불가능하다는 결론은 아니다.

추가 평가의 위양성 10개 중 9개는 보정 모델의 최고점에서 affine 추정이 실패해 원래 흐름을 사용했고, 6개는 최고점까지 최근 최대 5개 표본 모두 같은 실패 처리였다. 기존 음성 01538·01510·01250도 최고점과 직전 표본에서 보정이 적용되지 않았다(01250은 최고점까지 2개 표본만 존재). 따라서 합성 자극에서 작동한 보정이 실제 문제 장면에서 충분히 적용되지 않았다. 실패 처리로 raw 흐름이 같아도 모델별 음성 표준화 통계가 달라 최종 점수는 다를 수 있다.

기존 80개에서 프레임 차분 AUROC 0.6338이 보였지만 새 100개에서는 0.5252였다. 기존 평가 결과를 보고 선택한 방법을 같은 표본에서만 평가하면 낙관적인 결론을 낼 수 있음을 보여 준다. 추가 100개도 동일 주행/장소 비중복은 검증하지 못했고, 조건 다양성 편의 표본이며 calibration 불확실성을 신뢰구간에 포함하지 않았다.

`Highway + Normal` 원본 재집계는 7개(양성 3/음성 4)에 불과했고 AUROC 0.5833, FPR 25%였다. `low_visibility` 34개는 양성 미검출 17개와 올바르게 거부된 음성 17개여서 모두 가시성 때문에 놓친 사고라고 해석할 수 없다. BADAS-Open은 공식 카드상 Nexar train 1,500개로 학습했으므로 이 train 표본에서 독립 성능 상한으로 비교하지 않았다.

[조건별 감사](outputs/v0_review/REPORT.md), [새 100개 대조 실험 보고서](outputs/ego_motion/REPORT.md), [사전 고정 프로토콜](docs/FROZEN_EGO_MOTION_PROTOCOL.md), [검토 근거·공식 출처](docs/next_experiment_review.md)에 숫자와 판단 범위를 남겼다. 합성 검증에서는 전역 움직임 억제와 국소 looming 보존 외에 **화면 전체의 진짜 looming도 보정이 지우는 반례**를 확인했다. 전체 테스트는 **95개 통과**했다.

```bash
python scripts/analyze_subgroups.py --run outputs/v0 --out outputs/v0_review
python scripts/run_ego_ablation.py --phase prepare
python scripts/run_ego_ablation.py --phase download
python scripts/run_ego_ablation.py --phase run
python scripts/run_ego_ablation.py --phase report
```

동일 기능은 `python -m fly_ttc.cli experiment --phase ...`로 실행할 수 있다. 완료된 실험의 `run` 재실행은 거부하고 `report` 재생성을 지원한다. 점수 코드·설정 변경 후 기존 confirmation으로 다시 튜닝하는 것을 막기 위해 해시가 다르면 실패한다. 새 연구 실험은 별도 출력 경로·프로토콜로 만들고 이미 확인한 표본을 새 미관측 평가라고 부르지 않아야 한다.

이후의 연구 순서는 위 Current research 절과 [검토 대응 문서](docs/REVIEW_RESPONSE_2026-09-13.md)를 따른다. 기존 200개는 이미 알려진 개발 자료이며, 새 미관측 평가로 재사용하지 않는다.

## Share without environment or videos

```bash
python scripts/export_review.py
```

`exports/fly_ttc_review.zip`에 코드와 허용한 보고서·점수·그림, 파일별 SHA256 매니페스트를 넣는다. `.venv`, `.git`, 캐시, 원본 데이터·영상·NPZ 진단 캐시는 제외한다. **`.gitignore`는 Google Drive 동기화 제외 설정이 아니다.** 이 ZIP은 원격 Drive의 기존 `.venv`를 삭제하거나 실제 동기화 설정을 변경하지 않는다. 설치 환경은 lock 파일로 재생성하고, 공유에는 이 ZIP을 사용한다.

## Install

Python 3.10 이상이 필요하다. 이번 환경은 Python 3.12로 구성했다. 다음 명령은 이 저장소 루트에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

실행 환경의 정확한 의존성 버전은 `requirements-lock.txt`에 기록했다. 같은 버전을 원하면 `python -m pip install -r requirements-lock.txt` 후 `python -m pip install -e .`를 실행한다. 다른 Python/OS에서는 호환되는 wheel이 필요할 수 있다.

## Data and license

1차 데이터는 공식 [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)의 **train**뿐이다. 공식 설명상 train은 1,500개이며 양성 750개, 음성 750개다. 양성은 collision과 near-miss를 함께 포함한다. test에는 사건 전 잘린 영상이 포함되므로 리드 타임 평가에 사용하지 않는다.

HF 원본의 `train/positive/metadata.csv`, `train/negative/metadata.csv`에는 `file_name`이 있으며, 라벨은 해당 공식 폴더에서 가져온다. 원본의 필드를 임의로 추가하지 않는다. 생성 매니페스트의 `video_id`, `video_path`, 무결성·분할 정보는 파이프라인 관리 필드다. 실제 파일 목록에서 확인한 경로만 다운로드하며, revision과 SHA256을 기록한다.

데이터 라이선스는 **nexar-open-data-license**다. 받은 `data/raw/nexar/LICENSE`와 [공식 라이선스](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)를 확인해야 한다. Nexar 영상은 이 저장소에 포함하거나 재배포하지 않는다. 데이터 라이선스를 코드 라이선스로 바꾸지 않는다. `data/`, `outputs/`는 Git에서 제외되며 수동 배치 설명과 `.gitkeep`만 추적한다.

데이터 저작권: © 2025 Nexar Inc. 출처 표기: Moura, Daniel C., and Zvitia, Orly. “Nexar Collison Dataset.” Hugging Face, 2025. 위 공식 데이터 링크를 함께 인용한다.

```bash
# 영상 없이 메타데이터만 다운로드
python scripts/download_nexar.py --metadata-only

# 기본 실험: 양성 50 + 음성 50, seed 0
python scripts/download_nexar.py --subset 100 --seed 0

# 전체 train은 명시적으로 요청할 때만
python scripts/download_nexar.py --full
```

최초 다운로드에서 `data/manifests/subset_100_seed0.csv`에 ID와 소스 revision을 고정한다. 재실행은 고정된 목록을 사용한다. 인증이 필요하면 `HF_TOKEN` 환경변수로만 전달하며 토큰은 로그에 출력하지 않는다. 다운로드 실패 시 정확한 명령과 원인을 출력하고 중단하며 다른 데이터셋으로 전환하지 않는다. 일부 영상이 없으면 해당 행은 `missing_video`로 남긴다.

scene/light를 순환해 희귀 조건도 섞는 균형 서브셋이므로 실제 주행 환경의 조건 빈도나 사고 발생률을 대표하지 않는다.

네트워크 없이 보유한 Nexar/Kaggle train CSV와 MP4를 쓰는 방법은 [수동 배치 안내](docs/DATA_DOWNLOAD.md)를 따른다. CSV에 매칭되는 파일명이 불명확하면 실패한다. 가짜 영상을 Nexar로 채우지 않는다.

## Run and reproduce

```bash
pytest -q
python scripts/run_v0.py --config configs/default.yaml
python scripts/make_report.py --run outputs/v0

# 같은 동작의 CLI
python -m fly_ttc.cli download --subset 100 --seed 0
python -m fly_ttc.cli run --model v0 --config configs/default.yaml
python -m fly_ttc.cli report --run outputs/v0

# 매니페스트에 실제 존재하는 ID로 디버그
python -m fly_ttc.cli debug --video-id 00012 --model v0 --save-frames
```

예제 ID `00012`가 해당 서브셋에 없으면 매니페스트에서 ID를 선택한다. `--save-frames`는 점수 Parquet을 저장하며 원본 이미지를 내보내지 않는다. 입력·출력 경로와 모든 모델·평가 파라미터는 `configs/default.yaml`에서 변경한다. 기본 config의 상대 경로는 저장소 루트에 대해 해석한다. 별도 위치의 config는 실행 작업 디렉터리에 대해 해석한다.

CPU 분석은 15 fps, 짧은 변 256 px다. 양성은 사건 전 8초부터 사건 후 1초까지 읽되, **모든 사건 메트릭에는 `t < time_of_event`만 사용**한다. 음성은 앞뒤 1초를 제외한 중간의 최대 9초를 읽는다. 따라서 양성과 음성의 메트릭 노출 길이는 다를 수 있다. 이 실험은 라벨로 정한 구간의 탐색적 평가이며 지속 주행 성능 추정은 아니다.

시간은 `frame_index / actual_fps + video.time_offset_s`이다. offset을 바꾸면 프레임의 시계가 라벨에 대해 이동하며, 양성 구간 선택에도 같은 보정을 적용한다. 프레임을 복제하거나 사건 이후 값을 보간하지 않는다. `s_at_event`는 사건 직전 마지막 관측 점수다.

## Calibration and metrics

- ID 단위로 각 라벨의 20%를 validation, 나머지 80%를 evaluation으로 고정한다.
- `S_div`, `S_rad`, `S_blob`의 평균/표준편차는 **validation 음성 프레임만** 사용한다. 클립별 재표준화는 하지 않는다.
- `S_raw = z(S_div) + z(S_rad) + 0.5*z(S_blob)`에 alpha 0.3의 causal EMA를 적용한다. 최초 프레임은 흐름 쌍이 없으므로 집계에서 제외한다.
- validation 음성의 클립 최대값에서 `S > theta` 기준 FPR이 0.10 이하가 되는 가장 낮은 관측 임계값을 선택한다. 10개 음성 validation이면 허용 위양성은 최대 1개다. evaluation FPR이 0.10 이하라는 보장은 없다.
- 리포트의 주 결과는 evaluation 80개이며, validation 20개와 전체 100개는 별도 표로 제공한다. AUROC/AUPRC는 사건 이전 클립 최대 점수로 계산하고, AUPRC는 average precision 정의다.
- `lead_to_alert_s = time_of_alert - t_peak`는 **최대점의 리드 타임**이다. 첫 임계값 통과 시각이 아니다. 음수이면 최대점이 사람의 alert보다 늦다. 미검출 양성도 최대점 리드 통계에 포함되므로 hit rate와 함께 읽어야 한다.
- early hit 500/1000/1500 ms는 사건에서 해당 시간만큼 앞선 시점까지 한 번이라도 임계값을 넘었는지다. 분석 구간 밖인 창은 결측으로 두고 분모를 기록한다. Nexar 공식 test mAP의 복제 점수는 아니다.

점수는 arbitrary unit이다. 초기 blob 출현은 이전 지름이 없어 증가 점수를 0으로 두며, 부드러운 중앙 가중치의 가장자리는 0.3으로 제한한다. Farneback 입력은 0–1 float에서 0–255 uint8로 변환한다. 정규화된 낮은 밝기 범위를 그대로 넣어 흐름이 사라지는 오류를 방지한다.

## Outputs

`outputs/v0/REPORT.md` 및 `REPORT.html`에 설정, ID, 숫자, 해석, 재현 명령을 저장한다.

| 파일 | 내용 |
|---|---|
| `clip_scores.csv` | 요청한 모든 ID, 라벨, 분할, 점수·리드·히트·상태 |
| `threshold.yaml` | theta, 실제 validation FPR, 분할 ID, 음성 표준화 통계 |
| `metrics.json` | evaluation/validation/전체 메트릭과 유효 분모 |
| `manifest.csv`, `config.yaml`, `environment.json` | 실행에 사용한 입력과 환경 |
| `frames/{video_id}.parquet` | 양성 전체 및 위양성 음성의 점수 시계열 |
| `figures/` | 양성·음성 12개 시계열, 리드 히스토그램, ROC/PR, 진단 패널, 실패 태그 |
| `diagnostics/` | 원본 대신 전체 프레임을 강하게 블러한 이미지와 flow/radial |

음성 정상 클립의 임시 프레임 점수는 `_report_frames/`에 두고 리포트가 그림을 만든 뒤 삭제한다. 실패 태그는 원인 정답이 아닌 공개된 휴리스틱이며 `src/fly_ttc/eval/failure_tags.py`와 config에 조건을 명시한다. 같은 클립에 여러 태그가 붙을 수 있다. 화면 중심 대리 지표의 실패를 실제 초파리의 능력으로 일반화하지 않는다. 대상 물체에 가시적인 팽창 단서가 있었는지를 확인해야 하며, 카메라 흔들림·옆방향 흐름 반응도 별도 실패로 기록한다.

재실행은 임시 디렉터리에서 성공한 뒤 결과 경로로 이동한다. 기존 결과는 같은 부모 디렉터리의 `.v0.previous-*`에 보존한다. 실패한 실행의 `.v0.staging-*`는 완료된 결과가 아니다.

## Tests

실데이터 성능 주장 전에 `pytest -q`를 통과해야 한다. 실제 Farneback으로 중앙 검은 원 팽창, 수축, 1 px/frame 수평 격자, 전역 단색 깜빡임, 오른쪽 가장자리 팽창을 비교한다. 팽창의 증가 추세는 시간 블록 평균과 순위 상관으로 확인한다. 광학 흐름의 프레임별 변동까지 완전 단조라고 가정하지 않는다. 해석적 흐름의 부호와 Sobel 스케일, 상태 reset, validation 표준화, 임계값 동점, 사건 경계, 스키마, FPS 시계를 추가 검증한다.

평행이동에서도 `relu(radial)`은 양수가 될 수 있다. 특정 합성 속도에서 loom보다 낮다는 테스트가 모든 카메라 움직임에 대한 불변성을 보장하지 않는다. 단색 깜빡임 테스트도 실제 장면의 자동노출 변동 전체를 보장하지 않는다. 합성 자극은 `tests/`와 `outputs/synthetic/`에만 둔다.

## Limitations

- 이 구현은 화면 중심의 팽창 대리 지표다. 사고 유형만으로 검출 가능성을 단정할 수 없으며, 대상 물체의 가시적 팽창 단서와 위치를 확인해야 한다. 실제 초파리 전체의 능력을 평가한 결과가 아니다.
- Nexar 양성은 collision과 near-miss를 구분하지 않는다.
- 광학 흐름은 야간·비·와이퍼·큰 카메라 흔들림에 깨진다.
- 점수는 TTC의 물리적 초가 아니다. 상대적 looming proxy다.
- 임계값은 이 서브셋 음성 FPR에 맞춰져 있어 다른 도메인에 그대로 못 옮긴다.
- 이 파이프라인은 자율주행 스택을 대체하지 않는다. 희귀 이벤트 마이닝/라벨 제안용이다.

## Next research

다음 기술적 질문은 논문 자극 조건과 대조해 밝기 극성·RF 사이 위치·샘플링 간격에 따른 국소 응답 실패를 설명할 수 있는가이다. 새 모듈도 `proxy_not_zhao_star`로 표시하며 Zhao 전체 모델을 재현했다고 주장하지 않는다. 합성 선택성이 충분히 확보된 경우에만 별도 프로토콜의 사람 정답 ROI 실험을 검토한다. 범용 사고확률·자동 탐지기·전체 뇌 시뮬레이션으로 확대하지 않는다.
