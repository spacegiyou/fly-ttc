"""Save reproducible synthetic diagnostics without creating fake dataset media.

Run from the repository root: python tests/generate_synthetic_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fly_ttc.models.v0_expansion import ExpansionScorer
from synthetic_stimuli import stimulus_clips


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "outputs" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    summaries, traces = [], []
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for name, frames in stimulus_clips().items():
        scorer = ExpansionScorer()
        trace = pd.DataFrame([scorer.update(frame) for frame in frames])
        trace.insert(0, "frame", np.arange(len(trace)))
        trace.insert(0, "stimulus", name)
        traces.append(trace)
        summaries.append({"stimulus": name, "n_frames": len(trace), "late_mean_S": float(trace.S.iloc[-15:].mean()), "peak_S": float(trace.S.max()), "median_horizontal_flow": float(trace.flow_horizontal.iloc[5:].median())})
        axis.plot(trace.frame, trace.S, label=name)
    axis.set(xlabel="Frame index (synthetic)", ylabel="Uncalibrated S (unit component statistics)", title="Farneback v0: synthetic stimulus controls")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "synthetic_traces.png", dpi=140)
    plt.close(figure)
    pd.concat(traces, ignore_index=True).to_csv(output / "synthetic_scores.csv", index=False)
    summary = {row["stimulus"]: row for row in summaries}
    (output / "synthetic_summary.json").write_text(json.dumps({"kind": "synthetic_only", "opencv_version": cv2.__version__, "normalization": "none_unit_statistics_not_for_real_video_claims", "stimuli": summary}, indent=2) + "\n", encoding="utf-8")
    table = "\n".join(f"| {row['stimulus']} | {row['n_frames']} | {row['late_mean_S']:.6f} | {row['peak_S']:.6f} |" for row in summaries)
    report = f"""# Fly-TTC v0 Synthetic Report

이 점수는 초파리 LPLC2/Giant Fiber에서 영감을 받은 고정 팽창 검출기다. 전체 초파리 뇌를 시뮬레이션한 것이 아니다.

## Setup

OpenCV {cv2.__version__} Farneback를 실제로 실행했다. 자극 5종, 각 60프레임, 192×192픽셀이다. 중앙 검은 원 반지름은 12→70픽셀로 이차 증가하며, 수축 자극은 같은 프레임의 역순이다. 격자는 프레임당 1픽셀 수평 이동하고, 가장자리 원은 같은 속도로 오른쪽 화면 경계에서 커진다. 밝기 자극은 공간적으로 균일하며 0.2와 0.8 사이를 번갈아 바꾼다.

## Results

단위 평균 0·표준편차 1의 **보정 전 합성 점수**이며 Nexar 성능 수치가 아니다. 실제 영상의 표준화는 음성 validation 프레임으로만 계산한다. 마지막 15프레임 평균을 비교했다.

| Stimulus | Frames | Late mean S | Peak S |
|---|---:|---:|---:|
{table}

![Synthetic traces](synthetic_traces.png)

## Limits

v0의 rectified radial 항은 평행 이동에도 반응할 수 있다. 이 테스트는 명시된 속도와 위치에서 중앙 팽창이 더 큰지 확인하며, 모든 속도·방향의 이동을 제거한다는 주장은 하지 않는다. 픽셀 경계에 따른 작은 출렁임 때문에 상승 추세는 연속 10프레임 블록 평균과 프레임 순위 상관으로 검사한다.

## How to reproduce

```bash
.venv/bin/python -m pytest tests/test_expansion.py -q
.venv/bin/python tests/generate_synthetic_report.py
```
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
