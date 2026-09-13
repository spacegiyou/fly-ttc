"""Read-only Korean reports for the finite local visual-reflex stimulus matrix."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .report import _html_report, _number, _table

IDENTITY = "proxy_not_zhao_star"
FIGURES = ["all_model_metrics.png", "core_schedule_selectivity.png",
           "localization_coverage.png", "sampling_sensitivity.png", "fixed_stimulus_maps.png"]


def _label(model):
    if model.startswith("direction_permutation_"):
        return "Direction perm. " + model.rsplit("_", 1)[-1]
    return {IDENTITY: "Local opponency (proxy)", "simple_motion_energy": "Simple motion energy",
            "timed_global_flow": "Global flow, dt scaled", "v0_legacy_unscaled": "Legacy v0, per pair"}[model]


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _localization_rows(scores, models, tolerance):
    rows = []
    local = scores.loc[scores.split.eq("challenge") & scores.suite.eq("position") & scores.kind.eq("loom")]
    for model in models:
        part = local.loc[local.model.eq(model)]
        if model in {"timed_global_flow", "v0_legacy_unscaled"}:
            continue
        correct = part.detected & part.localization_error_px.le(tolerance)
        rows.append(dict(model=model, requested=len(part), localized=int(correct.sum()),
                         detected_wrong=int((part.detected & ~correct).sum()),
                         missed=int((~part.detected).sum()),
                         zero_response=int(part.s_peak.eq(0).sum())))
    return rows


def _sampling_rows(scores, models):
    """Match fixed core stimuli to 60 Hz, retaining each model's active support."""
    core = scores.loc[scores.split.eq("challenge") & scores.suite.eq("core")]
    rows = []
    for model in models:
        part = core.loc[core.model.eq(model)]
        reference = part.loc[part.schedule.eq("60hz"), ["kind", "contrast", "polarity", "s_peak"]].rename(columns={"s_peak": "reference"})
        pairs = part.loc[~part.schedule.eq("60hz")].merge(reference, on=["kind", "contrast", "polarity"], validate="many_to_one")
        floor = max(float(reference.reference.max()) * 1e-6, 1e-12)
        for schedule, group in pairs.groupby("schedule"):
            active = group.loc[group.reference > floor]
            error = (active.s_peak - active.reference).abs() / active.reference
            rows.append(dict(model=model, schedule=schedule, n_pairs=len(active),
                             excluded=len(group) - len(active), floor=floor,
                             median=float(error.median()) if len(active) else None,
                             maximum=float(error.max()) if len(active) else None))
    return rows


def _plot_overview(metrics, models, path):
    frame = pd.DataFrame(metrics["results"]).query("split == 'challenge'").set_index("model").loc[models]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True, layout="constrained")
    positions = np.arange(len(models))
    for ax, key, title, color in zip(axes, ["auroc", "expansion_recall", "nuisance_fpr"],
                                     ["Expansion vs nuisance: AUROC", "Expansion detection fraction", "Nuisance detection fraction"],
                                     ["#286d93", "#2b8a72", "#b96341"]):
        values = frame[key].to_numpy()
        ax.barh(positions, values, color=color, alpha=0.85)
        for pos, value in zip(positions, values):
            ax.text(min(value + .018, .9), pos, f"{value:.3f}", va="center", fontsize=9)
        ax.set(xlim=(0, 1.08), title=title, xlabel="Fraction / rank area")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(positions, [_label(m) for m in models])
    axes[0].invert_yaxis()
    axes[0].axvline(.5, color="#888888", linestyle="--", linewidth=1)
    fig.suptitle(f"All challenge conditions | {int(frame.n.iloc[0])} correlated synthetic cases per model\nNo road-accident evaluation or population confidence interval", fontsize=13)
    _save(fig, path)


def _plot_schedule(scores, models, schedules, path):
    core = scores.loc[scores.split.eq("challenge") & scores.suite.eq("core")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True, layout="constrained")
    for ax, label, title in zip(axes, [1, 0], ["Expansion detection fraction", "Nuisance detection fraction"]):
        array = np.full((len(models), len(schedules)), np.nan)
        for i, model in enumerate(models):
            for j, schedule in enumerate(schedules):
                group = core.loc[core.model.eq(model) & core.schedule.eq(schedule) & core.label.eq(label)]
                if len(group):
                    array[i, j] = group.detected.mean()
                    ax.text(j, i, f"{int(group.detected.sum())}/{len(group)}", ha="center", va="center", fontsize=9,
                            color="white" if array[i, j] > .55 else "#1b2730")
        im = ax.imshow(array, vmin=0, vmax=1, cmap="Blues" if label else "Oranges", aspect="auto")
        ax.set_xticks(range(len(schedules)), schedules)
        ax.set_yticks(range(len(models)), [_label(m) for m in models])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.75, label="Fraction")
    fig.suptitle("Core matrix by sampling schedule | matched kinds, contrasts and polarities\nEach cell includes every requested case, including zero responses", fontsize=12)
    _save(fig, path)


def _plot_localization(scores, models, tolerance, path):
    rows = _localization_rows(scores, models, tolerance)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    positions = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, label, color in [("localized", "Detected + localized", "#2b8a72"),
                              ("detected_wrong", "Detected, outside tolerance", "#d49b42"),
                              ("missed", "Missed / zero response", "#bfc9d0")]:
        values = np.array([r[key] / r["requested"] if r["requested"] else 0 for r in rows])
        axes[0].barh(positions, values, left=left, label=label, color=color)
        left += values
    axes[0].set_yticks(positions, [_label(r["model"]) for r in rows])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 1), xlabel="Fraction of all requested off-center looms",
                title="All seven readouts with spatial support")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=8)
    local = scores.loc[scores.split.eq("challenge") & scores.suite.eq("position") & scores.kind.eq("loom") & scores.model.eq(IDENTITY)]
    grouped = list(local.groupby(["center_x", "center_y"], sort=True))
    for polarity, shift, color, name in [(-1, -.18, "#286d93", "Dark disk (-1)"), (1, .18, "#d49b42", "Bright disk (+1)")]:
        parts = [group.loc[group.polarity.eq(polarity)] for _, group in grouped]
        counts = [int((group.detected & group.localization_error_px.le(tolerance)).sum()) for group in parts]
        values = [n/len(group) if len(group) else 0 for n, group in zip(counts, parts)]
        bars = axes[1].bar(np.arange(len(grouped))+shift, values, width=.34, color=color, label=name)
        for bar, n, group in zip(bars, counts, parts):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025, f"{n}/{len(group)}", ha="center", fontsize=9)
    axes[1].set_xticks(range(len(grouped)), [f"({x:.2f}, {y:.2f})" for (x, y), _ in grouped])
    axes[1].set(ylim=(0, 1.12), xlabel="Stimulus center (image fraction)", ylabel="Detected and localized fraction",
                title="Identity proxy: every location and both polarities")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=8, ncols=2)
    fig.suptitle(f"Localization coverage | tolerance {tolerance:.2f} px | all schedules and polarities\nGlobal scalar controls have no location output and are not assigned localization accuracy", fontsize=12)
    _save(fig, path)


def _plot_sampling(rows, models, schedules, path):
    schedules = [s for s in schedules if s != "60hz"]
    lookup = {(r["model"], r["schedule"]): r for r in rows}
    values = np.full((len(models), len(schedules)), np.nan)
    for i, model in enumerate(models):
        for j, schedule in enumerate(schedules):
            value = lookup.get((model, schedule), {}).get("median")
            if value is not None:
                values[i, j] = value
    fig, ax = plt.subplots(figsize=(10, 6.8), layout="constrained")
    finite = values[np.isfinite(values)]
    maximum = float(np.log1p(finite.max())) if len(finite) else 1.0
    im = ax.imshow(np.log1p(values), aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(maximum, 1e-12))
    for i, model in enumerate(models):
        for j, schedule in enumerate(schedules):
            row = lookup.get((model, schedule), {})
            label = "N/A" if not np.isfinite(values[i, j]) else f"{values[i,j]:.0%}\nn={row['n_pairs']}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(values[i,j]) and np.log1p(values[i,j]) > .65*maximum else "#242424")
    ax.set_xticks(range(len(schedules)), schedules)
    ax.set_yticks(range(len(models)), [_label(m) for m in models])
    ax.set_title("Observed median relative peak error versus 60 Hz\nSame core stimuli; each model excludes its own near-zero reference responses", fontsize=12)
    ax.set_xlabel("Text is relative error; color uses log(1 + relative error)")
    fig.colorbar(im, ax=ax, label="log(1 + median relative error)", shrink=.8)
    _save(fig, path)


def _plot_maps(samples, path):
    selected = list(samples.items())
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    maximum = max((float(np.max(v["map"])) for _, v in selected), default=0)
    maximum = max(maximum, 1e-12)
    im = None
    for ax, (case_id, sample) in zip(axes.flat, selected):
        array = np.asarray(sample["map"], dtype=float)
        centers = np.asarray(sample["centers"], dtype=float)
        im = ax.imshow(array, cmap="viridis", vmin=0, vmax=maximum, origin="upper")
        ax.set_xticks(range(array.shape[1]), [f"{x:.0f}" for x in centers[0, :, 0]])
        ax.set_yticks(range(array.shape[0]), [f"{y:.0f}" for y in centers[:, 0, 1]])
        ax.set(title=f"{sample['kind']}\npeak unit={array.max():.3g}", xlabel="RF center x (px)", ylabel="RF center y (px)")
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=.65, label="Identity proxy unit activation (common linear scale)")
    fig.suptitle("Prespecified examples: core, 30 Hz, contrast 0.8, dark polarity\nAll nine stimulus kinds; map at each case's post-warmup maximum, no outcome selection", fontsize=12)
    _save(fig, path)


def make_report(out_dir) -> Path:
    """Generate presentation artifacts from a complete run; never rewrite scores."""
    out = Path(out_dir).resolve()
    state = json.loads((out / "run_state.json").read_text())
    if state.get("status") != "complete":
        raise ValueError("Local-reflex report requires a complete run")
    config = yaml.safe_load((out / "config.yaml").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    samples = json.loads((out / "sample_maps.json").read_text())
    scores = pd.read_csv(out / "case_scores.csv")
    models = [IDENTITY, *sorted(protocol["direction_permutations"]), "simple_motion_energy", "timed_global_flow", "v0_legacy_unscaled"]
    if len(models) != 9 or set(scores.model) != set(models):
        raise ValueError("Report requires the declared identity, five permutations and three controls")
    if scores.duplicated(["case_id", "model"]).any() or not scores.groupby("case_id").model.nunique().eq(9).all():
        raise ValueError("Incomplete or duplicate model-case rows")
    if len(scores) != state["n_rows"] or scores.case_id.nunique() != protocol["n_cases"]:
        raise ValueError("Scores do not match completed run counts")
    if scores.detected.isna().any() or not scores.detected.isin([True, False]).all():
        raise ValueError("Missing or invalid detection decisions")
    if not np.isfinite(scores[["s_peak", "theta"]]).all().all():
        raise ValueError("Nonfinite score or cutoff")
    fixed = scores.loc[scores.model.eq(IDENTITY) & scores.suite.eq("core") & scores.schedule.eq("30hz") & scores.contrast.eq(.8) & scores.polarity.eq(-1)]
    if set(samples) != set(fixed.case_id) or set(fixed.kind) != set(config["matrix"]["kinds"]):
        raise ValueError("Sample maps must contain every prespecified stimulus kind")
    traces = pd.read_parquet(out / "traces.parquet", columns=["case_id", "model", "t_s", "dt_s", "S"])
    if set(traces.case_id) != set(scores.case_id) or set(traces.model) != set(models) or traces.duplicated(["case_id", "model", "t_s"]).any():
        raise ValueError("Invalid trace case membership or duplicate timestamps")
    trace_pairs = set(map(tuple, traces[["case_id", "model"]].drop_duplicates().to_numpy()))
    if trace_pairs != set(map(tuple, scores[["case_id", "model"]].to_numpy())):
        raise ValueError("Missing model-case traces")
    if not np.isfinite(traces[["t_s", "dt_s", "S"]]).all().all() or traces.dt_s.lt(0).any():
        raise ValueError("Invalid trace times or scores")
    schedules = config["matrix"]["schedules"]
    tolerance = metrics["localization"]["tolerance_px"]
    local = _localization_rows(scores, models, tolerance)
    sampling = _sampling_rows(scores, models)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    _plot_overview(metrics, models, figures / FIGURES[0])
    _plot_schedule(scores, models, schedules, figures / FIGURES[1])
    _plot_localization(scores, models, tolerance, figures / FIGURES[2])
    _plot_sampling(sampling, models, schedules, figures / FIGURES[3])
    _plot_maps(samples, figures / FIGURES[4])

    records = pd.DataFrame(metrics["results"])
    challenge = records.loc[records.split.eq("challenge")].set_index("model")
    calibration = records.loc[records.split.eq("calibration")].set_index("model")
    counts = scores.loc[scores.model.eq(IDENTITY)].groupby(["split", "suite"]).size()
    identity = challenge.loc[IDENTITY]
    local_identity = scores.loc[scores.model.eq(IDENTITY) & scores.split.eq("challenge") & scores.suite.eq("position") & scores.kind.eq("loom")]
    polarity_rows = [["dark (-1)" if polarity == -1 else "bright (+1)", len(group), int(group.detected.sum()),
                      int((group.detected & group.localization_error_px.le(tolerance)).sum()), int((~group.detected).sum()),
                      int(group.s_peak.eq(0).sum())] for polarity, group in local_identity.groupby("polarity")]
    status = "충족" if metrics["advance_to_roi"] else "미충족"
    gates = metrics["gates"]
    criteria = config["assessment"]
    gate_rows = [
        ["모든 core 일정에서 팽창 recall", f">= {criteria['minimum_expansion_recall']:.0%}",
         "; ".join(f"{r['schedule']} {_number(r['recall'])}" for r in metrics["identity_core_by_schedule"]), gates["all_schedules_recall"]],
        ["모든 core 일정에서 nuisance FPR", f"<= {criteria['maximum_nuisance_false_positive_rate']:.0%}",
         "; ".join(f"{r['schedule']} {_number(r['fpr'])}" for r in metrics["identity_core_by_schedule"]), gates["all_schedules_nuisance_fpr"]],
        ["모든 off-center loom 중 검출+위치 일치", f">= {criteria['minimum_localized_detection_fraction']:.0%}",
         f"{metrics['localization']['n_detected_and_localized']}/{metrics['localization']['n_requested']} ({metrics['localization']['fraction']:.1%})", gates["localized_detection"]],
        ["활성 60Hz 기준 대비 median peak 상대 오차", f"<= {criteria['maximum_median_sampling_relative_error']:.0%}",
         _number(metrics["sampling"]["median_relative_peak_error"]), gates["median_sampling_stability"]],
    ]
    condition_rows = metrics["identity_by_suite_schedule"]
    condition_recalls = [r["recall"] for r in condition_rows if r["recall"] is not None]
    condition_fprs = [r["fpr"] for r in condition_rows if r["fpr"] is not None]
    gate_rows[2:2] = [
        ["모든 suite×일정의 팽창 recall", f">= {criteria['minimum_expansion_recall']:.0%}",
         f"최소 {_number(min(condition_recalls))}", gates["all_suites_schedules_recall"]],
        ["모든 suite×일정의 nuisance FPR", f"<= {criteria['maximum_nuisance_false_positive_rate']:.0%}",
         f"최대 {_number(max(condition_fprs))}", gates["all_suites_schedules_nuisance_fpr"]],
    ]
    body = ["# Local visual reflex: 합성 단계 A 보고서", "",
            f"**고정된 진전 조건: {status}.** `proxy_not_zhao_star`의 challenge 팽창 recall은 **{identity.expansion_recall:.1%}**, nuisance FPR은 **{identity.nuisance_fpr:.1%}**다. 아래 결과는 상관된 합성 진단 행렬의 응답이며 Nexar AUROC나 실도로 성능이 아니다.",
            "이 구현은 시간 기반 EMD와 국소 방향 대립을 조합한 공학적 proxy다. 실제 connectome, Zhao STAR의 충실한 재현, GF 뉴런 또는 초파리 전체 뇌를 실행하지 않았다. 생물학적 구조의 우위도 입증하지 않았다. 이 보고서는 원래 점수·임계값·진전 조건을 변경하지 않는다.",
            "## 범위와 사전 고정", "", _table(["split", "suite", "사례 수"], [[split, suite, int(n)] for (split, suite), n in counts.items()]),
            f"총 **{protocol['n_cases']}개 자극 × {len(models)}개 모델 = {len(scores)}개 결과행**, 저장 시계열 **{len(traces):,}행**이다. 동일 자극의 극성·대비·속도·위치·샘플링 변형은 서로 상관돼 있다. 무작위 모집단의 독립 표본이나 통계적 확증 자료로 취급하지 않고 신뢰구간을 붙이지 않았다.",
            "팽창 정답은 `loom`, `linear_expand`, `loom_on_background`이며 나머지는 nuisance다. 실제 접근 기하의 loom과 임의 선형 확대는 구분해 보존했다. calibration은 challenge 이전에 실행됐으며 각 모델은 같은 합성 음성 사례에서 자신의 cutoff를 받았다. identity·다섯 방향 permutation·simple_motion_energy의 7개는 EMD 상태와 RF 지지 영역을 공유하는 공학 대조다. 두 Farneback 흐름 대조는 이들과 계산량을 맞춘 모델이 아니다. 실제 신경 연결 교란 실험이 아니며 재학습할 계수는 없다.",
            f"[동결 protocol](protocol.json) · [고정 설정](config.yaml) · [원본 지표](metrics.json) · [모델 정의와 1차 출처](../../../docs/LOCAL_REFLEX_MODEL_SOURCES.md) · [기존 자료 재감사](../../review_response/AUDIT.md)",
            "## 모든 모델의 결과", "",
            _table(["모델", "challenge AUC", "AP", "팽창 recall", "nuisance FPR", "calibration FPR", "cutoff"],
                   [[m, _number(challenge.loc[m, "auroc"]), _number(challenge.loc[m, "ap"]),
                     _number(challenge.loc[m, "expansion_recall"]), _number(challenge.loc[m, "nuisance_fpr"]),
                     _number(calibration.loc[m, "nuisance_fpr"]), f"{float(scores.loc[scores.model.eq(m), 'theta'].iloc[0]):.8g}"] for m in models]),
            "AUC/AP는 이 합성 행렬에서 팽창과 nuisance의 peak 순서를 요약한 기술 통계다. 모델별 출력 단위와 cutoff가 다르므로 raw S의 크기를 모델 간 성능이나 에너지 효율로 비교하지 않는다.",
            f"![All model metrics](figures/{FIGURES[0]})", "## 샘플링 일정별 선택성", "",
            "core에서는 같은 자극 종류·대비·극성 조합을 각 일정으로 표시했다. 수치는 모델마다 고정된 calibration cutoff를 사용한 검출 수/요청 수이며, 응답 0을 포함한 모든 사례가 분모에 남는다.",
            f"![Core schedule selectivity](figures/{FIGURES[1]})",
            "고속 이동·배경 이동·위치 변형도 별도 suite×일정으로 평가했다. core 평균이 다른 조건의 실패를 가리지 않도록 아래 모든 조건의 recall/FPR을 진전 기준에 포함한다. 한 정답만 있는 조건의 반대쪽 지표는 N/A다.",
            _table(["suite", "일정", "양성 수", "nuisance 수", "팽창 recall", "nuisance FPR"],
                   [[r["suite"], r["schedule"], r["n_positive"], r["n_negative"], _number(r["recall"]), _number(r["fpr"])] for r in condition_rows]),
            "## 화면 중심에서 벗어난 자극의 적용 범위", "",
            "아래 위치는 화면 **내부의 off-center** 위치다. 성공한 위치만 고르거나 무응답 사례를 제거하지 않았다. 위치 허용 오차는 사전 정의된 RF 중심 간격이며 예측이 cutoff를 넘고 오차가 허용치 이내인 경우에만 성공으로 센다.",
            _table(["모델", "요청", "검출+위치 일치", "검출/위치 불일치", "미검출", "응답 0 (미검출과 중복 가능)"],
                   [[r["model"], r["requested"], r["localized"], r["detected_wrong"], r["missed"], r["zero_response"]] for r in local]),
            f"위치 허용치는 **{tolerance:.2f}px**다. 두 전역 흐름 대조는 좌표를 출력하지 않아 국소화 점수를 부여하지 않았다.",
            "identity proxy의 동일 off-center loom을 극성별로 나누면 다음과 같다. 전체 평균이 밝은/어두운 원판의 실패 차이를 숨기지 않도록 두 극성의 모든 위치와 일정을 포함했다.",
            _table(["원판 극성", "요청", "검출", "검출+위치 일치", "미검출", "응답 0"], polarity_rows),
            f"![Localization coverage](figures/{FIGURES[2]})", "## 시간 간격과 관측된 sampling 오차", "",
            "시간 기반 코드라고 샘플링 불변성이 자동으로 성립하지 않는다. 같은 연속 자극을 60Hz로 표시한 peak와 비교한 실제 차이를 기록했다. 각 모델의 최대 기준 peak×1e-6(최소 1e-12) 이하인 60Hz 응답은 상대 오차의 분모가 불안정하므로 제외한다. 모델별 활성 기준 사례가 다를 수 있어 이 수치만으로 모델을 순위 매기지 않는다.",
            _table(["모델", "일정", "활성 비교 수", "제외 수", "median 상대 오차", "최대 상대 오차"],
                   [[r["model"], r["schedule"], r["n_pairs"], r["excluded"], _number(r["median"]), _number(r["maximum"])] for r in sampling]),
            f"![Sampling sensitivity](figures/{FIGURES[3]})"]
    core_ids = scores.loc[scores.model.eq(IDENTITY) & scores.suite.eq("core"), ["case_id", "schedule"]]
    clock = traces.loc[traces.model.eq(IDENTITY) & traces.dt_s.gt(0)].merge(core_ids, on="case_id", validate="many_to_one")
    observation = metrics["sampling"]
    geometry = scores.loc[scores.model.eq(IDENTITY) & scores.split.eq("challenge") & scores.disk_fully_visible_fraction.notna()]
    body += [f"identity proxy의 활성 비교 전체 median/p90/최대 상대 오차는 **{_number(observation['median_relative_peak_error'])} / {_number(observation['p90_relative_peak_error'])} / {_number(observation['max_relative_peak_error'])}**다. 60Hz 응답이 거의 0이라 상대 오차에서 제외한 **{observation['excluded_near_zero_pairs']}개 비교** 중 다른 일정에서 cutoff를 넘은 경우는 **{observation['near_zero_reference_other_rate_alerts']}개**다. 이 사례는 상대 오차에서만 제외했고 검출/FPR 분모에서 제거하지 않았다.",
             _table(["일정", "기록된 dt 최솟값 (s)", "중앙값 (s)", "최댓값 (s)"],
                    [[schedule, _number(g.dt_s.min(), 6), _number(g.dt_s.median(), 6), _number(g.dt_s.max(), 6)] for schedule, g in clock.groupby("schedule")]),
             "여기서 dt는 합성 렌더링에 전달한 정확한 시각 차이다. legacy v0는 이 시각을 받지 않고 프레임 쌍당 변위를 사용한다. timed_global_flow는 displacement×dt_ref/dt와 시간 EMA를 사용하지만 정확한 흐름이나 생물학적 선택성을 보증하지 않는다.",
             "## 화면 경계와 사전 지정 응답 지도", "",
             f"원판 가시성 기록이 있는 challenge **{len(geometry)}개** 중 평가 시각의 일부에서 원판이 화면 경계에 잘린 사례는 **{int(geometry.disk_fully_visible_fraction.lt(1-1e-12).sum())}개**다. 이 사례도 지표의 분모에 남겼다. 아래 비율은 각 사례의 warmup 이후 시각 중 원판 전체가 화면에 있는 비율을 뜻하며, 데이터가 없는 전역 자극에 원판 가시성을 부여하지 않았다.",
             _table(["suite", "원판 사례 수", "일부 clipping 사례", "온전한 원판 관측 비율 최솟값"],
                    [[suite, len(group), int(group.disk_fully_visible_fraction.lt(1-1e-12).sum()), _number(group.disk_fully_visible_fraction.min())] for suite, group in geometry.groupby("suite")]),
             "core의 30Hz, 대비 0.8, dark polarity(-1)라는 결과 관찰 전 조건으로 아홉 자극을 모두 표시한다. 각 그림은 warmup 이후 자기 사례의 최대 응답 시점이며 모든 패널에 동일한 선형 색 범위를 사용한다. 이 패널로 전체 성공률을 대신하지 않는다.",
             f"![Fixed stimulus maps](figures/{FIGURES[4]})", "## 진전 조건과 다음 단계", "",
             _table(["사전 기준", "목표", "관측", "충족"], [[*r[:3], "예" if r[3] else "아니오"] for r in gate_rows]),
             f"사전 조건의 AND 결과는 **{status}**다. " + ("이는 다음 진단 단계로 검토할 수 있다는 공학적 조건 충족이며 실영상 효용의 증거는 아니다." if metrics["advance_to_roi"] else "이 설정을 성공한 국소 반사 모듈로 채택하거나 실영상 자동 탐지로 확대할 근거가 없다. 조건·실패 유형을 보존하고 별도 설계에서 다시 진단해야 한다."),
             "사람이 대상 물체를 지정한 ROI 실험, 자동 객체 검출/추적, 실제 사고 예측 및 배포 평가는 **이번 실행에서 수행하지 않았다**. 실제 연결 구조의 학습된 대조군 실험도 하지 않았으므로 방향 permutation 대비 수치 차이를 connectome의 장점이라고 해석하지 않는다. 기존 200개 Nexar 자료를 다시 튜닝하거나 미사용 확인 표본으로 재포장하지 않았다.",
             "재생성: `.venv/bin/python scripts/run_local_reflex.py --config configs/local_reflex.yaml --phase report`. 이 명령은 저장된 점수로 보고서만 만든다.", ""]
    if (out / "independent_checks.json").exists():
        body.append("[독립 수치 검증: 전체 최고값·시각·적분·보정·지표·동결 해시](independent_checks.json)")
    if (out / "validation_checks.json").exists():
        body.append("[전체 테스트와 실행 검증 기록](validation_checks.json)")
    text = "\n\n".join(body)
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    rendered = _html_report(text, FIGURES).replace("<title>Fly-TTC v0 Report</title>", "<title>Local visual reflex — synthetic Stage A</title>")
    (out / "REPORT.html").write_text(rendered, encoding="utf-8")
    return out / "REPORT.md"
