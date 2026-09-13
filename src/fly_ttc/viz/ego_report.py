"""Korean report for the prespecified ego-motion proxy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from fly_ttc.eval.comparison import MODELS, SPLITS, summarize_comparison
from fly_ttc.eval.metrics import valid_clips
from .report import _html_report, _number, _table


LABELS = {
    "v0_original": "Original v0", "raw_flow_no_blob": "Raw flow (no blob)",
    "affine_residual_no_blob": "Affine residual (no blob)", "frame_difference": "Frame difference",
}
COLORS = {model: color for model, color in zip(MODELS, ("#607d8b", "#3b73b9", "#c45b36", "#328975"))}


def _finish(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _point_interval(ax, value, low, high, position, color, offset=0):
    if value is None or not np.isfinite(value):
        ax.text(0.02, position + offset, "unavailable", va="center", color="#888888")
        return
    ax.scatter(value, position + offset, color=color, s=42, zorder=3)
    if low is not None and high is not None:
        ax.hlines(position + offset, low, high, color=color, linewidth=2)


def _plot_metrics(results, path, dpi):
    table = {r["model"]: r for r in results if r["split"] == "confirmation"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, model in enumerate(MODELS):
        row = table[model]
        _point_interval(axes[0], row["auroc"], row["auroc_ci_low"], row["auroc_ci_high"], index, COLORS[model])
        _point_interval(axes[1], row["fpr"], row["fpr_ci_low"], row["fpr_ci_high"], index, "#c45b36", -0.12)
        _point_interval(axes[1], row["true_positive_rate"], row["tpr_ci_low"], row["tpr_ci_high"], index, "#3b73b9", 0.12)
    for ax in axes:
        ax.set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(MODELS) - 0.5, -0.5)
        ax.grid(axis="x", alpha=0.2)
    axes[0].axvline(0.5, color="#888888", linestyle="--", linewidth=1)
    axes[0].set(title="New clip-disjoint confirmation: AUROC", xlabel="AUROC with stratified bootstrap 95% interval")
    axes[1].axvline(0.1, color="#888888", linestyle="--", linewidth=1)
    axes[1].scatter([], [], c="#c45b36", label="FPR (lower is better)")
    axes[1].scatter([], [], c="#3b73b9", label="TPR (higher is better)")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set(title="Frozen thresholds: FPR / TPR", xlabel="Clip-level rate with Wilson 95% interval")
    _finish(fig, path, dpi)


def _plot_confusion(results, path, dpi):
    table = {r["model"]: r for r in results if r["split"] == "confirmation"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7))
    for ax, model in zip(axes, MODELS):
        row = table[model]
        tp, fp = row["tpr_count"], row["fpr_count"]
        n_pos, n_neg = row["true_positive_rate_n"], row["fpr_n"]
        counts = np.asarray([[n_neg - fp, fp], [n_pos - tp, tp]])
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(n_pos, n_neg, 1))
        for (y, x), value in np.ndenumerate(counts):
            ax.text(x, y, str(value), ha="center", va="center", fontsize=15,
                    color="white" if value > max(n_pos, n_neg, 1) * 0.5 else "#233c56")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
               yticklabels=["Negative", "Positive"], xlabel="Predicted label", ylabel="Actual label",
               title=f"{LABELS[model]}\nn={n_pos+n_neg}")
    fig.suptitle("Confirmation confusion counts at frozen model-specific thresholds", fontsize=13)
    _finish(fig, path, dpi)


def _paired_rows(clips, left, right, split):
    valid = valid_clips(clips)
    valid = valid[valid.split.eq(split)]
    return valid[valid.model.eq(left)].merge(
        valid[valid.model.eq(right)], on="video_id", how="inner", suffixes=("_left", "_right"),
        validate="one_to_one",
    )


def _plot_pairs(clips, path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axes, ("exploratory", "confirmation")):
        pair = _paired_rows(clips, "raw_flow_no_blob", "affine_residual_no_blob", split)
        for label, color, title in ((0, "#437b9d", "Negative"), (1, "#ca6f3d", "Positive")):
            subset = pair[pair.label_left.eq(label)]
            ax.scatter(subset.s_peak_left, subset.s_peak_right, color=color, alpha=0.65, label=f"{title} n={len(subset)}", s=24)
        if len(pair) and "theta_used_left" in pair and "theta_used_right" in pair:
            ax.axvline(pair.theta_used_left.iloc[0], color="#777777", linestyle="--", linewidth=1)
            ax.axhline(pair.theta_used_right.iloc[0], color="#777777", linestyle="--", linewidth=1)
        ax.set(xlabel="Raw flow peak (its calibrated units)", ylabel="Residual peak (its calibrated units)",
               title=f"{split.capitalize()} paired clip peaks | n={len(pair)}")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    _finish(fig, path, dpi)


def _plot_known_negatives(clips, path, dpi):
    ids = ["01538", "01510", "01250"]
    rows = valid_clips(clips)
    rows = rows[rows.video_id.isin(ids) & rows.label.eq(0) & rows.split.ne("confirmation")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7))
    for ax, model in zip(axes, MODELS):
        subset = rows[rows.model.eq(model)].set_index("video_id")
        for index, video_id in enumerate(ids):
            if video_id in subset.index:
                peak = float(subset.loc[video_id, "s_peak"])
                ax.scatter(peak, index, color=COLORS[model], s=42)
                ax.annotate(f"{peak:.2f}", (peak, index), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        if len(subset) and "theta_used" in subset:
            theta = float(subset.theta_used.iloc[0])
            ax.axvline(theta, color="#777777", linestyle="--", label=f"theta={theta:.2f}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set(yticks=range(3), yticklabels=ids, ylim=(-0.5, 2.7), xlabel="Model-specific peak units", title=LABELS[model])
        ax.grid(axis="x", alpha=0.2)
        ax.margins(x=0.2)
    fig.suptitle("Previously highlighted negatives (post-hoc examples; not a success-rate sample)", fontsize=12)
    _finish(fig, path, dpi)


def _plot_motion_fields(run_dir, path, dpi):
    """Use only recorded vector fields with one shared magnitude/vector scale."""
    snapshots = []
    for cache_path in sorted((run_dir / "diagnostics").glob("*.npz")):
        with np.load(cache_path, allow_pickle=False) as cache:
            keys = ("raw_flow", "background_flow", "residual_flow")
            if not all(key in cache for key in (*keys, "t")):
                continue
            for index, time in enumerate(cache["t"]):
                snapshots.append((cache_path.stem, float(time), [cache[key][index] for key in keys]))
    if not snapshots:
        return False
    magnitudes = [np.hypot(field[..., 0], field[..., 1]) for _, _, fields in snapshots for field in fields]
    vmax = max(float(np.percentile(np.concatenate([m.ravel() for m in magnitudes]), 99)), 1e-6)
    fig, axes = plt.subplots(len(snapshots), 3, figsize=(12, 3.0 * len(snapshots)), squeeze=False)
    for row, (video_id, time, fields) in enumerate(snapshots):
        for column, (field, title) in enumerate(zip(fields, ("Raw flow", "Estimated global flow", "Residual flow"))):
            ax = axes[row, column]
            magnitude = np.hypot(field[..., 0], field[..., 1])
            heat = ax.imshow(magnitude, vmin=0, vmax=vmax, cmap="magma")
            step = max(8, min(field.shape[:2]) // 14)
            y, x = np.mgrid[step // 2:field.shape[0]:step, step // 2:field.shape[1]:step]
            ax.quiver(x, y, field[y, x, 0], field[y, x, 1], angles="xy", scale_units="xy", scale=0.2,
                      width=0.0025, color="white", alpha=0.75)
            ax.set_title(f"{video_id} | {title} | t={time:.3f}s", fontsize=9)
            ax.set_axis_off()
    fig.subplots_adjust(top=0.91, right=0.9, hspace=0.18, wspace=0.04)
    color_axis = fig.add_axes([0.92, 0.16, 0.018, 0.62])
    fig.colorbar(heat, cax=color_axis, label="Flow magnitude (pixels / observed pair)")
    fig.suptitle("Prespecified old-clip diagnostics | one color scale; vectors magnified 5x", fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def _rate(row, metric, prefix):
    value = row.get(metric)
    if value is None:
        return "N/A"
    count = row.get(f"{prefix}_count", "?")
    total = row.get(f"{metric}_n", "?")
    text = f"{100 * value:.1f}% ({count}/{total})"
    low, high = row.get(f"{prefix}_ci_low"), row.get(f"{prefix}_ci_high")
    if low is not None and high is not None:
        text += f" [{100 * low:.1f}, {100 * high:.1f}]"
    return text


def _outcome(value):
    if value.get("estimate") is None:
        return "N/A"
    return f"{value['estimate']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]"


def _conclusion(metrics):
    primary = metrics["primary_comparison"]
    auc = primary["outcomes"]["delta_auroc"]
    if auc["estimate"] is None:
        return "새 확인 표본에서 주 비교에 필요한 양성·음성 쌍이 충분하지 않아 보정의 효과를 판단할 수 없다."
    if auc["ci_low"] > 0:
        text = "새 확인 표본에서 이번에 고정한 affine 보정은 같은 blob 제외 점수식의 AUROC를 높였다."
    elif auc["ci_high"] < 0:
        text = "새 확인 표본에서 이번에 고정한 affine 보정은 같은 blob 제외 점수식의 AUROC를 낮췄다."
    else:
        text = "새 확인 표본에서 이번에 고정한 affine 보정이 같은 blob 제외 점수식의 AUROC를 개선한다는 근거는 충분하지 않았다."
    text += f" 주 비교 ΔAUROC는 {_outcome(auc)}이며 대괄호는 95% paired bootstrap 구간이다."
    text += " 임계값 변경이나 좋은 모델만 선택하는 후속 재튜닝은 이 비교에 포함하지 않는다."
    return text


def make_ego_report(run_dir) -> Path:
    """Render compact recorded-data results and save comparison.csv without fitting."""
    run_dir = Path(run_dir).resolve()
    state_path = run_dir / "run_state.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("status") != "complete":
        raise ValueError("Cannot report an incomplete ego-motion run")
    clips = pd.read_csv(run_dir / "clip_scores.csv", dtype={"video_id": str})
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    if not isinstance(metrics, dict) or metrics.get("schema_version") != 1 or "primary_comparison" not in metrics:
        metrics = summarize_comparison(clips, config)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(metrics["results"]).to_csv(run_dir / "comparison.csv", index=False)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    dpi = int(config.get("report", {}).get("dpi", 130))
    names = ["comparison_metrics.png", "confirmation_confusion.png", "paired_scores.png", "known_negatives.png"]
    _plot_metrics(metrics["results"], figures / names[0], dpi)
    _plot_confusion(metrics["results"], figures / names[1], dpi)
    _plot_pairs(clips, figures / names[2], dpi)
    _plot_known_negatives(clips, figures / names[3], dpi)
    if _plot_motion_fields(run_dir, figures / "motion_fields.png", dpi):
        names.append("motion_fields.png")
    sections = [
        "# Fly-TTC 전역 흐름 보정 대조 실험",
        "이 점수는 초파리 LPLC2/Giant Fiber에서 영감을 받은 고정 팽창 검출기의 공학적 대조 실험이다. 전체 초파리 뇌나 생물 뉴런 회로를 재현한 것이 아니다.",
        "## 판단", _conclusion(metrics),
        "## 설계와 데이터",
        "주 비교는 `affine_residual_no_blob − raw_flow_no_blob`다. 두 대상의 blob 항을 함께 제외해 보정의 추가 영향을 분리했다. 원본 v0와 프레임 차분은 고정된 참고 대조다. 네 대상 모두 결과를 보고하며, 가장 좋은 것을 고르는 다중 비교 유의성 주장은 하지 않는다.",
        "`validation`은 원래 calibration ID이며 각 모델의 표준화와 임계값은 그중 같은 음성 10개만으로 정한다. `v0_original`은 이전 표준화·임계값을 그대로 사용한다. `exploratory`는 이미 분석된 기존 평가 80개로 이번 설계에 대해 독립 평가가 아니다. `confirmation`은 기존 100개와 ID가 겹치지 않는 추가 표본이다. 새 점수를 보기 전에 설정과 ID를 고정하며 새 표본에서 재보정하지 않는다. Clip 비중복은 주행·장소 비중복을 증명하지 않는다.",
    ]
    counts = clips.drop_duplicates("video_id").groupby(["split", "label"]).size()
    indexed = {(row["model"], row["split"]): row for row in metrics["results"]}
    old_difference = indexed[("frame_difference", "exploratory")]
    new_difference = indexed[("frame_difference", "confirmation")]
    if old_difference["auroc"] is not None and new_difference["auroc"] is not None:
        sections.insert(4, f"프레임 차분의 AUROC도 기존 탐색 표본에서는 {old_difference['auroc']:.3f}였지만 새 확인 표본에서는 {new_difference['auroc']:.3f}였다. 두 표본은 구성 차이가 있어 이 변화의 원인을 단정할 수 없지만, 기존 표본의 더 좋은 점수만으로 방법을 채택해서는 안 된다는 점을 보여 준다.")
    sections.append(_table(["역할", "양성 요청", "음성 요청"], [[split, int(counts.get((split, 1), 0)), int(counts.get((split, 0), 0))] for split in SPLITS]))
    titles = {"confirmation": "새 clip-disjoint 확인 결과", "exploratory": "기존 80개 탐색 재사용", "validation": "Calibration 기술 통계"}
    for split in ("confirmation", "exploratory", "validation"):
        rows = [row for row in metrics["results"] if row["split"] == split]
        sections.extend([
            "## " + titles[split],
            _table(["모델", "유효/요청 (양/음)", "AUROC", "AP", "FPR % [95%]", "TPR % [95%]", "alert hit"], [[
                row["model"], f"{row['n_scored']}/{row['n_requested']} ({row['n_positive']}/{row['n_negative']})",
                _number(row["auroc"], 3), _number(row["auprc"], 3), _rate(row, "fpr", "fpr"),
                _rate(row, "true_positive_rate", "tpr"),
                f"{_number(row.get('hit_alert_rate'), 3)} (n={row.get('hit_alert_n', 0)})",
            ] for row in rows]),
        ])
        if split == "confirmation":
            early = sorted({k for row in rows for k in row if k.startswith("hit_early_") and k.endswith("_rate")},
                           key=lambda key: int(key.removeprefix("hit_early_").removesuffix("ms_rate")))
            sections.append(_table(["모델", *[key.removesuffix("_rate") for key in early], "peak lead→alert 중앙값 s", "peak lead→event 중앙값 s"], [[
                row["model"], *[f"{_number(row.get(key), 3)} (n={row.get(key.removesuffix('_rate') + '_n', 0)})" for key in early],
                _number(row.get("lead_to_alert_median_s"), 3), _number(row.get("lead_to_event_median_s"), 3),
            ] for row in rows]))
        elif split == "validation":
            sections.append("선택된 calibration FPR/TPR에는 추론용 신뢰구간을 붙이지 않는다. 음성 10개에서 한 건은 10%p이므로 calibration FPR 10%는 실제 주행 FPR 10%를 보증하지 않는다.")
        else:
            sections.append("위 구간은 표본 변동의 기술적 요약이다. 실패 사례를 이미 보고 설계를 바꾼 적응적 재사용을 취소하거나 새 독립 평가로 바꾸지 않는다.")
    primary = metrics["primary_comparison"]
    sections.extend([
        "## 사전 지정한 보정 효과",
        f"Confirmation에서 양 모델이 모두 유효한 {primary['n_match']}개(양성 {primary['n_positive']}, 음성 {primary['n_negative']})를 clip ID로 짝지었다. 주 비교 제외 ID는 {primary['n_excluded']}개다. 동일한 클래스별 재표본 인덱스를 양 모델에 적용한 {primary['bootstrap_iterations']}회 bootstrap, seed {primary['bootstrap_seed']}의 95% percentile 구간이다. 모델·calibration 표본은 재적합하지 않으므로 구간은 고정 calibration에 조건부다.",
        _table(["차이 (보정 − raw)", "점추정 [95% 구간]", "분모"], [[name, _outcome(value), value["n"]] for name, value in primary["outcomes"].items()]),
        "ΔAUROC는 사전 지정한 주 지표다. ΔFPR/ΔTPR는 보조 결과이며 FPR은 낮을수록, TPR은 높을수록 좋다. 개별 AUROC 오차막대의 겹침만으로 paired 차이의 유의성을 판단하지 않는다.",
        f"![새 표본 AUROC와 FPR/TPR](figures/{names[0]})",
        f"![새 표본 confusion counts](figures/{names[1]})",
        "## 점수와 기존 실패 사례",
        "각 모델의 표준화·임계값이 달라 점수 절대 크기는 모델끼리 직접 비교하지 않는다. 점선은 각 축의 고정 임계값이다. 아래 기존 음성 3개는 사용자 검토에서 지정된 사후 사례이며 대표 표본이나 성공률의 분모가 아니다.",
        f"![동일 클립의 raw와 residual peak](figures/{names[2]})",
        f"![기존 음성 01538, 01510, 01250](figures/{names[3]})",
    ])
    delta_fpr = primary["outcomes"]["delta_fpr"]
    if delta_fpr["n"] and delta_fpr["estimate"] == delta_fpr["ci_low"] == delta_fpr["ci_high"] == 0:
        sections.insert(sections.index("## 점수와 기존 실패 사례"),
                        f"ΔFPR의 [0, 0] 구간은 이번 {delta_fpr['n']}개 음성에서 두 모델의 판정이 모두 같아 재표본에서도 차이가 없기 때문이다. 새로운 주행에서도 차이가 0이라는 보증은 아니다.")
    if "motion_fields.png" in names:
        sections.extend([
            "아래 진단은 새 확인 표본의 좋은 결과를 골라 만든 것이 아니라 사전에 지정한 기존 클립의 흐름장이다. 세 열과 모든 시점이 같은 색상 범위와 화살표 배율을 사용한다. 원본 영상은 표시하지 않는다. 여기서는 clipping 이전의 raw = background + residual 분해를 표시하며, 실제 점수 계산은 감산 후 residual의 99퍼센타일 clipping을 적용한다.",
            "![동일 스케일의 raw, 전역, residual 흐름장](figures/motion_fields.png)",
        ])
    quality_path = run_dir / "fit_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path, dtype={"video_id": str})
        expected = {"split", "fit_valid_fraction", "inlier_fraction_mean", "fit_error_px_median"}
        if expected.issubset(quality):
            sections.extend(["## 보정 적합 진단", _table(
                ["분할", "클립 수", "클립별 valid 비율 평균", "inlier 평균", "클립별 오차 중앙값의 중앙값 px"],
                [[split, len(group), _number(group.fit_valid_fraction.mean(), 3),
                  _number(group.inlier_fraction_mean.mean(), 3), _number(group.fit_error_px_median.median(), 3)]
                 for split, group in quality.groupby("split", sort=True)],
            ), "비율은 양성의 사건 이전 프레임과 음성 분석 구간에서 구한 클립별 값의 비가중 평균이다. 적합 성공률은 배경 자기운동 정답률이 아니다. [클립별 적합 진단](fit_quality.csv)에 실패·inlier·잔차 통계를 보존했다."])
    mechanism_path = run_dir / "mechanism_audit.json"
    if mechanism_path.exists():
        mechanism = json.loads(mechanism_path.read_text())
        fp = mechanism.get("confirmation_false_positive_summary", {})
        if fp:
            sections.extend([
                "## 결과 확인 뒤 보정 적용 감사",
                f"새 확인 평가의 위양성 {fp['n_fps']}개 중 {fp['n_peak_fallback']}개는 보정 모델의 최고점에서 affine 추정이 실패해 원래 흐름을 사용했다. {fp['n_all_last5_fallback']}개는 최고점까지 최근 최대 5개 표본 모두 같은 실패 처리였다. 이는 이번 추정기가 문제 장면에서 충분히 적용되지 않았다는 진단이며, 모든 음성의 실패 비율이나 인과효과 추정은 아니다.",
                "원래 흐름을 사용하는 프레임도 모델별 음성 표준화 통계와 EMA 이력이 달라 최종 점수가 다를 수 있다. 더 큰 보정 모델 점수가 물리적 팽창의 증가를 뜻하지는 않는다. 결과를 본 뒤 적합 허용 기준을 조정하지 않았다. [수치 감사](mechanism_audit.json)",
            ])
            if (run_dir / "AUDIT.md").exists():
                sections.append("[메커니즘 감사 해설](AUDIT.md)")
    sections.extend([
        "## 해석의 한계와 다음 판단",
        "전역 affine은 영상의 주된 2D 움직임에 대한 근사다. 물리적인 자차 운동이나 지면/무한원 팽창점을 복원한 것이 아니다. 깊이가 다른 물체의 parallax가 남으며, 크게 다가오는 물체를 전역 확대 성분으로 제거할 수도 있다. Homography도 평면 또는 순수 회전 조건이 필요하다. [OpenCV 설명](https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html)",
        "이 결과는 이번에 고정한 affine 근사와 점수식의 한계를 보여 주며, 모든 자기운동 보정 방법이 불가능함을 입증하지 않는다. v1의 4방향 회로나 앞차 검출기가 해결책이라고도 결론 내리지 않는다. 이번 근사를 반복 튜닝하기보다는 국소 객체의 팽창을 보존하면서 카메라 흐름을 분리할 수 있는지를 별도 프로토콜로 검증하는 것이 다음 기술적 질문이다. 이 점수는 TTC 초 단위가 아니며 희귀사고 빈도의 실서비스 정밀도를 추정하지 않는다.",
        "BADAS-Open은 공개 Nexar 1,500개로 학습했다고 명시하므로 현재 train 표본에서 독립 성능 상한으로 비교할 수 없다. 추가 train 100개도 BADAS의 학습중복 문제를 해결하지 않는다. 이번에는 실행하지 않았다. [공식 모델 카드](https://huggingface.co/nexar-ai/BADAS-Open), [논문](https://arxiv.org/html/2510.14876v1)",
        "모든 사건 메트릭은 사건 이전 점수만 사용한다. Early hit는 해당 마감시각까지 한 번이라도 임계값을 넘었는지이며, lead는 첫 경보가 아닌 최대점의 시각이다. 양성·음성의 원래 분석 구간 길이는 같지 않을 수 있다. AP는 이 균형 표본의 양성 비율에 의존한다.",
        "## 재현과 감사 자료",
        "```bash\npython scripts/analyze_subgroups.py --run outputs/v0 --out outputs/v0_review\npython scripts/run_ego_ablation.py --phase prepare\npython scripts/run_ego_ablation.py --phase download\npython scripts/run_ego_ablation.py --phase run\npython scripts/run_ego_ablation.py --phase report\npython scripts/export_review.py\n```",
        "[비교 CSV](comparison.csv), [클립별 원자료](clip_scores.csv), [전체 메트릭·paired 제외 ID](metrics.json), [설정](config.yaml)을 보존했다. [기존 v0·Highway + Normal 감사 보고서](../v0_review/REPORT.md)와 함께 읽는다.",
    ])
    if (run_dir / "audit.json").exists():
        audit = json.loads((run_dir / "audit.json").read_text())
        audit_labels = {
            "original_clips_reproduced": "원본 v0 재현 클립", "baseline_unchanged": "이전 결과 파일 불변",
            "confirmation_requested": "새 확인 요청 클립", "confirmation_decoded": "새 확인 디코딩 클립",
            "confirmation_sha256_verified": "새 확인 SHA256 일치", "clip_id_overlap": "기존·새 ID 중복",
            "scoring_code_unchanged_since_freeze": "동결 뒤 점수 코드 불변",
        }
        sections.append(_table(["감사 항목", "기록된 값"], [[label, audit[key]] for key, label in audit_labels.items() if key in audit]))
    if (run_dir / "thresholds.yaml").exists():
        thresholds = yaml.safe_load((run_dir / "thresholds.yaml").read_text()) or {}
        sections.append(_table(["모델", "동결 theta", "Calibration FPR"], [[
            model, _number(thresholds.get(model, {}).get("theta"), 6), _number(thresholds.get(model, {}).get("val_fpr"), 3),
        ] for model in MODELS]))
    for name, label in (("protocol.json", "고정 프로토콜·ID·소스 해시"), ("thresholds.yaml", "동결 모델별 표준화·임계값"),
                        ("validation_checks.json", "실행 검증 기록"), ("audit.json", "감사 증거"),
                        ("mechanism_audit.json", "보정 적용·피크 실패 처리 메커니즘 감사"), ("environment.json", "실행 환경")):
        if (run_dir / name).exists():
            sections.append(f"[{label}]({name})")
    source = "\n\n".join(sections) + "\n"
    report_path = run_dir / "REPORT.md"
    report_path.write_text(source)
    rendered = _html_report(source, names).replace("<title>Fly-TTC v0 Report</title>", "<title>Fly-TTC Ego-Motion Comparison</title>")
    (run_dir / "REPORT.html").write_text(rendered)
    return report_path
