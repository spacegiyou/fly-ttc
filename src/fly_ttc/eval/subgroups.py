"""Read-only, post-hoc subgroup audit with the original split and threshold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from fly_ttc.eval.metrics import bool_values, tag_values, valid_clips


def wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson score interval, including zero/all successes."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Expected 0 <= successes <= total")
    if not total:
        return None, None
    z = NormalDist().inv_cdf(0.975)
    p, z2 = successes / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * np.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def bootstrap_auroc(
    labels, scores, seed: int = 0, iterations: int = 2000,
) -> tuple[float | None, float | None]:
    """Percentile 95% CI, resampling clips separately within each label.

    This conditions on observed class counts and a fixed detector/threshold.
    It does not cover calibration uncertainty or post-hoc subgroup selection.
    """
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("labels and finite scores must have matching shapes")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None, None
    # Pairwise ranking is exactly AUROC, including half credit for ties.
    pairs = (pos[:, None] > neg[None, :]).astype(float)
    pairs += 0.5 * (pos[:, None] == neg[None, :])
    rng, values = np.random.default_rng(seed), []
    for start in range(0, iterations, 128):
        size = min(128, iterations - start)
        pos_idx = rng.integers(len(pos), size=(size, len(pos), 1))
        neg_idx = rng.integers(len(neg), size=(size, 1, len(neg)))
        values.extend(pairs[pos_idx, neg_idx].mean(axis=(1, 2)).tolist())
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _summarize(group: pd.DataFrame, theta: float, seed: int, iterations: int) -> dict:
    valid = valid_clips(group)
    pos, pred = valid.label.eq(1), valid.s_peak.gt(theta)
    tp, fp = int((pos & pred).sum()), int((~pos & pred).sum())
    tn, fn = int((~pos & ~pred).sum()), int((pos & ~pred).sum())
    n_pos, n_neg = tp + fn, fp + tn
    fpr_ci, tpr_ci = wilson_interval(fp, n_neg), wilson_interval(tp, n_pos)
    auc_ci = bootstrap_auroc(valid.label, valid.s_peak, seed, iterations)
    result = {
        "n_requested": len(group), "n_scored": len(valid), "n_excluded": len(group) - len(valid),
        "n_positive": n_pos, "n_negative": n_neg,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "theta": theta,
        "auroc": float(roc_auc_score(valid.label, valid.s_peak)) if n_pos and n_neg else None,
        "average_precision": float(average_precision_score(valid.label, valid.s_peak)) if n_pos and n_neg else None,
        "positive_fraction": n_pos / len(valid) if len(valid) else None,
        "auroc_ci_low": auc_ci[0], "auroc_ci_high": auc_ci[1],
        "fpr": fp / n_neg if n_neg else None,
        "fpr_ci_low": fpr_ci[0], "fpr_ci_high": fpr_ci[1],
        "tpr": tp / n_pos if n_pos else None,
        "tpr_ci_low": tpr_ci[0], "tpr_ci_high": tpr_ci[1],
    }
    for column in ["hit_alert", *sorted(c for c in group if c.startswith("hit_early_"))]:
        values = bool_values(valid.loc[pos, column]).dropna() if column in valid else pd.Series(dtype=float)
        result[f"{column}_n"] = len(values)
        result[f"{column}_count"] = int(values.sum())
        result[f"{column}_rate"] = float(values.mean()) if len(values) else None
    return result


def _groups(clips: pd.DataFrame):
    yield "all", "All clips", clips
    for column in ["scene", "light"]:
        for value, subset in clips.groupby(column, dropna=False, sort=True):
            yield column, f"{column}={value}", subset
    for (scene, light), subset in clips.groupby(["scene", "light"], dropna=False, sort=True):
        yield "scene_light", f"scene={scene} | light={light}", subset


def _count_outcomes(clips: pd.DataFrame, theta: float) -> dict:
    pos, pred = clips.label.eq(1), clips.s_peak.gt(theta)
    return {
        "n": len(clips), "n_positive": int(pos.sum()), "n_negative": int((~pos).sum()),
        "tp": int((pos & pred).sum()), "fp": int((~pos & pred).sum()),
        "tn": int((~pos & ~pred).sum()), "fn": int((pos & ~pred).sum()),
    }


def _tag_audit(clips: pd.DataFrame, theta: float) -> dict:
    tags = clips.get("failure_tags", pd.Series("", index=clips.index)).map(tag_values)
    risk_metadata = clips.light.str.casefold().isin(["dark", "twilight"]) | clips.weather.str.casefold().eq("rain")
    tagged = tags.map(lambda value: "low_visibility" in value)
    expected_tag = risk_metadata & clips.s_peak.le(theta)
    return {
        "tag_counts_by_outcome": {
            tag: _count_outcomes(clips[tags.map(lambda value: tag in value)], theta)
            for tag in sorted({tag for values in tags for tag in values})
        },
        "adverse_light_or_rain_metadata": _count_outcomes(clips[risk_metadata], theta),
        "other_light_weather_metadata": _count_outcomes(clips[~risk_metadata], theta),
        "low_visibility_rule_mismatch_ids": clips.loc[tagged.ne(expected_tag), "video_id"].tolist(),
        "low_visibility_definition": "(Dark or Twilight or Rain) AND no score > fixed theta; applies to both labels",
        "caveat": "A descriptive metadata-and-prediction heuristic, not verified visibility or an independent failure cause.",
    }


def _id(value) -> str:
    text = str(value)
    return text.zfill(5) if text.isdigit() else text


def _json_records(frame: pd.DataFrame) -> list[dict]:
    # Pandas maps missing scalar values to JSON null and converts numpy scalars.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _plot(table: pd.DataFrame, path: Path) -> None:
    names = ["All clips", "scene=Highway", "scene=Highway | light=Normal", "scene=Urban", "scene=Sub-urban"]
    selected = table[(table.split == "evaluation") & table.subgroup.isin(names)].copy()
    selected["order"] = selected.subgroup.map({name: i for i, name in enumerate(names)})
    selected = selected.sort_values("order")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    y = np.arange(len(selected))
    for ax, metric, title in zip(axes, ["auroc", "tpr", "fpr"], ["AUROC (bootstrap 95% CI)", "TPR (Wilson 95% CI)", "FPR (Wilson 95% CI)"]):
        for position, (_, row) in zip(y, selected.iterrows()):
            if pd.notna(row[metric]):
                ax.plot([row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]], [position] * 2, color="#177e89")
                ax.plot(row[metric], position, "o", color="#123c4a")
        ax.axvline(0.5 if metric == "auroc" else 0.1 if metric == "fpr" else 0, color="grey", linestyle=":", linewidth=1)
        ax.set(xlim=(-0.03, 1.03), title=title, xlabel="Rate")
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, [f"{row.subgroup.replace('scene=', '').replace('light=', '')}\n(n+={row.n_positive}, n-={row.n_negative})" for _, row in selected.iterrows()])
    axes[0].invert_yaxis()
    fig.suptitle("v0: fixed original threshold / post-hoc exploratory subgroups", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _interval(row, name: str, percent: bool = False) -> str:
    if pd.isna(row[name]):
        return "NA"
    scale, suffix, precision = (100, "%", 1) if percent else (1, "", 3)
    return f"{row[name] * scale:.{precision}f}{suffix} [{row[name + '_ci_low'] * scale:.{precision}f}, {row[name + '_ci_high'] * scale:.{precision}f}]"


def _report(result: dict, table: pd.DataFrame) -> str:
    evaluation = table[table.split.eq("evaluation")]
    visible = evaluation[evaluation.group_type.isin(["all", "scene"]) | evaluation.subgroup.eq("scene=Highway | light=Normal")]
    lines = [
        "# v0 고정 결과 재감사 및 하위집단 탐색",
        "",
        f"원본 `clip_scores.csv`와 `threshold.yaml`만 읽었다. 원래 분할과 theta={result['theta']:.12f}를 고정했고 재보정·재실행·평가셋 튜닝을 하지 않았다. 원본 파일 SHA256은 `analysis.json`에 기록했다.",
        "",
        "## 독립 평가 분할 재집계",
        "",
        "아래 대괄호는 95% 신뢰구간이다. AUROC는 라벨별 클립 복원추출 bootstrap, TPR/FPR은 Wilson 구간이다. TP/FN은 사건 이전 클립 최대값이 기존 임계값을 넘었는지로 판정했다.",
        "",
        "| 하위집단 | 양성/음성 | TP/FP/TN/FN | AUROC [95% CI] | AP | TPR [95% CI] | FPR [95% CI] | alert hit |",
        "|---|---:|---:|---|---:|---|---|---:|",
    ]
    for _, row in visible.iterrows():
        ap = f"{row.average_precision:.3f}" if pd.notna(row.average_precision) else "NA"
        name = row.subgroup.replace(" | ", " + ")
        lines.append(f"| {name} | {row.n_positive}/{row.n_negative} | {row.tp}/{row.fp}/{row.tn}/{row.fn} | {_interval(row, 'auroc')} | {ap} | {_interval(row, 'tpr', True)} | {_interval(row, 'fpr', True)} | {row.hit_alert_count}/{row.hit_alert_n} |")
    highway_normal = evaluation[evaluation.subgroup.eq("scene=Highway | light=Normal")]
    if len(highway_normal):
        row = highway_normal.iloc[0]
        lines += ["", f"요청된 Highway + Normal light는 {row.n_scored}개(양성 {row.n_positive}, 음성 {row.n_negative})뿐이다. AUROC {_interval(row, 'auroc')}, FPR {_interval(row, 'fpr', True)}이므로 이 조건을 운용 대상으로 확정할 근거가 부족하다."]
    audit = result["tag_audit"]["evaluation"]
    low = audit["tag_counts_by_outcome"].get("low_visibility", dict(n=0, fn=0, tn=0))
    metadata = audit["adverse_light_or_rain_metadata"]
    lines += [
        "", "![고정 임계값 하위집단 구간](subgroup_intervals.png)", "", "## 실패 태그 해석 교정", "",
        f"`low_visibility` {low['n']}개는 양성 미검출 {low['fn']}개와 올바르게 거부한 음성 {low['tn']}개를 포함한다. 이 태그는 `(Dark 또는 Twilight 또는 Rain) AND 임계값 미통과` 규칙이며 양성과 음성 모두에 적용된다. 따라서 전체 태그 비율을 실제 시야 불량의 비율이나 미검출 원인 비율로 해석하면 안 된다.",
        f"점수와 독립적인 Dark/Twilight/Rain 메타데이터 조건은 {metadata['n']}개(양성 {metadata['n_positive']}, 음성 {metadata['n_negative']})이며 TP/FP/TN/FN={metadata['tp']}/{metadata['fp']}/{metadata['tn']}/{metadata['fn']}이다. 이것도 실제 영상 가시성의 수동 검증 결과는 아니다.",
        f"원본 low_visibility 규칙 불일치 ID: {audit['low_visibility_rule_mismatch_ids'] or '없음'}. `ego_shake`도 원시 흐름 요약값 기반 휴리스틱이므로 카메라 흔들림의 수동 확정 라벨은 아니다.",
        "", "## 인용 사례 재확인", "",
        "| video_id | label | split | scene | light | s_peak | predicted_positive | hit_alert |",
        "|---|---:|---|---|---|---:|---|---|",
    ]
    for row in result["cited_examples"]:
        lines.append(f"| {row['video_id']} | {row['label']} | {row['split']} | {row['scene']} | {row['light']} | {row['s_peak']:.6f} | {row['predicted_positive']} | {row['hit_alert']} |")
    lines += [
        "", "## 해석 범위와 다음 실험", "",
        "- 모든 하위집단은 v0 결과를 본 뒤 선택한 사후 탐색이다. 다중비교 보정이나 독립 확인을 하지 않았으므로 가장 높은 행을 골라 성능 개선으로 주장할 수 없다. 점수 순위가 완전히 분리된 작은 표본에서는 bootstrap AUROC 구간이 [1, 1]이 될 수 있지만, 이는 모집단에서 완벽함을 뜻하지 않는다. 고정 임계값의 TP/FP도 함께 봐야 한다.",
        "- `subgroups.csv`에는 evaluation, validation, all을 분리하고 scene, light, 관측된 모든 scene×light 조합을 기록했다. validation과 all은 보정 표본을 포함하는 참고 집계다. 한 라벨만 있는 행의 AUROC/AP는 NA다.",
        f"- Bootstrap은 seed={result['seed']}, 반복 {result['bootstrap_iterations']}회, 라벨별 원래 표본 수를 유지했다. 고정 검출기 및 관측 클립 집합의 불확실성만 다루며 임계값 보정, 같은 출처 영상의 상관, 사후 선택에 따른 불확실성을 포함하지 않는다. 작은 하위집단의 bootstrap 구간은 불안정하거나 퇴화할 수 있다.",
        "- AP의 무작위 참고선은 각 하위집단의 양성 비율이다. 서로 다른 양성 비율을 가진 행의 AP를 바로 비교하면 안 된다. `positive_fraction`에 분모를 기록했다.",
        "- v1 확대 전에 자기운동 보정의 제한된 가설을 합성 자극으로 검증하는 순서가 합리적이다. 합성 평행 이동 억제와 국소 팽창 보존을 먼저 잠그고 기존 음성 validation만으로 새 점수의 통계와 임계값을 보정한다. 이미 확인한 80개 결과를 기준으로 설정을 반복 선택하면 안 된다.",
        "- 같은 80개를 새 모델로 다시 평가하면 비교 실험은 가능하지만, 이번 결과를 보고 가설을 선택했으므로 새로운 독립 최종 검증은 아니다. 다음 확증에는 별도로 고정한 미사용 데이터가 필요하다.",
        "", "## 재현", "", "```bash",
        f"python scripts/analyze_subgroups.py --run outputs/v0 --out outputs/v0_review --seed {result['seed']} --bootstrap-iterations {result['bootstrap_iterations']}",
        "```", "",
    ]
    return "\n".join(lines)


def analyze_subgroups(run_dir, out_dir, seed: int = 0, bootstrap_iterations: int = 2000) -> dict:
    """Audit frozen outputs; fail rather than silently resplit/recalibrate."""
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir == run_dir or out_dir.is_relative_to(run_dir):
        raise ValueError("Audit output must be outside the original run directory")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    paths = [run_dir / "clip_scores.csv", run_dir / "threshold.yaml"]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    clips = pd.read_csv(paths[0], dtype={"video_id": str})
    required = {"video_id", "label", "s_peak", "theta_used", "split", "scene", "light", "weather"}
    if missing := required - set(clips):
        raise ValueError(f"Missing subgroup audit columns: {sorted(missing)}")
    clips.video_id = clips.video_id.map(_id)
    if clips.video_id.duplicated().any():
        raise ValueError("Duplicate video IDs in source run")
    if not clips.split.isin(["evaluation", "validation"]).all():
        raise ValueError("Expected the original evaluation/validation split")
    threshold = yaml.safe_load(paths[1].read_text())
    theta = float(threshold["theta"])
    if not np.isfinite(theta):
        raise ValueError("Original theta must be finite")
    valid = valid_clips(clips)
    if not np.isclose(valid.theta_used.to_numpy(dtype=float), theta, rtol=0, atol=1e-12).all():
        raise ValueError("Source scores do not share the saved fixed theta")
    if "predicted_positive" in valid:
        saved = bool_values(valid.predicted_positive)
        if saved.isna().any() or saved.ne(valid.s_peak.gt(theta).astype(float)).any():
            raise ValueError("Source predictions disagree with strict S > theta")
    for split in ["evaluation", "validation"]:
        if f"{split}_ids" not in threshold:
            raise ValueError(f"Original {split} IDs missing from threshold.yaml")
        expected = {_id(value) for value in threshold[f"{split}_ids"]}
        observed = set(clips.loc[clips.split.eq(split), "video_id"])
        if observed != expected:
            raise ValueError(f"Source {split} IDs disagree with threshold.yaml")
    records, audits = [], {}
    for split, subset in [("evaluation", clips[clips.split.eq("evaluation")]), ("validation", clips[clips.split.eq("validation")]), ("all", clips)]:
        audits[split] = _tag_audit(valid_clips(subset), theta)
        for group_type, name, group in _groups(subset):
            group_seed = int.from_bytes(hashlib.sha256(f"{seed}:{split}:{name}".encode()).digest()[:8], "little")
            records.append({"split": split, "group_type": group_type, "subgroup": name, **_summarize(group, theta, group_seed, bootstrap_iterations)})
    table = pd.DataFrame(records)
    example_ids = ["01538", "01510", "01250", "00300", "00510", "00773", "00877", "01024"]
    examples = clips[clips.video_id.isin(example_ids)].copy()
    examples["example_order"] = examples.video_id.map({value: i for i, value in enumerate(example_ids)})
    examples = examples.sort_values("example_order").drop(columns="example_order")
    result = {
        "analysis": "posthoc_exploratory_fixed_v0", "theta": theta, "seed": seed,
        "bootstrap_iterations": bootstrap_iterations, "source_run": str(run_dir),
        "source_sha256": hashes, "source_unchanged": True,
        "decision_rule": "s_peak > original theta; no recalibration or model selection",
        "auroc_ci": "95% percentile bootstrap, resample clips within labels",
        "rate_ci": "95% Wilson score interval", "multiple_comparison_adjustment": False,
        "subgroups": _json_records(table), "tag_audit": audits,
        "cited_examples": _json_records(examples),
        "cited_example_ids_absent": sorted(set(example_ids) - set(examples.video_id)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "subgroups.csv", index=False)
    examples.to_csv(out_dir / "cited_examples.csv", index=False)
    _plot(table, out_dir / "subgroup_intervals.png")
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
            raise RuntimeError(f"Source changed during audit: {path}")
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (out_dir / "REPORT.md").write_text(_report(result, table))
    return result
