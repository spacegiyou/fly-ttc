"""Korean Markdown/HTML reports using recorded scores only, without raw frames."""

from __future__ import annotations

import html
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown as markdown_library
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

from fly_ttc.eval.failure_tags import FAILURE_TAGS
from fly_ttc.eval.metrics import bool_values, stratified_metrics, summarize_metrics, tag_values, valid_clips
from .flow_overlay import plot_diagnostics
from .traces import plot_traces, shared_limits, unavailable_figure


LIMITATIONS = [
    "이 구현은 화면 중심의 팽창 대리 지표다. 사고 유형만으로 검출 가능성을 단정할 수 없으며, 대상 물체의 가시적 팽창 단서와 위치를 확인해야 한다. 실제 초파리 전체의 능력을 평가한 결과가 아니다.",
    "Nexar 양성은 collision과 near-miss를 구분하지 않는다.",
    "광학 흐름은 야간·비·와이퍼·큰 카메라 흔들림에 깨진다.",
    "점수는 TTC의 물리적 초가 아니다. 상대적 looming proxy다.",
    "임계값은 이 서브셋 음성 FPR에 맞춰져 있어 다른 도메인에 그대로 못 옮긴다.",
    "이 파이프라인은 자율주행 스택을 대체하지 않는다. 희귀 이벤트 마이닝/라벨 제안용이다.",
]


def _number(value, digits=4):
    if value is None or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "N/A"
    return str(int(value)) if isinstance(value, (int, np.integer)) else f"{value:.{digits}f}"


def _table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(clean, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(clean, row)) + " |" for row in rows),
    ])


def _validation_summary(checks):
    rows = []
    tests = checks.get("pytest", {})
    if tests:
        rows.append(["pytest", f"통과 {tests.get('passed', 'N/A')}개 / 실패 {tests.get('failed', 'N/A')}개"])
    if "sha256_match_count" in checks:
        rows.append(["영상 SHA256 검증", f"{checks['sha256_match_count']}/{checks.get('real_video_count', 'N/A')}개 일치"])
    if "positive_pre_event_peak_hit_and_clock_checks" in checks:
        rows.append(["양성 사건 이전 peak·hit·시간축", f"{checks['positive_pre_event_peak_hit_and_clock_checks']}개 점검"])
    if "audit_errors" in checks:
        rows.append(["검증 오류", f"{len(checks['audit_errors'])}개"])
    recalibrated = checks.get("validation_threshold_recomputed", {})
    if recalibrated:
        rows.append(["Validation 임계값 재계산", f"theta={_number(recalibrated.get('theta'), 6)}, FPR={_number(recalibrated.get('val_fpr'))}"])
    if "normalization_membership" in checks:
        rows.append(["표준화 입력 확인", str(checks["normalization_membership"])])
    result = "기록된 검증 결과:\n\n" + _table(["검증 항목", "결과"], rows)
    result += "\n\n[전체 검증 기록](validation_checks.json)을 함께 보존했다."
    stimuli = checks.get("synthetic", {}).get("stimuli", {})
    if stimuli:
        result += "\n\n합성 자극 결과 (단위 통계로 계산한 테스트 점수이며 실제 영상 점수·임계값과 직접 비교하지 않는다):\n\n"
        result += _table(["Stimulus", "Frames", "Late mean S", "Peak S"], [
            [name, values.get("n_frames", "N/A"), _number(values.get("late_mean_S"), 6), _number(values.get("peak_S"), 6)]
            for name, values in stimuli.items()
        ])
    return result


def _load_frames(run_dir, clips):
    frames = {}
    for video_id in clips.video_id.astype(str):
        # IDs originate from the validated manifest; reject unsafe relative paths.
        if Path(video_id).name != video_id or video_id in {".", ".."}:
            raise ValueError(f"Unsafe video_id: {video_id!r}")
        for directory in ("frames", "_report_frames"):
            path = run_dir / directory / f"{video_id}.parquet"
            if path.exists():
                frames[video_id] = pd.read_parquet(path)
                break
    return frames


def _plot_lead(clips, path, dpi):
    values = pd.to_numeric(clips.get("lead_to_alert_s", pd.Series(dtype=float)), errors="coerce").dropna()
    if not len(values):
        unavailable_figure(path, "Held-out positive peak lead to alert", "Unavailable: no valid held-out positive alert leads.", dpi)
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=min(15, max(3, int(np.ceil(np.sqrt(len(values)))))), color="#247aa6", edgecolor="white")
    ax.axvline(0, color="#cc522f", linestyle="--")
    ax.set(xlabel="Alert time minus pre-event peak time (s)", ylabel="Positive clips",
           title=f"Held-out peak lead to alert | n={len(values)}, median={values.median():.3f}s")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_curves(clips, path, dpi):
    if clips.empty or clips.label.nunique() < 2:
        unavailable_figure(path, "Held-out ROC / precision-recall",
                           f"Unavailable: both labels are required; scored held-out clips={len(clips)}.", dpi)
        return
    fpr, tpr, _ = roc_curve(clips.label, clips.s_peak)
    precision, recall, _ = precision_recall_curve(clips.label, clips.s_peak)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#247aa6")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--")
    axes[0].set(title="Held-out ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].step(recall, precision, where="post", color="#247aa6")
    axes[1].axhline(float(clips.label.mean()), color="#999999", linestyle="--", label="Class prevalence")
    axes[1].set(title="Held-out precision-recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_failures(metrics, path, dpi):
    n = metrics.get("n_scored", 0)
    if not n:
        unavailable_figure(path, "Held-out heuristic tag rates", "Unavailable: no scored held-out clips.", dpi)
        return
    counts = metrics.get("failure_tag_counts", {})
    fig, ax = plt.subplots(figsize=(9, 4))
    values = [counts.get(tag, 0) / n for tag in FAILURE_TAGS]
    bars = ax.barh(FAILURE_TAGS, values, color=["#c06a49"] * 4 + ["#35866b"])
    for bar, tag in zip(bars, FAILURE_TAGS):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{counts.get(tag, 0)}/{n}", va="center", fontsize=9)
    ax.set_xlim(0, max(1, max(values) + 0.15))
    ax.set(xlabel="Fraction of all scored held-out clips", title="Descriptive heuristics (overlap allowed; not collision labels)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


class _SafeReportHTML(HTMLParser):
    """Allow formatting but no executable markup or remote image requests."""

    tags = {"h1", "h2", "h3", "p", "em", "strong", "ul", "ol", "li", "blockquote",
            "pre", "code", "table", "thead", "tbody", "tr", "th", "td", "hr", "br", "img", "a"}
    void = {"hr", "br", "img"}

    def __init__(self, figure_names):
        super().__init__(convert_charrefs=True)
        self.output = []
        self.figure_paths = {f"figures/{name}" for name in figure_names}

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            return
        allowed = []
        for key, value in attrs:
            if value is None:
                continue
            if tag == "a" and key == "href":
                parsed = urlsplit(value)
                if parsed.scheme in {"https", "http"} or (not parsed.scheme and not parsed.netloc and not value.startswith("//")):
                    allowed.append((key, value))
            elif tag == "img" and key == "src" and value in self.figure_paths:
                allowed.append((key, value))
            elif tag in {"img", "a"} and key in {"alt", "title"}:
                allowed.append((key, value))
        if tag == "img" and not any(key == "src" for key, _ in allowed):
            return
        attributes = "".join(f' {key}="{html.escape(value, quote=True)}"' for key, value in allowed)
        self.output.append(f"<{tag}{attributes}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in self.tags and tag not in self.void:
            self.output.append(f"</{tag}>")

    def handle_data(self, data):
        self.output.append(html.escape(data, quote=False))


def _html_report(markdown, figure_names):
    # Escape raw HTML before Markdown conversion, then restrict emitted tags and
    # link protocols: config/manifest strings must never become active scripts.
    rendered = markdown_library.markdown(html.escape(markdown, quote=False), extensions=["tables", "fenced_code"])
    sanitizer = _SafeReportHTML(figure_names)
    sanitizer.feed(rendered)
    sanitizer.close()
    return """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fly-TTC v0 Report</title><style>
body{font-family:system-ui,sans-serif;line-height:1.65;max-width:1120px;margin:40px auto;padding:0 24px;color:#203047;background:#f7f9fc}h1,h2{color:#163951}h2{border-bottom:1px solid #cad5de;padding-top:16px}p{overflow-wrap:anywhere}pre{background:#e9eef3;padding:18px;overflow:auto}img{display:block;width:100%;height:auto;margin:24px 0;background:white;border-radius:8px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{border-bottom:1px solid #dce3e9;padding:7px;text-align:left}th{background:#e9eef3}code{font-family:ui-monospace,monospace}
</style><body>""" + "".join(sanitizer.output) + "</body></html>"


def make_report(run_dir) -> Path:
    """Create six figure types, REPORT.md/HTML and strata CSVs from one run.

    ``_report_frames`` holds temporary negative traces. It is removed only after
    all report artifacts succeed. Existing negative panel/scale is preserved on
    later report runs after this cache has been deleted.
    """
    run_dir = Path(run_dir).resolve()
    scores_path = run_dir / "clip_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}; run v0 first")
    clips = pd.read_csv(scores_path, dtype={"video_id": str})
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    threshold_path = run_dir / "threshold.yaml"
    threshold = yaml.safe_load(threshold_path.read_text()) if threshold_path.exists() else {}
    config, threshold = config or {}, threshold or {}
    theta = threshold.get("theta")
    if theta is not None:
        theta = float(theta)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else summarize_metrics(clips)
    if not {"all", "evaluation", "validation"}.issubset(metrics):
        metrics = summarize_metrics(clips)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cfg = config.get("report", {})
    n_traces, dpi = int(cfg.get("n_traces", 12)), int(cfg.get("dpi", 130))
    scored = valid_clips(clips).sort_values("video_id")
    frames = _load_frames(run_dir, scored)
    selection_path = run_dir / "report_selection.json"
    previous = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    limits = previous.get("y_limits") or shared_limits(frames, theta)
    selection = {"y_limits": limits}
    for label, filename, title in [(1, "positive_traces.png", "Positive"), (0, "negative_traces.png", "Negative")]:
        path = figures / filename
        rows = scored[scored.label.eq(label)]
        available = sum(str(video_id) in frames for video_id in rows.video_id)
        previous_ids = previous.get(title.lower(), [])
        if path.exists() and previous_ids and available < len(previous_ids):
            selection[title.lower()] = previous_ids
        else:
            selection[title.lower()] = plot_traces(rows, frames, path, theta, limits, title, n_traces, dpi)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    split = scored.get("split", pd.Series("unspecified", index=scored.index))
    evaluation = scored[split.isin(["evaluation", "heldout", "train"])]
    eval_pos = evaluation[evaluation.label.eq(1)]
    _plot_lead(eval_pos, figures / "lead_to_alert.png", dpi)
    _plot_curves(evaluation, figures / "roc_pr.png", dpi)
    _plot_failures(metrics["evaluation"], figures / "failure_modes.png", dpi)
    diagnostic_path, diagnostic_id = None, None
    positives = scored[scored.label.eq(1)].copy()
    if "failure_tags" in positives:
        positives["_good"] = positives.failure_tags.map(lambda value: "good_loom" in tag_values(value))
        positives = positives.sort_values("_good", ascending=False, kind="stable")
    for _, row in positives.iterrows():
        path = run_dir / "diagnostics" / f"{row.video_id}.npz"
        # A successful diagnostic needs an actual pre-event threshold crossing.
        if path.exists() and theta is not None and float(row.s_peak) > theta:
            diagnostic_path, diagnostic_id = path, str(row.video_id)
            break
    diagnostic_ok = plot_diagnostics(diagnostic_path, figures / "loom_diagnostic.png", diagnostic_id,
                                     dpi, cfg.get("privacy_blur_ksize", 61))
    strata = {}
    for column in ("scene", "light"):
        strata[column] = stratified_metrics(clips, column)
        strata[column].to_csv(run_dir / f"metrics_by_{column}.csv", index=False)
    summary = metrics["evaluation"]
    event_hits = int(bool_values(eval_pos.get("hit_event", pd.Series(dtype=float))).sum())
    metric_rows = []
    for key, title in [("evaluation", "Evaluation (headline)"), ("validation", "Validation (calibration)"), ("all", "All (descriptive)")]:
        values = metrics[key]
        metric_rows.append([title, values["n_scored"], values["n_positive"], values["n_negative"],
                            _number(values["auroc"]), _number(values["auprc"]), _number(values["fpr"]),
                            _number(values["lead_to_alert_median_s"])])
    metric_table = _table(["Split", "n", "pos", "neg", "AUROC", "AUPRC (AP)", "FPR", "Median lead→alert (s)"], metric_rows)
    early_rows = [[str(ms), _number(summary.get(f"hit_early_{ms}ms_rate")), summary.get(f"hit_early_{ms}ms_n", 0)]
                  for ms in config.get("eval", {}).get("lead_windows_ms", [500, 1000, 1500])]
    provenance = config.get("provenance", {})
    source = provenance.get("source", "Nexar train (recorded manifest)") if isinstance(provenance, dict) else str(provenance)
    manifest_path = run_dir / "manifest.csv"
    revision = "N/A"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={"video_id": str})
        if "source_repo" in manifest:
            source = ", ".join(manifest.source_repo.dropna().astype(str).unique()) or source
        if "source_revision" in manifest:
            revision = ", ".join(manifest.source_revision.dropna().astype(str).unique()) or revision
    if "synthetic" in run_dir.parts or "synthetic" in str(source).casefold():
        source = "Synthetic pipeline smoke test — Nexar 실데이터 성능으로 해석할 수 없음"
    fps = pd.to_numeric(scored.get("fps", pd.Series(dtype=float)), errors="coerce").dropna()
    fps_range = f"{fps.min():.3f}–{fps.max():.3f}" if len(fps) else "N/A"
    checks_path = run_dir / "validation_checks.json"
    checks_text = "검증 기록 파일은 아직 없음. pytest 및 별도 검증 결과를 확인해야 한다."
    if checks_path.exists():
        checks = json.loads(checks_path.read_text())
        checks_text = _validation_summary(checks)
    audit_path = run_dir / "scorer_clock_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        checks_text += (
            f"\n\n추가 감사에서 실제 영상 {len(audit.get('samples', []))}개의 흐름 방향·입력 스케일·PTS를 대조했다. "
            f"양성 {audit.get('positive_pre_event_peak_and_count_checked', 0)}개의 집계에서 "
            f"사건 이후 {audit.get('post_event_frames_stored_but_excluded', 0)}프레임이 제외됨을 확인했다. "
            "이 검사 범위에서는 구체적인 구현 오류를 발견하지 못했으며 평가 결과를 보고 설정을 바꾸지 않았다. "
            "[흐름·시간축 감사 기록](scorer_clock_audit.json)."
        )
    auroc = summary.get("auroc")
    if auroc is None:
        performance_note = "Evaluation 양쪽 클래스가 충분하지 않아 AUROC로 분리 성능을 판단할 수 없다."
    elif auroc < 0.55:
        performance_note = f"Evaluation AUROC={auroc:.4f}로 0.55 미만이다. 이 서브셋에서 유용한 양성/음성 분리는 확인되지 않았다. 파이프라인 실행 성공은 검출 성능 성공을 뜻하지 않는다. Flow 부호·시간축·사건 이후 제외를 별도로 검증해야 한다."
    elif auroc <= 0.60:
        performance_note = f"Evaluation AUROC={auroc:.4f}로 제한적인 분리만 관측되었다. 지시서의 v0 성공 신호인 0.60 초과에는 도달하지 않았다."
    else:
        performance_note = f"Evaluation AUROC={auroc:.4f}로 지시서의 v0 성공 신호인 0.60을 초과했다. 이 편의 표본의 결과만으로 다른 데이터나 실제 운전 성능을 보장할 수 없다."
    ids = ", ".join(clips.video_id.astype(str))
    status = json.dumps(metrics["all"].get("status_counts", {}), ensure_ascii=False)
    relative_run = str(run_dir)
    text = f"""# Fly-TTC v0 Report

이 점수는 초파리 LPLC2/Giant Fiber에서 영감을 받은 고정 팽창 검출기다. 전체 초파리 뇌를 시뮬레이션한 것이 아니다.

## Setup

모델: looming proxy inspired by LPLC2/GF, v0 Farneback expansion. 출력은 arbitrary unit이다.
분석 fps={config.get('video', {}).get('target_fps', 'N/A')}, short side={config.get('video', {}).get('short_side', 'N/A')}px, EMA={config.get('v0', {}).get('ema', 'N/A')}.
평가 성공 클립의 실제 입력 fps 범위={fps_range}. 시간축은 실제 프레임 인덱스/fps에 config의 offset만 적용한다.
theta={_number(theta, 6)}, validation negative FPR={_number(threshold.get('val_fpr'))}, target FPR={_number(threshold.get('fpr_target'))}.
표준화와 임계값은 음성 validation에서만 보정한다. 양성 통계 또는 evaluation 통계로 보정하지 않는다.

## Data used (ids, seed, n)

Source: {source}
Source revision: {revision}
seed={config.get('seed', 'N/A')}, 요청 n={len(clips)}, 평가 성공 n={metrics['all']['n_scored']}, 제외 n={metrics['all']['n_excluded']}.
Status counts: {status}
IDs: {ids or '없음'}
이 소량 서브셋은 scene/light 다양성을 고려한 편의 표본이며 전체 Nexar train의 무작위 대표 표본 또는 배포 환경의 사고 발생률 추정치가 아니다.
양성 trace {len(selection.get('positive', []))}개, 음성 trace {len(selection.get('negative', []))}개를 같은 y축 범위로 그렸다. 최대 요청은 각 {n_traces}개다.

## Metrics

{metric_table}

{performance_note}

핵심 성능은 Evaluation 행이다. Validation은 보정에 사용했고 All에는 그 클립이 포함되어 독립 성능 추정치가 아니다. N/A는 평가 가능한 표본 또는 양쪽 클래스가 부족함을 뜻하며 0으로 대체하지 않았다.
양성 점수 최대값과 모든 사건 메트릭은 t < time_of_event만 사용한다. 그래프의 회색 사건 이후 구간은 집계에서 제외한다. 음성은 분석 구간 전체를 사용한다.
임계값 판정은 S > theta (엄격한 초과), FPR 단위는 프레임이 아닌 클립이다. s_at_alert는 alert 이하 마지막 샘플, s_at_event는 event 미만 마지막 샘플이다.
lead는 첫 발화가 아니라 사건 이전 최대 점수 시각 기준이다. Evaluation median lead→event={_number(summary.get('lead_to_event_median_s'))}s; lead→alert IQR=[{_number(summary.get('lead_to_alert_q25_s'))}, {_number(summary.get('lead_to_alert_q75_s'))}]s, n={summary.get('lead_to_alert_n', 0)}.
이 peak lead 통계에는 임계값을 넘지 못한 양성도 포함된다. 따라서 median lead→alert={_number(summary.get('lead_to_alert_median_s'))}s를 실제 조기 경보의 중앙값으로 해석하면 안 된다.
hit_alert={_number(summary.get('hit_alert_rate'))}, n={summary.get('hit_alert_n', 0)}. 창은 [alert−{config.get('eval', {}).get('alert_window_pre_s', 1.5)}s, event)다.
Early hit은 event−window 이하의 관측 프레임에서 한 번이라도 초과했는지다. 관측 구간이 cutoff 이후 시작하면 N/A다.

{_table(['Early window (ms)', 'Evaluation hit rate', 'Observable positives'], early_rows)}

scene/light별 수치와 분모는 metrics_by_scene.csv, metrics_by_light.csv에 저장했다.

## What worked

Evaluation에서 양성 {summary['n_positive']}개와 음성 {summary['n_negative']}개가 집계되었다. 양성 pre-event threshold hit={event_hits}/{summary.get('hit_event_n', 0)}, rate={_number(summary.get('hit_event_rate'))}.
진단 패널: {'성공 클립 ' + str(diagnostic_id) + ', 전체 프레임 블러 적용' if diagnostic_ok else '성공 클립 캐시 없음; 대체 이미지를 만들지 않음'}.
합성 불변식 테스트 통과 여부는 실행 시 pytest 결과로 확인한다. 이 리포트 자체가 합성 테스트 통과를 추정하지 않는다.

{checks_text}

## What failed (and why that is expected)

화면 중심 대리 지표의 실패를 실제 초파리의 능력으로 일반화하지 않는다. 사고 유형 이름보다 관측 영상에 대상 물체의 팽창 단서가 있었는지를 확인해야 한다.
실패 태그는 영상 사고 유형의 정답이 아닌 휴리스틱이다. 태그는 중복 가능하고, 비율 분모는 해당 split의 평가 성공 클립 전체다. 필요한 flow 진단 또는 validation 기준이 없으면 해당 태그를 추측하지 않는다.
규칙은 src/fly_ttc/eval/failure_tags.py와 config.yaml의 failure_tags에 공개되어 있다. good_loom은 마지막 1초 radial 평균이 음성 validation radial 95퍼센타일을 초과함을 뜻한다.
Blob의 최초 출현은 이전 물체 지름이 없으므로 증가 점수를 0으로 둔다. 이는 출현 자체를 팽창으로 세지 않는 구현 선택이다. 단순 전역 밝기 깜빡임 합성 검사는 전체 조명 변화 불변성을 보장하지 않는다. 실제 텍스처, 노출 변화, 와이퍼는 여전히 흐름과 blob 항에 영향을 줄 수 있다.
AUROC가 낮으면 flow 부호, 시간축, 사건 이후 제외를 먼저 점검해야 하며, 이 보고서는 낮은 값을 숨기지 않는다.

""" + "\n".join("- " + limitation for limitation in LIMITATIONS) + f"""

## How to reproduce

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m fly_ttc.cli run --model v0 --config "{relative_run}/config.yaml"
python -m fly_ttc.cli report --run "{relative_run}"
```

clip_scores.csv, threshold.yaml, config.yaml, metrics.json 및 매니페스트를 함께 보존한다. 모든 양성과 위양성 음성의 parquet를 보존하며, 정상 음성 trace는 보고서 생성 후 그림으로만 보존한다. 재생성 시 기존 정상 음성 그림을 유지한다.

## Next (v1)

v1은 구현하지 않았다. v0 리포트 검토 이후 논문 STAR Methods에 근거한 4방향 모델을 검토한다. 스케치 대체 구현이라면 proxy_not_zhao_star로 명시해야 한다. BADAS-Open: optional, skipped.

"""
    figure_names = ["positive_traces.png", "negative_traces.png", "lead_to_alert.png", "roc_pr.png", "loom_diagnostic.png", "failure_modes.png"]
    text += "\n".join(f"![{name.removesuffix('.png')}](figures/{name})\n" for name in figure_names)
    report_path = run_dir / "REPORT.md"
    report_path.write_text(text, encoding="utf-8")
    (run_dir / "REPORT.html").write_text(_html_report(text, figure_names), encoding="utf-8")
    temporary = run_dir / "_report_frames"
    if temporary.exists():
        shutil.rmtree(temporary)
    return report_path
