"""Render a Korean research film from the frozen synthetic reflex experiment.

Run with --preview for stills, or without it for the full 44-second film.
Presentation only: no model/configuration/threshold or original result is edited.
Requires the project environment, Pillow, and ffmpeg/ffprobe on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from fly_ttc.experiments.local_reflex import verify
from fly_ttc.models.local_reflex_proxy_not_zhao_star import LocalReflexConfig, LocalReflexScorer
from fly_ttc.synthetic.reflex_stimuli import StimulusSpec, render_frame, timestamps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/synthetic/local_reflex"
OUT = ROOT / "outputs/synthetic/local_reflex_video"
EXPORT = ROOT / "exports"
MODEL = "proxy_not_zhao_star"
W, H, FPS, DURATION = 1080, 1350, 30, 44
MAX_RESPONSE = 0.0021
BG = (13, 20, 25)
PANEL = (22, 32, 38)
LINE = (51, 68, 75)
INK = (242, 242, 226)
MUTED = (160, 178, 181)
TEAL = (100, 236, 192)
AMBER = (255, 184, 109)
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
CASES = {"loom": "challenge_0023", "recede": "challenge_0039",
         "fast": "challenge_0283", "dark": "challenge_0163",
         "bright": "challenge_0167", "central_bright": "challenge_0031",
         "translate": "challenge_0071"}
SCENES = [(0, 4, "질문"), (4, 11, "접근"), (11, 19, "오반응"),
          (19, 29, "맹점"), (29, 37, "검증"), (37, 44, "다음 실험")]
DATA = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_files():
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    files = [ROOT / p for p in protocol["code_sha256"]]
    files += [ROOT / "configs/local_reflex.yaml"]
    files += sorted(p for p in SOURCE.rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    EXPORT.mkdir(exist_ok=True)
    verify(ROOT, ROOT / "configs/local_reflex.yaml", json.loads((SOURCE / "protocol.json").read_text()))
    before = frozen_files()
    cfg = yaml.safe_load((SOURCE / "config.yaml").read_text())
    cases = {c["case_id"]: c for c in json.loads((SOURCE / "cases.json").read_text())}
    stored = pd.read_parquet(SOURCE / "traces.parquet")
    scores = pd.read_csv(SOURCE / "case_scores.csv")
    checks = {}
    for key, case_id in CASES.items():
        case = cases[case_id]
        values = dict(case["spec"])
        for field in ("image_size", "center_fraction"):
            values[field] = tuple(values[field])
        spec = StimulusSpec(**values)
        ts = timestamps(spec.duration_s, case["schedule"])
        scorer = LocalReflexScorer(LocalReflexConfig(**cfg["model"]))
        inputs, maps, signal = [], [], []
        for t in ts:
            gray = render_frame(spec, float(t))
            state = scorer.update(gray, float(t))
            inputs.append(gray)
            maps.append(state["activation_map"])
            signal.append(state["S"])
        signal, maps = np.array(signal), np.array(maps)
        ref = stored[(stored.case_id == case_id) & (stored.model == MODEL)].sort_values("t_s")
        row = scores[(scores.case_id == case_id) & (scores.model == MODEL)].iloc[0]
        np.testing.assert_array_equal(ts, ref.t_s.to_numpy())
        np.testing.assert_allclose(signal, ref.S.to_numpy(), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(maps.max(axis=(1, 2)), signal, rtol=0, atol=0)
        peak = float(signal[ts >= cfg["assessment"]["warmup_s"]].max())
        np.testing.assert_allclose(peak, row.s_peak, rtol=1e-12, atol=1e-15)
        assert maps.shape == (61, 5, 5) and maps.max() <= MAX_RESPONSE
        DATA[key] = dict(inputs=np.array(inputs), maps=maps, signal=signal, ts=ts,
                         peak=peak, case=case, theta=float(row.theta))
        np.savez_compressed(OUT / f"replay_{key}.npz", inputs=np.array(inputs), maps=maps,
                            S=signal, t_s=ts, centers=state["unit_centers_xy"])
        checks[key] = {"case_id": case_id, "n_frames": len(ts), "peak": peak,
                       "max_trace_absolute_error": float(np.max(np.abs(signal - ref.S.to_numpy()))),
                       "max_map": float(maps.max()), "theta": float(row.theta)}
    assert not DATA["bright"]["maps"].any()
    assert DATA["central_bright"]["peak"] > DATA["central_bright"]["theta"]
    a, b = dict(DATA["fast"]["case"]["spec"]), dict(DATA["translate"]["case"]["spec"])
    assert a.pop("motion_scale") == 4 and b.pop("motion_scale") == 1 and a == b
    challenge = scores[(scores.split == "challenge") & (scores.model == MODEL)]
    pos, neg = challenge[challenge.label == 1], challenge[challenge.label == 0]
    assert len(cases) == 354 and scores.model.nunique() == 9
    assert len(pos) == 100 and len(neg) == 236
    assert int(pos.detected.sum()) == 84 and int(neg.detected.sum()) == 61
    off_ids = [c["case_id"] for c in cases.values() if c["split"] == "challenge"
               and c["spec"]["kind"] == "loom" and c["spec"]["center_fraction"] != [0.5, 0.5]]
    off = challenge[challenge.case_id.isin(off_ids)]
    assert len(off[off.polarity == -1]) == 16 and int(off[off.polarity == -1].detected.sum()) == 16
    assert len(off[off.polarity == 1]) == 16 and int(off[off.polarity == 1].detected.sum()) == 0
    assert before == frozen_files()
    report = {"source_artifacts_sha256": before, "selected_cases": checks,
              "common_linear_color_scale": [0, MAX_RESPONSE],
              "playback": "30 Hz recorded samples at 0.5x; no interpolated model states; final state held",
              "input_display": "Fixed linear grayscale 0..1; nearest-neighbor display resize",
              "map_display": "Actual 5x5 EMA activation; spatial cell order preserved; common linear colors",
              "model_and_results_unchanged": True,
              "stats": {"conditions": 354, "models": 9, "calibration": 18, "challenge": 336,
                        "expansion_detected": 84, "expansion_total": 100,
                        "nuisance_false_response": 61, "nuisance_total": 236}}
    (OUT / "replay_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified 427 replayed frames against frozen traces; aggregate claims verified.", flush=True)
    return before


@lru_cache(maxsize=64)
def font(size, weight="regular"):
    return ImageFont.truetype(MONO if weight == "mono" else FONT, size,
                              index=0 if weight in ("regular", "mono") else 6)


def txt(im, xy, text, size=30, color=INK, weight="regular", anchor="lt"):
    draw = ImageDraw.Draw(im)
    face = font(size, weight)
    box = draw.textbbox(xy, text, font=face, anchor=anchor)
    # Text clipping is a production failure; all labels use the same check.
    if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
        raise ValueError(f"Text outside canvas: {text!r}: {box}")
    draw.text(xy, text, font=face, fill=color, anchor=anchor)


def rule(im, y, x1=64, x2=1016, color=LINE):
    ImageDraw.Draw(im).line((x1, y, x2, y), fill=color, width=1)


def pill(im, xy, label, color=TEAL, size=22):
    x, y = xy
    tw = ImageDraw.Draw(im).textlength(label, font=font(size, "bold"))
    ImageDraw.Draw(im).rounded_rectangle((x, y, x + tw + 28, y + size + 18), radius=8,
                                         fill=tuple(int(c * .12 + b * .88) for c, b in zip(color, BG)))
    txt(im, (x + 14, y + 9), label, size, color, "bold")


def ease(value):
    x = min(1, max(0, value))
    return x * x * (3 - 2 * x)


def base(scene, t):
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)
    txt(im, (64, 58), "FLY / LOCAL REFLEX", 23, TEAL, "mono")
    txt(im, (1016, 59), "RESEARCH FILM  01", 20, MUTED, "mono", "rt")
    rule(im, 111)
    rule(im, 1216)
    for i, (start, end, label) in enumerate(SCENES):
        x = 64 + i * 162
        width = 142
        draw.rounded_rectangle((x, 1243, x + width, 1247), radius=2, fill=LINE)
        frac = np.clip((t - start) / (end - start), 0, 1)
        if frac:
            draw.rounded_rectangle((x, 1243, x + width * frac, 1247), radius=2, fill=TEAL)
        txt(im, (x, 1270), f"{i+1:02d}  {label}", 20, INK if i == scene else MUTED)
    return im


def sample(key, elapsed, delay=0.5):
    # Hold the actual initial/final sample. Never synthesize intermediate states.
    n = int(math.floor(max(0.0, elapsed - delay) * 15 + 1e-7))
    return min(n, 60)


def input_panel(im, key, idx, x, y, size):
    gray = np.rint(DATA[key]["inputs"][idx] * 255).clip(0, 255).astype("uint8")
    panel = Image.fromarray(gray).convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    im.paste(panel, (x, y))
    d = ImageDraw.Draw(im)
    d.rectangle((x-1, y-1, x+size, y+size), outline=LINE, width=1)
    # Registration marks are outside the source image.
    for xx, yy, sx, sy in [(x,y,1,1),(x+size,y,-1,1),(x,y+size,1,-1),(x+size,y+size,-1,-1)]:
        d.line((xx, yy, xx+sx*12, yy), fill=INK, width=2)
        d.line((xx, yy, xx, yy+sy*12), fill=INK, width=2)


def map_panel(im, key, idx, x, y, size):
    d = ImageDraw.Draw(im)
    gap = 5
    low, high = np.array(PANEL), np.array(TEAL)
    for row in range(5):
        for col in range(5):
            amplitude = DATA[key]["maps"][idx, row, col] / MAX_RESPONSE
            color = tuple(np.rint(low + amplitude * (high - low)).astype(int))
            x0, y0 = x + round(col * size / 5), y + round(row * size / 5)
            x1, y1 = x + round((col + 1) * size / 5) - gap, y + round((row + 1) * size / 5) - gap
            d.rounded_rectangle((x0, y0, x1, y1), radius=7, fill=color, outline=LINE, width=1)


def arrow(im, x, y, color=MUTED):
    d = ImageDraw.Draw(im)
    d.line((x - 23, y, x + 23, y), fill=color, width=2)
    d.line((x + 15, y - 8, x + 23, y, x + 15, y + 8), fill=color, width=2)


def scale(im, y, x=616, width=395):
    d = ImageDraw.Draw(im)
    for i in range(width):
        c = tuple(np.rint(np.array(PANEL) + (i/(width-1))*(np.array(TEAL)-np.array(PANEL))).astype(int))
        d.line((x+i, y, x+i, y+7), fill=c)
    txt(im, (x, y+21), "0", 18, MUTED, "mono")
    txt(im, (x+width, y+21), "2.1 × 10⁻³ AU", 19, MUTED, "regular", "rt")


def clock_label(im, key, idx, x, y, right=1016):
    t = DATA[key]["ts"][idx]
    txt(im, (x, y), f"자극 시각 {t:0.2f} s", 23, MUTED)
    state = "초기 안정화" if t < .25 else ("마지막 프레임 유지" if idx == 60 else "0.5배속")
    txt(im, (right, y), state, 23, MUTED, anchor="rt")


def replay_meta(im, key, idx, still=False):
    txt(im, (1016, 153), "공통 척도 0–2.1 × 10⁻³ AU · 선형", 20, MUTED, anchor="rt")
    t = DATA[key]["ts"][idx]
    state = "같은 시각의 정지 화면" if still else (
        "초기 안정화 · 0.5배속" if t < .25 else (
            "마지막 프레임 유지" if idx == 60 else "0.5배속"))
    txt(im, (1016, 187), f"자극 {t:0.2f} s · {state}", 20, MUTED, anchor="rt")


def trace(im, key, idx, rect, color=TEAL):
    x, y, width, height = rect
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((x, y, x+width, y+height), radius=10, fill=PANEL)
    pad = 22
    left, right, top, bottom = x+pad, x+width-pad, y+18, y+height-22
    d.line((left, bottom, right, bottom), fill=LINE)
    # The first 250ms are present in the replay but excluded from scoring.
    d.rectangle((left, top, left+(right-left)*.125, bottom), fill=(28, 40, 46))
    series = DATA[key]["signal"]
    points = [(left + j/60*(right-left), bottom - float(series[j]/MAX_RESPONSE)*(bottom-top))
              for j in range(idx+1)]
    if len(points) > 1:
        d.line(points, fill=color, width=3)
    xx, yy = points[-1]
    d.ellipse((xx-4, yy-4, xx+4, yy+4), fill=color)


def pair_row(im, key, idx, y, label, note, color=TEAL):
    txt(im, (64, y), label, 31, color, "bold")
    txt(im, (1016, y+4), note, 22, MUTED, anchor="rt")
    input_panel(im, key, idx, 64, y+54, 260)
    arrow(im, 383, y+184)
    map_panel(im, key, idx, 446, y+54, 260)
    txt(im, (762, y+80), "현재 S", 21, MUTED)
    txt(im, (762, y+115), f'{DATA[key]["signal"][idx]*1000:.3f}', 42, color, "mono")
    txt(im, (762, y+169), "× 10⁻³ AU", 22, MUTED)
    trace(im, key, idx, (755, y+219, 261, 95), color)


def render_scene(scene, elapsed, t):
    im = base(scene, t)
    d = ImageDraw.Draw(im)
    if scene == 0:
        pill(im, (64, 156), "초파리 시각에서 얻은 힌트")
        txt(im, (64, 243), "다가오는 물체를", 76, INK, "bold")
        txt(im, (64, 338), "작은 회로가 알아볼까?", 76, INK, "bold")
        txt(im, (64, 468), "움직임을 읽는 작은 모델을 만들고, 시험했습니다.", 33, MUTED)
        idx = sample("loom", elapsed, 0)
        replay_meta(im, "loom", idx)
        input_panel(im, "loom", idx, 140, 598, 340)
        arrow(im, 546, 768, TEAL)
        map_panel(im, "loom", idx, 616, 598, 340)
        txt(im, (140, 960), "합성 입력", 25, MUTED)
        txt(im, (616, 960), "실제 모델 반응", 25, TEAL)
        txt(im, (64, 1081), "공학적 근사 모델의 합성 실험", 30, INK)
        txt(im, (64, 1132), "고정된 실험을 재생합니다.  ·  0.5배속", 25, MUTED)
    elif scene == 1:
        pill(im, (64, 155), "01 / APPROACH")
        txt(im, (64, 240), "커지는 원판에", 72, INK, "bold")
        txt(im, (64, 329), "반응이 생깁니다.", 72, INK, "bold")
        txt(im, (64, 435), "합성 입력 128 × 128", 24, MUTED)
        txt(im, (616, 435), "국소 반응 5 × 5", 24, TEAL)
        idx = sample("loom", elapsed)
        replay_meta(im, "loom", idx)
        input_panel(im, "loom", idx, 64, 486, 400)
        arrow(im, 541, 685, TEAL)
        map_panel(im, "loom", idx, 616, 486, 400)
        clock_label(im, "loom", idx, 64, 912, 464)
        scale(im, 906)
        txt(im, (64, 993), "최대 국소 반응 S", 25, MUTED)
        txt(im, (1016, 990), f'{DATA["loom"]["signal"][idx]*1000:.3f} × 10⁻³ AU', 31, TEAL, "regular", "rt")
        trace(im, "loom", idx, (64, 1039, 952, 108))
        txt(im, (64, 1168), "모든 지도·그래프는 같은 선형 척도", 21, MUTED)
    elif scene == 2:
        pill(im, (64, 155), "02 / COUNTEREXAMPLE", AMBER)
        txt(im, (64, 236), "하지만, 움직임을", 65, INK, "bold")
        txt(im, (64, 316), "완벽히 가려내진 못합니다.", 65, INK, "bold")
        txt(im, (64, 414), "입력", 22, MUTED)
        txt(im, (446, 414), "실제 모델 반응", 22, MUTED)
        idx = sample("fast", elapsed, 1)
        replay_meta(im, "fast", idx)
        pair_row(im, "recede", idx, 464, "멀어짐", "이 예에서는 거의 0")
        pair_row(im, "fast", idx, 844, "빠른 평행 이동", "크기 일정 · 기준 속도 ×4 · 오반응", AMBER)
        txt(im, (64, 1190), "팽창하지 않아도 반응하는 조건이 있습니다.", 21, AMBER)
    elif scene == 3:
        pill(im, (64, 155), "03 / THE BLIND SPOT", AMBER)
        txt(im, (64, 235), "같은 위치, 같은 팽창.", 66, INK, "bold")
        txt(im, (64, 317), "밝기를 바꾸자 갈렸습니다.", 66, INK, "bold")
        note = ("집계: 중심 밖 4개 위치 × 4개 샘플링 일정" if elapsed >= 6.0
                else "중심 밖, 왼쪽 위치의 두 원판을 비교합니다.")
        txt(im, (64, 418), note, 28, MUTED)
        idx = sample("dark", elapsed, 1)
        replay_meta(im, "dark", idx)
        pair_row(im, "dark", idx, 490, "어두운 원판", "중심 밖 조건 16/16 검출")
        pair_row(im, "bright", idx, 867, "밝은 원판", "중심 밖 조건 0/16 검출", AMBER)
        # Give the off-center aggregate scope and central sanity check their own
        # readable card after both source clips finish. Do not conceal the zero map.
        if elapsed >= 6.0:
            txt(im, (64, 1187), "중앙의 밝은 원판은 검출됩니다. 위치와 밝기의 조합이 문제입니다.", 23, MUTED)
    elif scene == 4:
        pill(im, (64, 155), "04 / MEASURE EVERYTHING")
        txt(im, (64, 242), "잘된 장면만으로는", 69, INK, "bold")
        txt(im, (64, 328), "충분하지 않아서.", 69, INK, "bold")
        txt(im, (64, 449), "354개 합성 조건 × 9개 모델", 40, TEAL, "bold")
        txt(im, (64, 508), "보정 18개 / 평가 336개 · 아래 수치는 이 근사 모델", 25, MUTED)
        d.rounded_rectangle((64, 592, 1016, 803), radius=20, fill=PANEL)
        txt(im, (98, 625), "팽창 검출", 29, INK, "bold")
        txt(im, (979, 615), "84 / 100", 72, TEAL, "bold", "rt")
        txt(im, (98, 687), "84%", 25, TEAL, "mono")
        for j in range(100):
            x = 98 + j * 8.85
            d.rounded_rectangle((x, 750, x+5.5, 770), radius=2, fill=TEAL if j < 84 else LINE)
        d.rounded_rectangle((64, 828, 1016, 1039), radius=20, fill=PANEL)
        txt(im, (98, 861), "비팽창 오반응", 29, INK, "bold")
        txt(im, (979, 849), "61 / 236", 72, AMBER, "bold", "rt")
        txt(im, (98, 920), "25.8%", 25, AMBER, "mono")
        for j in range(236):
            x = 98 + j * 3.75
            d.rectangle((x, 986, x+2, 1006), fill=AMBER if j < 61 else LINE)
        txt(im, (64, 1092), "현재는 합성 자극에서 한계를 찾는 단계입니다.", 30, INK)
        txt(im, (64, 1145), "이 수치는 실제 주행 성능을 뜻하지 않습니다.", 27, MUTED)
    else:
        pill(im, (64, 155), "05 / NEXT QUESTION")
        txt(im, (64, 245), "실패가 다음 실험을", 69, INK, "bold")
        txt(im, (64, 331), "정해줬습니다.", 69, INK, "bold")
        # A compact, fully labeled still from the actual paired replay.
        peak_idx = int(np.argmax(DATA["dark"]["signal"]))
        replay_meta(im, "dark", peak_idx, still=True)
        for key, x, label in [("dark", 140, "어두운 원판"), ("bright", 640, "밝은 원판")]:
            map_panel(im, key, peak_idx, x, 488, 290)
            txt(im, (x, 801), label, 25, MUTED)
        txt(im, (537, 605), "≠", 70, AMBER, "bold", "mt")
        txt(im, (64, 894), "왜 위치와 밝기가 반응을 가를까?", 46, TEAL, "bold")
        txt(im, (64, 965), "작은 시각 모듈의 한계를, 하나씩 좁혀갑니다.", 31, MUTED)
        rule(im, 1041)
        txt(im, (64, 1072), "공학적 근사 · 합성 실험 · 학습 없이 고정된 계산", 26, INK)
        txt(im, (64, 1123), "proxy_not_zhao_star", 25, MUTED, "mono")
        txt(im, (64, 1164), "Inspired by local motion opponency · fly_ttc / 2026.09", 20, MUTED)
    return im


def frame_at(t):
    scene = max(i for i, (start, _, _) in enumerate(SCENES) if t >= start)
    start = SCENES[scene][0]
    current = render_scene(scene, t-start, t)
    # Only editorial transitions blend; source/model state samples are held.
    if scene and t-start < .2:
        previous = render_scene(scene-1, SCENES[scene-1][1]-SCENES[scene-1][0]-.001, start-.001)
        current = Image.blend(previous, current, ease((t-start)/.2))
    return current


def preview():
    shots = [0.7, 3.5, 6.8, 7.7, 10.0, 13.6, 15.7, 18.0, 21.7, 23.7, 27.0, 33.0, 40.0]
    for t in shots:
        frame_at(t).save(OUT / f"frame_{t:04.1f}s.png")
    canvas = Image.new("RGB", (4*270, 4*360), (6, 10, 13))
    d = ImageDraw.Draw(canvas)
    for j, t in enumerate(shots):
        im = frame_at(t).resize((270, 337), Image.Resampling.LANCZOS)
        x, y = (j % 4)*270, (j//4)*360
        canvas.paste(im, (x,y))
        d.text((x+8,y+340), f"{t:.1f}s", font=font(15, "mono"), fill=INK)
    canvas.save(OUT / "contact_sheet.png")
    # Poster deliberately includes both the response and its limitation.
    im = base(0, 0)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1217, W, H), fill=BG)
    pill(im, (64, 158), "초파리 시각에서 영감을 받은 실험")
    txt(im, (64, 251), "다가옴을 읽는", 86, INK, "bold")
    txt(im, (64, 359), "작은 회로.", 86, INK, "bold")
    idx = int(np.argmax(DATA["loom"]["signal"]))
    input_panel(im, "loom", idx, 64, 550, 400)
    arrow(im, 540, 750, TEAL)
    map_panel(im, "loom", idx, 616, 550, 400)
    txt(im, (64, 979), "합성 입력", 27, MUTED)
    txt(im, (616, 979), "실제 모델 반응", 27, TEAL)
    txt(im, (64, 1080), "반응하는 순간도, 실패하는 조건도.", 41, INK, "bold")
    rule(im, 1171)
    txt(im, (64, 1210), "44초 연구 기록  ·  354개 합성 조건 × 9개 모델", 27, MUTED)
    txt(im, (64, 1262), "공학적 근사 모델 / fly_ttc", 23, MUTED)
    im.save(EXPORT / "fly_ttc_reflex_poster_ko.png")


def soundtrack():
    """Original, quiet additive-synthesis score; no samples or external music."""
    sr = 48000
    t = np.arange(DURATION*sr)/sr
    audio = np.zeros((len(t), 2), dtype=np.float64)

    def tone(start, length, midi, amp, pan=0., pluck=False):
        first, last = int(start*sr), min(len(t), int((start+length)*sr))
        u = np.arange(last-first)/sr
        f = 440 * 2**((midi-69)/12)
        envelope = (1-np.exp(-u/(.008 if pluck else .9)))
        envelope *= np.exp(-u/(.7 if pluck else 12))
        envelope *= np.minimum(1, np.maximum(0, (length-u)/(.8 if pluck else 1.6)))
        # Gentle detuning and a low second harmonic, without percussive transients.
        s = (np.sin(2*np.pi*f*u) + .2*np.sin(2*np.pi*f*2*u)
             + .18*np.sin(2*np.pi*f*1.0012*u)) * envelope * amp
        audio[first:last, 0] += s*math.sqrt((1-pan)/2)
        audio[first:last, 1] += s*math.sqrt((1+pan)/2)

    chords = [(0, [50,57,60,64]), (11,[46,53,57,60]), (19,[48,55,58,62]),
              (29,[53,60,64,67]), (37,[50,57,60,64])]
    for k,(start, notes) in enumerate(chords):
        end = chords[k+1][0] if k+1 < len(chords) else DURATION
        for j,note in enumerate(notes):
            tone(start, end-start+1, note, .009, (j-1.5)/3)
    melody = [74, 76, 81, 79, 76, 74, 72, 69]
    for j,start in enumerate(np.arange(1., 41., 1.2)):
        tone(float(start), 1.8, melody[j % len(melody)], .009, .35*math.sin(j), True)
    audio *= np.minimum(1, t/1.5)[:,None]*np.minimum(1, (DURATION-t)/2.5)[:,None]
    # Conservative fixed gain; no clipping or loudness-driven peaks.
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) < .5
    pcm = np.rint(audio*32767).astype("<i2")
    path = OUT / "original_score.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(sr); wav.writeframes(pcm.tobytes())
    return path


def captions():
    lines = [
        (0,4,"다가오는 물체를 작은 회로가 알아볼까?\n초파리 시각에서 영감을 받은 합성 실험"),
        (4,11,"커지는 원판에 반응이 생깁니다.\n실제 모델 반응 · 0.5배속 · 공통 선형 척도"),
        (11,19,"이 예의 멀어짐에는 거의 0. 빠른 평행 이동에는 오반응.\n팽창하지 않아도 반응하는 조건이 있습니다."),
        (19,25,"같은 위치, 같은 팽창. 밝기를 바꾸자 갈렸습니다.\n중심 밖에서 어두운 원판 16/16, 밝은 원판 0/16 검출"),
        (25,29,"중심 밖 4개 위치 × 4개 샘플링 일정의 집계입니다.\n중앙의 밝은 원판은 검출됩니다."),
        (29,37,"354개 합성 조건 × 9개 모델. 보정 18개, 평가 336개.\n이 모델: 팽창 84/100 검출, 비팽창 61/236 오반응(25.8%)."),
        (37,44,"왜 위치와 밝기가 반응을 가를까?\n공학적 근사 모델 proxy_not_zhao_star · 합성 실험")]
    def stamp(s):
        return f"00:00:{s:02d},000"
    body = "\n\n".join(f"{i+1}\n{stamp(a)} --> {stamp(b)}\n{text}" for i,(a,b,text) in enumerate(lines))+"\n"
    (EXPORT / "fly_ttc_reflex_film_ko.srt").write_text(body)


def render_video():
    silent = EXPORT / "fly_ttc_reflex_film_ko_silent.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg must be available on PATH")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               str(silent)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        for i in range(FPS*DURATION):
            proc.stdin.write(frame_at(i/FPS).tobytes())
            if i % (FPS*4) == 0:
                print(f"Rendering {i/FPS:.0f}/{DURATION}s", flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Video encoding failed")
    music = soundtrack()
    main = EXPORT / "fly_ttc_reflex_film_ko.mp4"
    subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-y","-i",str(silent),
                    "-i",str(music),"-c:v","copy","-c:a","aac","-b:a","160k",
                    "-af","loudnorm=I=-25:TP=-3:LRA=7","-ar","48000",
                    "-movflags","+faststart","-t",str(DURATION),str(main)], check=True)
    probe = json.loads(subprocess.check_output([shutil.which("ffprobe"), "-v", "error",
                     "-show_format", "-show_streams", "-of", "json", str(main)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video["width"] == W and video["height"] == H and video["pix_fmt"] == "yuv420p"
    assert video["codec_name"] == "h264" and int(video["nb_frames"]) == FPS*DURATION
    assert abs(float(probe["format"]["duration"])-DURATION) < .1
    (OUT / "media_probe.json").write_text(json.dumps(probe, indent=2)+"\n")
    return main


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    before = prepare()
    preview()
    captions()
    if not args.preview:
        path = render_video()
        print(f"Created {path}", flush=True)
    assert frozen_files() == before, "Frozen research files changed during production"
    print("Original research artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
