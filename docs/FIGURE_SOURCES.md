# English figure sources

All four publication PNGs were rendered from saved numerical artifacts. No
dashcam videos were decoded, no optical flow was recomputed, and no threshold
was retuned. The small source bundle contains scores, labels, timestamps and
source-file hashes; it contains no road images.

Regenerate the figures from the public bundle:

```bash
python scripts/make_english_figures.py
```

If the original local experiment outputs are available, refresh the numerical
bundle before plotting with `--refresh-sources`. This option only reads the
saved CSV, JSON, YAML and parquet files. It does not run a model.

## Hero: x_card.png

- Source: `outputs/v0/clip_scores.csv`, the frozen `threshold.yaml`, and the saved
  frame-score parquet files for `00300` and `01538`.
- These are two illustrative clips from the **first 80 evaluation clips**;
  they are not clips from the later confirmation set.
- Clip `00300`: labeled event, Highway; alert at 8.464 seconds, event at 10.064
  seconds; maximum **before** the event is 14.240634245723074 at 10.0 seconds.
- Clip `01538`: non-event, Sub-urban; peak 15.6748864737522 at
  23.737704918032787 seconds.
- Both panels use the same 0-18 score axis and the same frozen threshold,
  6.7908898322413265. Their horizontal axes use each original clip's timestamps.
- The positive trace contains only `t < time_of_event`. The original saved
  parquet also contains later scores, including 17.936565743477658 at
  10.066666666666666 seconds. That value is after the event and is deliberately
  excluded under the existing evaluation rule. This is essential to the
  pre-event comparison, not a new scoring decision.
- The footer's **0.49** refers separately to v0 on the new, clip-disjoint
  confirmation set of 100 clips. The image states that distinction explicitly.

The published `figures/hero_scores.csv` is an exact score-only extraction of
the plotted traces, without post-event positive rows. If a frame trace is
unavailable when refreshing the bundle, the script plots the saved scalar peak
as a bar and labels the missing trace; it never invents a time series.

## Confirmation: confirmation_metrics.png

Source: `outputs/ego_motion/comparison.csv` and `metrics.json`, split
`confirmation`, 50 positive and 50 negative clips for each model.

| Model | Exact saved AUROC | Figure label |
|---|---:|---:|
| v0 original | 0.4904 | 0.490 |
| Raw flow, no blob | 0.4872 | 0.487 |
| Affine residual, no blob | 0.4876 | 0.488 |
| Frame difference | 0.5252 | 0.525 |

The prespecified paired comparison is **affine residual minus raw flow**, both
without the blob component: +0.0004, 95% paired bootstrap interval
[-0.0136, +0.0160]. It is not a comparison against the original v0 model.
The requested display range is 0.45-0.70, and the truncated bar baseline is
disclosed on the figure. Individual AUROC intervals are available in the
source JSON; they are not shown because they extend outside that display range.

## Synthetic: synthetic_selectivity.png

Source: `outputs/review_response/audit.json`, `synthetic.stimulus_scores`,
cross-checked against `outputs/synthetic/synthetic_summary.json`. Flicker comes
from the latter file. The measure is the final 15-frame mean of the original
60-frame, 192-pixel synthetic stimulus run, without Nexar normalization.

| Stimulus | Exact saved raw late mean | Figure label |
|---|---:|---:|
| Expanding disk | 0.3244070608026703 | 0.324 |
| Receding disk | 0.001433172504714562 | 0.001 |
| Translation 1 px/frame | 0.17057227337730746 | 0.171 |
| Translation 2 px/frame | 0.3435341860893449 | 0.344 |
| Translation 4 px/frame | 0.6685952000191907 | 0.669 |
| Uniform flicker | 0.0 | 0.000 |

These are original v0 synthetic results, not the later local directional-model
experiment and not road-video scores. Raw synthetic scores are not on the
normalized scale used in the hero figure.

## Header: repo_header.png

The banner uses the v0 confirmation AUROC rounded to 0.49. It contains no
anatomical illustration, neural activity rendering or invented score trace.

## Files

- `figures/x_card.png`: 1600 x 900 pixels.
- `figures/confirmation_metrics.png`: 1600 x 900 pixels.
- `figures/synthetic_selectivity.png`: 1600 x 900 pixels.
- `figures/repo_header.png`: 1280 x 640 pixels.
- `figures/figure_sources.json`: full-precision saved values and original file
  SHA-256 hashes.
- `figures/hero_scores.csv`: 253 plotted rows, with original clip timestamps.

All visible labels are English. Visual inspection checked label spacing,
threshold and event markers, common score axes, the comparison denominator,
and the separation of the first 80 examples from the new 100 confirmation clips.
