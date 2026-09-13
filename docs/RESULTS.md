# Results and interpretation

The original frozen optical-flow looming proxy scored at chance on a clip-disjoint confirmation sample from the public Nexar train set. A robust global affine-flow correction did not improve its matched raw-flow comparator. These are results for the implemented proxy and this sampling protocol, not for a whole fly brain or a deployed driving system.

## Experiment boundaries

| Stage | Clips | Use |
|---|---:|---|
| Initial calibration | 20: 10 positive, 10 negative | Only negative clips determine normalization and threshold |
| Initial evaluation | 80: 40 positive, 40 negative | Original v0 headline; later exploratory reuse for the follow-up |
| Additional confirmation | 100: 50 positive, 50 negative | Frozen four-model comparison, disjoint from the initial 100 |

All clips came from official Nexar **train**, source revision `aa97deda5a59f00bb7187739053b7c72e14374df`. Positive means collision or near-miss. Scene/light conditions were deliberately mixed; this is not a prevalence-representative road sample. All 200 selected video files passed the recorded source SHA256 checks. There were no overlapping IDs or identical source-file hashes between stages. Trip, location, driver, and source-event independence were not established.

At the initial evaluation, v0 had not been calibrated on the 80 evaluation clips. The follow-up reused those now-inspected clips as exploratory data and scored an additional frozen 100 for confirmation. All 200 are now known data and cannot be presented as a new untouched test set in later research.

## Original v0

Farneback flow is computed after conversion to 8-bit grayscale, at target sampling rate 15 Hz and short side 256 pixels. The score combines normalized positive divergence, positive radial flow, and a difference-blob expansion term:

`S_raw = z(S_div) + z(S_rad) + 0.5 * z(S_blob)`

A causal EMA with alpha 0.3 produces the score. Means and standard deviations come only from negative calibration frames. A strict crossing `S > theta` defines a response; the frozen original threshold is `theta = 6.790889832241`.

| Original split | AUROC | AP | Pre-event positive responses | Negative responses |
|---|---:|---:|---:|---:|
| Evaluation: 80 | 0.546250 | 0.523546 | 7/40 | 8/40 |
| Calibration: 20 | 0.600000 | 0.631776 | 2/10 | 1/10 |
| All initial clips: 100, including calibration | 0.559200 | 0.531681 | 9/50 | 9/50 |

Source: [`outputs/v0/metrics.json`](../outputs/v0/metrics.json). The shorthand initial AUROC **0.55** refers to evaluation 80, not all 100. The pooled 100 include calibration and do not provide an unbiased headline estimate.

## Confirmation

| Model ID | AUROC | AP | TPR | FPR | AUROC 95% interval |
|---|---:|---:|---:|---:|---|
| `v0_original` | 0.4904 | 0.537974 | 9/50 | 10/50 | [0.3788, 0.6040] |
| `raw_flow_no_blob` | 0.4872 | 0.529936 | 9/50 | 10/50 | [0.3760, 0.5988] |
| `affine_residual_no_blob` | 0.4876 | 0.528249 | 8/50 | 10/50 | [0.3744, 0.5996] |
| `frame_difference` | 0.5252 | 0.547857 | 9/50 | 10/50 | [0.4108, 0.6352] |

Source: [`outputs/ego_motion/metrics.json`](../outputs/ego_motion/metrics.json), `results` entries with `split == "confirmation"`.

The prespecified candidate was `affine_residual_no_blob`, compared with `raw_flow_no_blob`. Its AUROC difference was **+0.0004**, paired-bootstrap 95% interval **[-0.0136, +0.0160]**. Both omit the blob term, isolating this subtraction more directly than a comparison against original v0. The delta against original v0 is **-0.0028**, a different contrast. The primary comparison used 2,000 paired within-class clip resamples with seed 0 and fixed calibration; the intervals do not include uncertainty in calibration, trip-level correlation, or later adaptive choices.

The affine method fits a robust global image-plane flow field and subtracts it when quality checks pass. It is not a physical reconstruction of camera pose, depth, or road geometry. Failed fits fall back to raw flow. The existing audit found fallback at 9/10 confirmation false-positive peak frames, so many problematic scenes did not receive an accepted correction. This experiment does not show that every form of ego-motion estimation is ineffective.

## What the score and hit metrics mean

Positive processing windows extend from up to eight seconds before the event to one second after it, but **every positive event metric excludes samples at or after the event**. Negative windows use up to nine seconds near the clip middle, excluding a one-second margin. The classes therefore do not necessarily have equal observation time. This is an evaluation of label-selected windows, not of uninterrupted driving.

AUROC and AP use the maximum allowed score per clip. AP is average precision and depends on class prevalence. TPR means at least one threshold crossing before the event; FPR means at least one crossing during a negative analysis window. These counts do not measure reliable early warning. For example, initial evaluation hit rate at least one second before the event was only 4/40.

Time is derived from frame index divided by the decoder's reported FPS, with the configured offset. It is not a presentation-timestamp audit of variable-frame-rate video. Saved flow-pair intervals vary in some clips; the legacy model's per-pair flow and fixed-alpha EMA are not fully normalized in physical time. Library, decoder, and operating-system changes may also affect re-extracted flow values.

These metrics are not the official [Nexar challenge](https://arxiv.org/abs/2503.03848) AP protocol. The title `fly-ttc` does not imply that the model outputs calibrated physical time to collision.

## Examples and synthetic check

The hero uses the same original score and threshold in both panels. Event clip `00300` is highway/normal light with peak **14.240634**. Non-event clip `01538` is sub-urban/normal light with peak **15.674886**. They are illustrative selected examples, not an independent two-clip test.

The original synthetic check uses the last 15 frames of a 60-frame, 192-by-192 stimulus. Scores below are raw late means without Nexar negative-calibration normalization:

| Stimulus | Raw late-mean score |
|---|---:|
| Expanding disk | 0.324407061 |
| Receding disk | 0.001433173 |
| Translation: 1 pixel per frame pair | 0.170572273 |
| Translation: 2 pixels per frame pair | 0.343534186 |
| Translation: 4 pixels per frame pair | 0.668595200 |
| Uniform flicker | 0.000000000 |

The original proxy responds to expansion, but faster translation can exceed that response. This is a counterexample to the implementation's selectivity. It does not establish that a real fly would respond in the same way, and it does not translate into Nexar detection performance.

## Separate local-reflex experiment

The later activation-map film concerns **`proxy_not_zhao_star`**, a separate local direction-opponent engineering model. It is not the original Farneback v0, not an exact reproduction of Zhao's STAR model, and not an implementation of the whole LPLC2/Giant Fiber circuit. Its synthetic response amplitudes must not be mixed with the v0 scores above.

A fixed matrix contained 354 conditions and nine models. Eighteen conditions were calibration, with 12 shared negative conditions used for each model's threshold. The remaining 336 challenge conditions contained 100 expansion and 236 non-expansion stimuli. The local proxy detected **84/100** expansion conditions and responded to **61/236** non-expansion conditions (25.85%); synthetic challenge AUROC was **0.86568**. At off-center positions, dark disks were detected in 16/16 matched conditions and bright disks in 0/16. Central bright expansion was detected, so the failure concerns the tested position/polarity combination rather than every bright stimulus.

These deterministic, correlated synthetic conditions are diagnostics, not independent road clips. The local model did not pass its progression criteria for selectivity, position tolerance, and sampling stability. No road-performance benefit, object detector, or driving application was established from this experiment. The separate source implementation is included so the engineering choices can be inspected; the Nexar publication makes no synthetic-to-road performance transfer claim.

## Reading and reusing the release

The English README and figures summarize saved numerical results. Selected original reports retain their Korean archival text. Code and scores are public; Nexar videos, dashcam pixels, local environments, media exports, and credentials are excluded. The English movie is a presentation asset distributed separately from Git history.

Code uses the MIT license. Source data remain under [nexar-open-data-license](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE), copyright (c) 2025 Nexar Inc. Please retain the [dataset attribution](../README.md#license) and cite [Moura et al. (2025)](https://arxiv.org/abs/2503.03848). Biological inspiration comes from [Klapoetke et al. (2017)](https://doi.org/10.1038/nature24626) and [Ache et al. (2019)](https://www.sciencedirect.com/science/article/pii/S0960982219301381); neither paper reports this project's dashcam evaluation.
