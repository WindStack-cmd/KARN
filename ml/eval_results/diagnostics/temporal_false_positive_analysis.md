# Temporal binary false-positive diagnostic

This is a read-only descriptive analysis of the fixed temporal experiment. It
does not calibrate a threshold, alter a checkpoint, or propose a test-selected
model change.

## Integrity

- Checkpoint: `ml/checkpoints/karn_binary_flare_temporal_experiment.pt`
- Test artifact: `ml/eval_results/temporal_binary_eval_predictions.npz`
- Test data: persisted August 1 binary tensors only.
- Input: `(43158, 30, 21)`.
- NaN / Inf: `0 / 0`; leakage audit: `PASS`.
- Threshold calibration: **blocked**. Validation has zero positive flare labels.

## Training probability separation

| Original training group | P(Flare) median | P(Flare) mean | P(Flare) 95th percentile |
|---|---:|---:|---:|
| Quiet | 0.000033 | 0.146386 | 0.831115 |
| M flare | 0.906906 | 0.903877 | 0.938346 |

The final training confusion matrix was `[[29930, 5861], [0, 872]]` (true
rows, predicted columns). M recall was 1.0, while 5,861 Quiet training windows
were predicted as flare. This confirms learned M-associated signal but poor
Quiet specificity. Probability distributions overlap: 6.64% of Quiet windows
were at least the minimum M probability, and 88.76% of M windows were below
the maximum Quiet probability.

## Untouched-test false-positive episodes

There were 22,097 false-positive Quiet windows in 225 consecutive episodes.
38 episodes lasted at least 60 seconds and account for 19,286 false positives.

| Windows | Start | End | Anomaly-flagged rows |
|---:|---|---|---:|
| 5,351 | 2026-08-01 12:00:43 | 2026-08-01 13:29:53 | 972 |
| 2,045 | 2026-08-01 15:22:30 | 2026-08-01 15:56:34 | 360 |
| 1,550 | 2026-08-01 19:24:13 | 2026-08-01 19:50:02 | 270 |
| 1,213 | 2026-08-01 15:02:12 | 2026-08-01 15:22:24 | 218 |
| 1,000 | 2026-08-01 19:50:12 | 2026-08-01 20:06:51 | 170 |

| Test group | Count | Mean P(Flare) | Anomaly rate | Quality rate |
|---|---:|---:|---:|---:|
| False-positive Quiet | 22,097 | 0.835741 | 0.178169 | 0.0 |
| True-negative Quiet | 20,160 | 0.070153 | 0.171875 | 0.0 |
| True X | 901 | 0.888503 | 0.187569 | 0.0 |

False-positive Quiet mean SoLEXS flux was 8.72, close to true X at 7.89, and
substantially below true-negative Quiet at 17.42. The binary model's high
positive output is therefore associated with a real telemetry regime shared by
the false positives and the X event. `anomaly_flag` rates are similar across
false positives, true negatives, and X, and `quality_flag` is always zero; they
do not identify the false-positive mechanism.

## Label/window audit

The lookback is 30 **timesteps (seconds)**, not 30 minutes. Each window uses
rows `i-29` through `i`; the target is the original scientific label at endpoint
`i`. Flare labels are assigned from catalog start through peak. Rolling features
use `closed='left'`, so they use observations before the target timestamp. The
existing 29-row purge prevents raw-window overlap across train/validation; the
final August 1 observation period remains isolated as test.

## Interpretation and next experiment

The 21:1 inverse-frequency class weighting was calculated correctly from real
training labels (Quiet 0.51218; Flare 21.02236). It makes flare errors roughly
41 times more costly than Quiet errors, which plausibly explains the model's
high-sensitivity/low-specificity operating point. This is an inference from
the objective and descriptive behavior, not a proof of causality.

Do not adjust weights, focal-loss parameters, architecture, or a probability
threshold using this test result. The safest next experiment requires a real
positive validation period: collect another earlier co-observed Aditya+HEL1OS
period containing a flare, then pre-register candidate loss weighting / focal
loss / regularization choices and calibrate threshold on validation only. Until
then, the current data can support descriptive training-only ablations but not
scientifically calibrated model selection.
