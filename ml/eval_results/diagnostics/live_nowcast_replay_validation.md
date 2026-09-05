# KARN causal live-nowcast replay validation

## Scope

This validation exercises the Phase B historical-replay engine using only the
saved synchronized Aditya-compatible telemetry. It is not a validation of a
live Aditya-L1 feed, which is unavailable in this repository.

## Causality controls

- Input is consumed sequentially, one timestamp at a time.
- The 15-second median smoother uses current and previous samples only.
- The 15-minute low-quantile background is computed before appending the
  current smoothed value, so it excludes the current and all future samples.
- There is no backward fill, centred window, future label, future peak, or
  future end-time use.
- Detector state resets at an observation-partition boundary.

## Detector rules

| Control | SoLEXS | HEL1OS |
| --- | ---: | ---: |
| Smoothing | 15 seconds | 15 seconds |
| Trailing background | 900 seconds, 20th percentile | 900 seconds, 20th percentile |
| Ratio threshold | 3.0 | 4.0 |
| Minimum excess | 12 | 15 |
| Persistence / trigger | 30 seconds | 30 seconds |
| Recovery | 30 non-candidate seconds | 30 non-candidate seconds |
| Association | 300 seconds | 300 seconds |

## Observed sequential replay checks

- At replay start, no observation is consumed and both detectors are `NORMAL`.
- After 1,000 real observations, both detectors and the master remain `NORMAL`.
- A HEL1OS-only event was triggered at `2026-07-30 16:36:54` after 30 sustained
  causal candidate samples. Its start estimate was `16:36:25`; peak at trigger
  was only **peak-so-far** `182.0`, not a final peak.
- As additional observations arrived, the final peak became `203.0` at
  `16:43:36`, and causal recovery ended the event at `16:49:12` (768 seconds).
- The event accumulated 476 flagged-anomaly samples; flags are exposed as
  diagnostic event context rather than being converted into labels.
- A master `BOTH` event is only emitted when two independently active events
  trigger within 300 seconds. The current tested replay window produced the
  HEL1OS-only case; no synthetic BOTH case was introduced.

## Metrics and limits

Detection latency is governed by the required 30 seconds of persistence; it
cannot trigger before those causal samples exist. A catalogue-level false
trigger count, precision, recall, and F1 are **not available**, because no
independent per-instrument live-detection ground truth exists. The historical
M/X labels remain outside the live detector decision path.
