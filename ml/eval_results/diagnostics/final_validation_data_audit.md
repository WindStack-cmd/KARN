# Final ML validation data audit

## Result

No additional compatible real Aditya co-observed telemetry exists in this
repository. No model was trained, selected, calibrated, or re-evaluated during
this audit.

## Compatible observation periods

Only the following raw paired SoLEXS + HEL1OS FITS periods are available:

| Period | Real telemetry interval | Rows | Scientific label coverage |
|---|---|---:|---|
| `day_20260730` | 2026-07-30 12:00:00–23:59:51 | 43,191 | Quiet and M (901 source-label rows) |
| `day_20260801` | 2026-08-01 12:00:00–23:59:46 | 43,187 | Quiet and X (901 source-label rows) |

The synchronized timeline contains 86,378 rows and exactly these two
partitions. The only project raw paired FITS files are:

- `data/raw/day_20260730/solexs_sample.fits.gz`
- `data/raw/day_20260730/hel1os_sample.fits`
- `data/raw/day_20260801/solexs_sample.fits.gz`
- `data/raw/day_20260801/hel1os_sample.fits`

FITS files under Python environments are package test fixtures, not KARN
telemetry, and were excluded.

## Other data sources

`full_catalog_v2.csv` and `donki_flare_catalog.csv` contain historical C/M/X
events, while `goes_flux_combined.csv` contains GOES telemetry. Neither has the
paired SoLEXS+HEL1OS measurements needed to derive KARN's 21-feature input
schema. They cannot legally be appended as KARN training or validation windows.

## Why positive chronological validation is impossible

The July 30 M event occurs at the beginning of the only earlier observation
period. A chronological validation portion containing that event would leave no
earlier Quiet data for a mixed-class training set. The only later paired period
is August 1, which is the pre-designated untouched X test period. Using it for
validation would contaminate the final temporal-generalization test.

Therefore the repository cannot provide a chronology with all of:

1. Mixed Quiet+flare training data;
2. A later independent flare-positive validation period; and
3. An untouched August 1 X test period.

## Preserved experiment and integrity

The existing temporal experiment remains a valid exploratory temporal
generalization experiment: real July 30 Quiet+M training and untouched August
1 Quiet+X test. It is not a production classifier because no independent
positive validation set exists and its false-positive burden is high.

The existing dataset report records 21 features, 30-step lookback, NaN=0,
Inf=0, and zero raw-window overlap across splits. No raw telemetry, labels,
tensors, checkpoint, or existing evaluation artifact was changed by this audit.

## Minimum additional real data required

Acquire at least one additional paired SoLEXS+HEL1OS period, chronologically
between a mixed-class training period and the final August 1 test, that contains
a real labeled flare. Prefer multiple independent paired periods containing
Quiet, M, C, and X activity. Then rebuild causally, reserve a flare-positive
validation period, select loss/regularization/threshold on validation only, and
evaluate August 1 exactly once as final test.

## Dashboard readiness

The dashboard may present the current temporal experiment as **exploratory**,
including X recall, false positives, and the data-coverage limitation. It must
not present it as a validated production flare classifier.
