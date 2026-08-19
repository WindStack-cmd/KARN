# KARN Day 11 Report (Prototype Validation Only)

## Purpose
Day 11 validated software integration for:
- Stage-1 neural binary detector training/inference/checkpointing.
- Dual-sensor fusion gate integration into a trainable model.
- End-to-end validation safety checks.

This run uses the existing **1-day prototype dataset with synthetic/prototype labels** and is **not scientifically valid for real-world solar flare forecasting**.

## Files created
- `ml/train_stage1.py`
- `ml/train_fusion_stage.py`
- `ml/validate_day11.py`
- `models/stage1_flare_detector.pt`
- `models/fusion_stage1.pt`
- `results/stage1_metrics.json`
- `results/fusion_metrics.json`
- `results/day11_report.md`

## Files modified
- None of the existing source/data pipeline files were modified for Day 11.

## Files intentionally untouched
- `ml/dataset_v3.py` (Phase 7/8 leakage-protected split logic)
- `ml/create_labels.py`
- `ml/train_baselines.py`
- `ml/sensor_fusion_gate.py`
- `ml/validate_pipeline.py`
- `data/raw/*.fits` (raw FITS untouched)
- `data/processed/aditya_features.csv` (read-only)
- `data/processed/aditya_labels.csv` (read-only)

## Dataset audit
- Features CSV shape: `(86400, 25)`
- Labels CSV shape: `(86400, 5)`
- Tensor shapes:
  - Train: `(906, 30, 23)`
  - Validation: `(258, 30, 23)`
  - Test: `(186, 30, 23)`
- Class distribution (`y_cls`):
  - Train: `{0: 875, 1: 31}`
  - Validation: `{0: 227, 2: 31}`
  - Test: `{0: 155, 3: 31}`
- NaN/Inf check on tensors: PASS
- Chronology/duplicate timestamp checks: PASS

## Phase 10 Stage-1 model
- Architecture: lightweight Conv1D temporal encoder + pooling + dropout + binary head.
- Input: `(B, 30, 23)`
- Output: one binary logit per sequence.
- Loss: `BCEWithLogitsLoss` with **computed** `pos_weight = negatives/positives` from training data.
- Parameter count: `5,377`
- Device: CPU
- Batch size: `32`
- Epoch limit: `30`
- Early stopping: enabled (patience 6), stopped at epoch 7, best epoch 1.

### Stage-1 metrics (prototype)
- Train: accuracy `0.9658`, precision `0.0000`, recall `0.0000`, F1 `0.0000`, ROC-AUC `0.9981`, PR-AUC `0.9550`
- Validation: accuracy `0.8798`, precision `0.0000`, recall `0.0000`, F1 `0.0000`, ROC-AUC `0.4768`, PR-AUC `0.1105`
- Test: accuracy `0.8333`, precision `0.0000`, recall `0.0000`, F1 `0.0000`, ROC-AUC `0.4720`, PR-AUC `0.1534`
- Undefined metric status: none (both classes present in all evaluated binary splits).

## Phase 11 fusion model integration
- Reused existing `ContextAwareFusionGate` from `ml/sensor_fusion_gate.py`.
- Dynamic feature grouping detected:
  - SoLEXS: `10`
  - HEL1OS: `10`
  - Context: `3`
- Fusion output preserved temporal structure: `(B, L, 16)` inside model before temporal head.
- Final output: one binary logit per sequence.
- Parameter count: `2,091`
- Device: CPU
- Batch size: `32`
- Epoch limit: `30`
- Early stopping: enabled (patience 6), stopped at epoch 7, best epoch 1.

### Fusion validity checks
- Differentiable forward/backward: PASS
- Weights finite: PASS
- Weights in `[0,1]`: PASS
- Weight sum-to-1 (within tolerance): PASS (`max deviation ≈ 1.19e-07`)
- No NaN/Inf in outputs: PASS
- Checkpoint reload + inference: PASS

### Fusion metrics (prototype)
- Train: accuracy `0.5508`, precision `0.0708`, recall `1.0000`, F1 `0.1322`, ROC-AUC `0.8107`, PR-AUC `0.0837`
- Validation: accuracy `0.3721`, precision `0.0982`, recall `0.5161`, F1 `0.1649`, ROC-AUC `0.3762`, PR-AUC `0.0933`
- Test: accuracy `0.2957`, precision `0.1914`, recall `1.0000`, F1 `0.3212`, ROC-AUC `0.3288`, PR-AUC `0.1247`

## Leakage and integrity validation
- Existing validation suite (`ml/validate_pipeline.py`): PASS
- Day 11 validation suite (`ml/validate_day11.py`): PASS
- Train/val/test tensor boundary integrity preserved: PASS
- No scaler fit on val/test for neural training path (none used): PASS
- No use of test set for early stopping/model selection: PASS
- Raw FITS metadata unchanged during run: PASS

## Laptop safety checks
- Device used: CPU only
- Parallel training jobs: 0 (sequential runs only)
- Hyperparameter sweeps: none
- Multiprocessing dataloader workers: 0
- Approx sequence tensor memory: ~`3.55 MB` (`X_train + X_val + X_test`)
- Approx batch tensor memory (batch=32): ~`0.084 MB` for raw input tensor
- Status: SAFE/LIGHTWEIGHT

## Prototype warning (mandatory)
- `evaluation_status`: `PROTOTYPE_ONLY`
- `scientific_validity`: `NOT_VALID_FOR_REAL_SOLAR_FLARE_PERFORMANCE`
- Current metrics validate software execution only and are **NOT** scientifically representative of real solar flare forecasting performance.

## Limitations
- Labels are synthetic/prototype labels.
- Dataset spans only one day.
- No claim is made about operational forecasting accuracy, lead time, or scientific readiness.

## Day 12 recommendation
1. Move to representative multi-day/historical labeled dataset.
2. Re-run leakage audit on expanded timeline splits.
3. Re-train Stage-1 and fusion models with calibrated thresholding.
4. Evaluate scientific metrics only after replacing prototype labels with real event-grounded labels.
