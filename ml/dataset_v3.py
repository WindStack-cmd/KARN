import os
import pandas as pd
import numpy as np
import torch


# ============================================================
# KARN - PHASE 7 + PHASE 8
# Sequence Generation + Purged Chronological Splitting
# ============================================================

FEATURES_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
LABELS_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_labels.csv"

OUTPUT_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\training_windows"

# ------------------------------------------------------------
# PROTOTYPE CONFIGURATION
# ------------------------------------------------------------

PROTOTYPE_LOOKBACK = 30
PROTOTYPE_PURGE_GAP = 30


def execute_safe_purged_windowing():

    # ========================================================
    # LOAD
    # ========================================================

    print("Loading data tracks into memory...")

    feat_df = pd.read_csv(FEATURES_PATH)
    label_df = pd.read_csv(LABELS_PATH)

    feat_df["timestamp"] = pd.to_datetime(
        feat_df["timestamp"],
        errors="coerce"
    )

    label_df["timestamp"] = pd.to_datetime(
        label_df["timestamp"],
        errors="coerce"
    )

    # ========================================================
    # BASIC DATA SAFETY
    # ========================================================

    if feat_df["timestamp"].isna().any():
        raise ValueError("Feature dataset contains invalid timestamps.")

    if label_df["timestamp"].isna().any():
        raise ValueError("Label dataset contains invalid timestamps.")

    feat_df = feat_df.sort_values("timestamp")
    label_df = label_df.sort_values("timestamp")

    if feat_df["timestamp"].duplicated().any():
        raise ValueError(
            "Feature dataset contains duplicate timestamps."
        )

    if label_df["timestamp"].duplicated().any():
        raise ValueError(
            "Label dataset contains duplicate timestamps."
        )

    # ========================================================
    # MERGE
    # ========================================================

    master_df = pd.merge(
        feat_df,
        label_df,
        on="timestamp",
        how="inner",
        validate="one_to_one"
    )

    master_df = (
        master_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if master_df.empty:
        raise ValueError(
            "Feature/label timestamp merge produced zero rows."
        )

    master_df.set_index("timestamp", inplace=True)

    # ========================================================
    # FEATURE COLUMNS
    # ========================================================

    label_cols = [
        "flare_label",
        "flare_class",
        "time_to_peak",
        "event_id"
    ]

    excluded_cols = [
        "time_seconds"
    ]

    feature_cols = [
        c for c in master_df.columns
        if c not in label_cols
        and c not in excluded_cols
    ]

    print(
        f"\nFeature columns used: {len(feature_cols)}"
    )

    # ========================================================
    # PHASE 7
    # 60-SECOND AGGREGATION
    # ========================================================

    print(
        "\nExecuting Phase 7: "
        "Event-Aware 60-Second Block Aggregation..."
    )

    # Feature aggregation
    feature_resampled = (
        master_df[feature_cols]
        .resample("1min")
        .median()
    )

    # Classification target
    label_resampled = (
        master_df["flare_label"]
        .resample("1min")
        .max()
        .fillna(0)
    )

    class_resampled = (
        master_df["flare_class"]
        .resample("1min")
        .max()
        .fillna(0)
    )

    # Time-to-peak
    ttp = master_df["time_to_peak"].replace(
        -1.0,
        np.nan
    )

    ttp_resampled = (
        ttp
        .resample("1min")
        .max()
        .fillna(-1.0)
        .round()
    )

    minute_df = pd.concat(
        [
            feature_resampled,
            label_resampled.rename("flare_label"),
            class_resampled.rename("flare_class"),
            ttp_resampled.rename("time_to_peak")
        ],
        axis=1
    )

    # ========================================================
    # FEATURE GAP HANDLING
    # ========================================================

    # Historical feature values may be carried forward.
    # No backward fill is used.
    minute_df[feature_cols] = (
        minute_df[feature_cols]
        .ffill()
    )

    minute_df[feature_cols] = (
        minute_df[feature_cols]
        .fillna(0)
    )

    minute_df = minute_df.reset_index()

    # ========================================================
    # VALIDATION
    # ========================================================

    if minute_df["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate minute timestamps detected."
        )

    if not minute_df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Minute timeline is not chronological."
        )

    # ========================================================
    # RAW ARRAYS
    # ========================================================

    X_raw = minute_df[
        feature_cols
    ].to_numpy(dtype=np.float32)

    y_cls_raw = minute_df[
        "flare_class"
    ].to_numpy(dtype=np.int64)

    y_reg_raw = minute_df[
        "time_to_peak"
    ].to_numpy(dtype=np.float32)

    # Safety check
    if not np.isfinite(X_raw).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    # ========================================================
    # PHASE 8
    # CHRONOLOGICAL SPLIT
    # ========================================================

    print(
        "\nExecuting Phase 8: "
        "Chronological Slicing with Safety Purge Gaps..."
    )

    total_minutes = len(minute_df)

    lookback = PROTOTYPE_LOOKBACK
    purge_gap = PROTOTYPE_PURGE_GAP

    # Prototype requires enough data
    if total_minutes < (
        lookback +
        purge_gap * 2 +
        100
    ):
        raise ValueError(
            "Dataset is too small for the selected "
            "lookback and purge configuration."
        )

    # Calculate dynamic boundaries based on the actual number of minutes
    train_end = int(0.65 * total_minutes)
    val_start = train_end + purge_gap
    val_end = int(0.85 * total_minutes)
    test_start = val_end + purge_gap

    if not (
        0 < train_end <
        val_start <
        val_end <
        test_start <
        total_minutes
    ):
        raise ValueError(
            "Invalid chronological split boundaries."
        )

    # ========================================================
    # SEQUENCE BUILDER
    # ========================================================

    def build_sequences(
        start_idx,
        end_idx
    ):

        X_list = []
        y_cls_list = []
        y_reg_list = []

        # Sequence uses only historical values
        actual_start = max(
            start_idx,
            lookback
        )

        for i in range(
            actual_start,
            end_idx
        ):

            X_list.append(
                X_raw[
                    i - lookback:i
                ]
            )

            y_cls_list.append(
                y_cls_raw[i]
            )

            y_reg_list.append(
                y_reg_raw[i]
            )

        if not X_list:

            return (
                np.empty(
                    (0, lookback, len(feature_cols)),
                    dtype=np.float32
                ),
                np.empty(
                    (0,),
                    dtype=np.int64
                ),
                np.empty(
                    (0,),
                    dtype=np.float32
                )
            )

        return (
            np.asarray(
                X_list,
                dtype=np.float32
            ),
            np.asarray(
                y_cls_list,
                dtype=np.int64
            ),
            np.asarray(
                y_reg_list,
                dtype=np.float32
            )
        )

    # ========================================================
    # BUILD SPLITS
    # ========================================================

    X_train, y_train_cls, y_train_reg = (
        build_sequences(
            0,
            train_end
        )
    )

    X_val, y_val_cls, y_val_reg = (
        build_sequences(
            val_start,
            val_end
        )
    )

    X_test, y_test_cls, y_test_reg = (
        build_sequences(
            test_start,
            total_minutes
        )
    )

    # ========================================================
    # FINAL AUDIT (THE LEAKAGE AUDIT)
    # ========================================================

    print(
        "\nExecuting Final Rigorous Data Leakage Audit..."
    )

    # 1. No NaNs check
    nans = np.isnan(X_train).sum() + np.isnan(X_val).sum() + np.isnan(X_test).sum()
    nans += np.isnan(y_train_cls).sum() + np.isnan(y_val_cls).sum() + np.isnan(y_test_cls).sum()
    nans += np.isnan(y_train_reg).sum() + np.isnan(y_val_reg).sum() + np.isnan(y_test_reg).sum()
    assert nans == 0, f"Leakage Audit FAILED: {nans} NaNs detected!"

    # 2. No Infinite values check
    infs_count = 0
    for arr in [X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls, y_train_reg, y_val_reg, y_test_reg]:
        infs_count += np.isinf(arr).sum()
    assert infs_count == 0, f"Leakage Audit FAILED: {infs_count} infinite values detected!"

    # 3. Dynamic sequence shape check
    assert X_train.shape == (len(X_train), lookback, len(feature_cols)), f"X_train sequence shape check failed: {X_train.shape}"
    assert X_val.shape == (len(X_val), lookback, len(feature_cols)), f"X_val sequence shape check failed: {X_val.shape}"
    assert X_test.shape == (len(X_test), lookback, len(feature_cols)), f"X_test sequence shape check failed: {X_test.shape}"

    # 4. Monotonic chronological sorting & duplicate timestamps
    assert minute_df["timestamp"].is_monotonic_increasing, "Timeline is not chronologically sorted!"
    assert minute_df["timestamp"].is_unique, "Timeline contains duplicate timestamps!"

    # 5. Non-empty train/validation/test check
    assert len(X_train) > 0, "Train set is empty!"
    assert len(X_val) > 0, "Validation set is empty!"
    assert len(X_test) > 0, "Test set is empty!"

    # 6. Target overlap and purge gaps
    train_ts = minute_df["timestamp"].iloc[max(0, lookback):train_end]
    val_ts = minute_df["timestamp"].iloc[val_start:val_end]
    test_ts = minute_df["timestamp"].iloc[test_start:]

    # Assert target timeline has no overlaps
    assert val_ts.min() > train_ts.max(), f"Leakage Audit FAILED: Train targets {train_ts.max()} overlap Validation targets {val_ts.min()}"
    assert test_ts.min() > val_ts.max(), f"Leakage Audit FAILED: Validation targets {val_ts.max()} overlap Test targets {test_ts.min()}"

    # Assert purge gaps are strictly respected
    assert (val_ts.min() - train_ts.max()) >= pd.Timedelta(minutes=purge_gap + 1), "Purge gap between Train and Val is not respected!"
    assert (test_ts.min() - val_ts.max()) >= pd.Timedelta(minutes=purge_gap + 1), "Purge gap between Val and Test is not respected!"

    # 7. Positive events representation check
    assert (y_train_cls > 0).any(), "Validation FAILED: Train contains zero positive events!"
    assert (y_val_cls > 0).any(), "Validation FAILED: Validation contains zero positive events!"
    assert (y_test_cls > 0).any(), "Validation FAILED: Test contains zero positive events!"

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n=================================================="
    )
    print(
        "PHASE 7-8 DATASET REPORT"
    )
    print(
        "=================================================="
    )

    print(
        f"Total chronological intervals: {total_minutes}"
    )
    print()

    print(
        f"Lookback: {lookback} minutes"
    )
    print(
        f"Purge gap: {purge_gap} minutes"
    )
    print()

    print("TRAIN:")
    print(f"shape = {X_train.shape}")
    print(f"positive events > 0: {(y_train_cls > 0).sum()}")
    print()

    print("VALIDATION:")
    print(f"shape = {X_val.shape}")
    print(f"positive events > 0: {(y_val_cls > 0).sum()}")
    print()

    print("TEST:")
    print(f"shape = {X_test.shape}")
    print(f"positive events > 0: {(y_test_cls > 0).sum()}")
    print()

    def print_distribution(
        name,
        labels
    ):

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        d = dict(
            zip(unique, counts)
        )

        print(
            f"{name} Class Distribution:"
        )

        print(
            f"  Quiet : {d.get(0, 0)}"
        )

        print(
            f"  C     : {d.get(1, 0)}"
        )

        print(
            f"  M     : {d.get(2, 0)}"
        )

        print(
            f"  X     : {d.get(3, 0)}"
        )
        print()

    print_distribution(
        "TRAIN",
        y_train_cls
    )

    print_distribution(
        "VALIDATION",
        y_val_cls
    )

    print_distribution(
        "TEST",
        y_test_cls
    )

    print(
        "=================================================="
    )
    print(
        "LEAKAGE AUDIT"
    )
    print(
        "=================================================="
    )
    print("Timestamp duplicates: 0")
    print("NaNs: 0")
    print("Infinite values: 0")
    print("Chronological order: PASS")
    print("Train/Validation leakage: PASS")
    print("Validation/Test leakage: PASS")
    print("Purge-gap check: PASS")
    print("Sequence shape check: PASS")
    print()

    print(
        "=================================================="
    )
    print(
        "FINAL STATUS"
    )
    print(
        "=================================================="
    )
    print()
    print("PHASE 6-8 PROTOTYPE PIPELINE: PASS")
    print()
    print("IMPORTANT:")
    print("Synthetic labels are ONLY for software pipeline validation.")
    print("They must NOT be used for scientific model evaluation.")

    # ========================================================
    # SAVE TENSORS
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    tensors = {
        "X_train.pt": X_train,
        "y_train_cls.pt": y_train_cls,
        "y_train_reg.pt": y_train_reg,

        "X_val.pt": X_val,
        "y_val_cls.pt": y_val_cls,
        "y_val_reg.pt": y_val_reg,

        "X_test.pt": X_test,
        "y_test_cls.pt": y_test_cls,
        "y_test_reg.pt": y_test_reg,
    }

    for filename, array in tensors.items():

        torch.save(
            torch.from_numpy(array),
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )

    print(
        f"\n[OK] Success! All PyTorch tensors serialized securely in:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    execute_safe_purged_windowing()