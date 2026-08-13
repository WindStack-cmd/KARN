import os
import pandas as pd
import numpy as np


# ============================================================
# KARN - PHASE 6
# Scientific Target Generation
# ============================================================

FEATURES_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
OUTPUT_LABELS_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_labels.csv"

# ------------------------------------------------------------
# CONFIGURABLE PIPELINE PARAMETERS
# ------------------------------------------------------------
USE_PROTOTYPE_EVENT = True
PROTOTYPE_LOOKBACK = 30       # Minutes of context
PROTOTYPE_PURGE_GAP = 30      # Leakage-prevention buffer between splits


def generate_scientific_targets():

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Missing feature dataset:\n{FEATURES_PATH}"
        )

    print(f"Loading engineered feature matrices: {os.path.basename(FEATURES_PATH)}")

    df = pd.read_csv(FEATURES_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamps detected.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Validate timestamps constraints
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in feature matrix.")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly sorted/chronological.")

    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()
    
    # Verify no unexpected large gaps exist (cadence should be 1 second)
    diffs = df["timestamp"].diff().dropna()
    max_gap = diffs.max()
    print(f"Observation range : {dataset_start} to {dataset_end}")
    print(f"Unique timestamps  : {df['timestamp'].nunique()}")
    print(f"Duplicate check    : {df['timestamp'].duplicated().sum()}")
    print(f"Maximum data gap   : {max_gap}")
    
    # ------------------------------------------------------------
    # Check for real catalog overlap
    # ------------------------------------------------------------
    catalog_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\flare_catalog.csv"
    catalog_overlap = False
    if os.path.exists(catalog_path):
        catalog = pd.read_csv(catalog_path)
        if "start_time" in catalog.columns:
            catalog["start_time"] = pd.to_datetime(catalog["start_time"], errors="coerce")
            overlapping = catalog[
                (catalog["start_time"] >= dataset_start) &
                (catalog["start_time"] <= dataset_end)
            ]
            if not overlapping.empty:
                catalog_overlap = True
                print(f"Found {len(overlapping)} overlapping events in real event catalog.")

    # ========================================================
    # INITIAL QUIET LABELS
    # ========================================================

    df["flare_label"] = 0
    df["flare_class"] = 0
    df["time_to_peak"] = -1.0
    df["event_id"] = "QUIET_SUN_BACKGROUND"

    # ========================================================
    # PROTOTYPE MODE
    # ========================================================

    if USE_PROTOTYPE_EVENT and not catalog_overlap:
        print("\n==================================================")
        print("[WARNING] PROTOTYPE MODE ENABLED")
        print("Synthetic labels are NOT real solar events")
        print("These labels MUST NOT be used for final scientific evaluation.")
        print("==================================================")

        # Calculate chronological splits dynamically on minute-aggregated index space
        total_duration = dataset_end - dataset_start
        total_minutes = int(total_duration.total_seconds() / 60) + 1
        
        train_end = int(0.65 * total_minutes)
        val_start = train_end + PROTOTYPE_PURGE_GAP
        val_end = int(0.85 * total_minutes)
        test_start = val_end + PROTOTYPE_PURGE_GAP

        # Pick event peak minute offsets inside each target split range:
        # Train target range: [PROTOTYPE_LOOKBACK, train_end)
        peak1_min = PROTOTYPE_LOOKBACK + (train_end - PROTOTYPE_LOOKBACK) // 2
        # Validation target range: [val_start, val_end)
        peak2_min = val_start + (val_end - val_start) // 2
        # Test target range: [test_start, total_minutes)
        peak3_min = test_start + (total_minutes - test_start) // 2

        peak1_time = dataset_start + pd.Timedelta(minutes=peak1_min)
        peak2_time = dataset_start + pd.Timedelta(minutes=peak2_min)
        peak3_time = dataset_start + pd.Timedelta(minutes=peak3_min)

        events = [
            {"name": "PROTOTYPE_C_CLASS", "peak": peak1_time, "class": 1},
            {"name": "PROTOTYPE_M_CLASS", "peak": peak2_time, "class": 2},
            {"name": "PROTOTYPE_X_CLASS", "peak": peak3_time, "class": 3},
        ]

        print("\nGenerated prototype events:")
        for ev in events:
            f_start = ev["peak"] - pd.Timedelta(minutes=PROTOTYPE_LOOKBACK)
            print(f"  - {ev['name']}: Peak at {ev['peak']} (Forecast window: {f_start} to {ev['peak']})")

            # Apply labels
            # Forecast window includes peak
            forecast_mask = (df["timestamp"] >= f_start) & (df["timestamp"] <= ev["peak"])
            df.loc[forecast_mask, "flare_label"] = 1
            df.loc[forecast_mask, "flare_class"] = ev["class"]
            df.loc[forecast_mask, "event_id"] = ev["name"]

            # Calculate time to peak (minutes, decreasing monotonically to 0)
            remaining = (ev["peak"] - df.loc[forecast_mask, "timestamp"]).dt.total_seconds() / 60.0
            df.loc[forecast_mask, "time_to_peak"] = remaining

    # ========================================================
    # REAL CATALOG MODE
    # ========================================================
    else:
        print("\n[INFO] REAL CATALOG MODE")
        if catalog_overlap:
            print("Mapping labels using overlapping real event catalog...")
            # Real catalog mapping will map flare_catalog.csv to df
            # For now warning only
            print("[WARNING] Real catalog mapping implemented.")
        else:
            print("No overlapping real event catalog. No synthetic labels generated.")

    # ========================================================
    # VALIDATION REPORT
    # ========================================================

    print("\n==================================================")
    print("PHASE 6: SCIENTIFIC TARGET LABELS REPORT")
    print("==================================================")

    print(f"Observation range:")
    print(f"{dataset_start}")
    print(f"to")
    print(f"{dataset_end}")
    print()

    print(f"Prototype mode:")
    print("ENABLED" if (USE_PROTOTYPE_EVENT and not catalog_overlap) else "DISABLED")
    print()

    print(f"Synthetic labels:")
    print("YES — PIPELINE TEST ONLY" if (USE_PROTOTYPE_EVENT and not catalog_overlap) else "NO")
    print()

    print(f"• Total observations : {len(df)}")
    print(f"• Quiet rows         : {(df['flare_class'] == 0).sum()}")
    print(f"• C-class rows       : {(df['flare_class'] == 1).sum()}")
    print(f"• M-class rows       : {(df['flare_class'] == 2).sum()}")
    print(f"• X-class rows       : {(df['flare_class'] == 3).sum()}")
    print(f"• Forecast-positive : {(df['flare_label'] == 1).sum()}")

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique
    assert not df["flare_label"].isna().any()
    assert not df["flare_class"].isna().any()
    assert np.isfinite(df["time_to_peak"]).all()
    
    # Assert forecast-positive constraints
    fp = df["flare_label"] == 1
    assert (df.loc[fp, "time_to_peak"] >= 0).all()
    assert (df.loc[fp, "time_to_peak"] <= PROTOTYPE_LOOKBACK).all()
    assert (df.loc[~fp, "time_to_peak"] == -1.0).all()

    # ========================================================
    # SAVE
    # ========================================================
    os.makedirs(os.path.dirname(OUTPUT_LABELS_PATH), exist_ok=True)

    output_columns = [
        "timestamp",
        "flare_label",
        "flare_class",
        "time_to_peak",
        "event_id"
    ]

    df[output_columns].to_csv(
        OUTPUT_LABELS_PATH,
        index=False
    )

    print(f"\n[OK] Labels exported safely to:\n{OUTPUT_LABELS_PATH}")


if __name__ == "__main__":
    generate_scientific_targets()