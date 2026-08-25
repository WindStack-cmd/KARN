import os
import torch
import numpy as np
import pickle
import pandas as pd
from sensor_fusion_gate import ContextAwareFusionGate, detect_sensor_features
from train_baselines import compute_sequence_stats, compute_flattened

# ============================================================
# KARN - PHASE 10 Automated Validation Suite
# ============================================================

WINDOWS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\training_windows"
MODELS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\models"
RESULTS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\results"
FEATURES_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
LABELS_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_labels.csv"

def test_dataset_integrity():
    """
    Verifies that generated tensors exist, have valid window dimensions, contain no NaNs/Infs,
    and remain consistent with the current processed label/feature files without hard-coding
    legacy synthetic class counts.
    """
    print("\n[Test 1] Verifying Dataset Integrity...")

    files = [
        "X_train.pt", "X_val.pt", "X_test.pt",
        "y_train_cls.pt", "y_val_cls.pt", "y_test_cls.pt",
        "y_train_reg.pt", "y_val_reg.pt", "y_test_reg.pt"
    ]

    tensors = {}
    for f in files:
        path = os.path.join(WINDOWS_DIR, f)
        if not os.path.exists(path):
            print(f"  [-] FAIL: Missing tensor file {f} at {path}")
            return False
        tensors[f] = torch.load(path, map_location="cpu").numpy()

    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)

    feature_cols = [c for c in features.columns if c not in ["timestamp", "time_seconds"]]
    expected_feature_dim = len(feature_cols)
    expected_window_shape = (30, expected_feature_dim)

    for split in ["train", "val", "test"]:
        x = tensors[f"X_{split}.pt"]
        y_cls = tensors[f"y_{split}_cls.pt"]
        y_reg = tensors[f"y_{split}_reg.pt"]

        if x.ndim != 3:
            print(f"  [-] FAIL: X_{split}.pt is not rank-3: {x.shape}")
            return False
        if x.shape[1:] != expected_window_shape:
            print(f"  [-] FAIL: X_{split}.pt shape {x.shape} does not match expected (N, 30, {expected_feature_dim})")
            return False
        if x.shape[0] != y_cls.shape[0] or x.shape[0] != y_reg.shape[0]:
            print(f"  [-] FAIL: Length mismatch for {split} (X={x.shape[0]}, y_cls={y_cls.shape[0]}, y_reg={y_reg.shape[0]})")
            return False

    for name, arr in tensors.items():
        if not np.isfinite(arr).all():
            print(f"  [-] FAIL: Tensor {name} contains NaNs or Infs!")
            return False

    for split in ["train", "val", "test"]:
        labels_arr = tensors[f"y_{split}_cls.pt"]
        unique, counts = np.unique(labels_arr, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  {split} actual class distribution: {dist}")
        if split in ["val", "test"]:
            positive_count = dist.get(1, 0) + dist.get(2, 0) + dist.get(3, 0)
            if positive_count == 0:
                print(f"  [WARN] {split} contains no positive flare labels; scientific flare metrics are undefined for this split.")

    features_ts = pd.to_datetime(features["timestamp"], errors="coerce")
    labels_ts = pd.to_datetime(labels["timestamp"], errors="coerce")
    if features_ts.isna().any() or labels_ts.isna().any():
        print("  [-] FAIL: Feature or label timestamps contain NaT values.")
        return False
    if features_ts.duplicated().any() or labels_ts.duplicated().any():
        print("  [-] FAIL: Duplicate timestamps detected in feature or label files.")
        return False
    if not features_ts.is_monotonic_increasing or not labels_ts.is_monotonic_increasing:
        print("  [-] FAIL: Features or labels are not strictly chronological.")
        return False

    print(f"  [+] Tensors dimension and integrity check: PASS for current dataset (window shape = {expected_window_shape})")
    return True

def test_baseline_models():
    """
    Verifies that the serialized baseline models exist, can be reloaded,
    produce finite predictions, and have appropriate shapes.
    """
    print("\n[Test 2] Verifying Baseline Models...")
    
    # Load train/val/test data
    X_train = torch.load(os.path.join(WINDOWS_DIR, "X_train.pt"), map_location="cpu").numpy()
    X_val = torch.load(os.path.join(WINDOWS_DIR, "X_val.pt"), map_location="cpu").numpy()
    X_test = torch.load(os.path.join(WINDOWS_DIR, "X_test.pt"), map_location="cpu").numpy()
    
    # Compute both representations for full split coverage
    representations = {
        "statistical": {
            "train": compute_sequence_stats(X_train),
            "val": compute_sequence_stats(X_val),
            "test": compute_sequence_stats(X_test)
        },
        "flattened": {
            "train": compute_flattened(X_train),
            "val": compute_flattened(X_val),
            "test": compute_flattened(X_test)
        }
    }

    for name in ["logistic_regression", "random_forest"]:
        for rep_name in ["statistical", "flattened"]:
            path = os.path.join(MODELS_DIR, f"baseline_{name}_{rep_name}.pkl")
            if not os.path.exists(path):
                if rep_name == "statistical":
                    fallback = os.path.join(MODELS_DIR, f"baseline_{name}.pkl")
                    if os.path.exists(fallback):
                        path = fallback
                    else:
                        print(f"  [-] FAIL: Missing model file for {name}/{rep_name} at {path} or {fallback}")
                        return False
                else:
                    print(f"  [-] FAIL: Missing model file for {name}/{rep_name} at {path}")
                    return False

            with open(path, "rb") as f:
                model = pickle.load(f)

            print(f"  Checking saved {name} model ({rep_name})...")

            if name == "logistic_regression":
                random_state = model.named_steps["classifier"].random_state
            else:
                random_state = model.named_steps["classifier"].random_state
            if random_state != 42:
                print(f"  [-] FAIL: {name} random_state is {random_state}, expected 42")
                return False

            train_rep = representations[rep_name]["train"]
            val_rep = representations[rep_name]["val"]
            test_rep = representations[rep_name]["test"]

            preds_train = model.predict(train_rep)
            preds_val = model.predict(val_rep)
            preds_test = model.predict(test_rep)

            if (
                preds_train.shape != (len(train_rep),)
                or preds_val.shape != (len(val_rep),)
                or preds_test.shape != (len(test_rep),)
            ):
                print(f"  [-] FAIL: {name}/{rep_name} prediction shape mismatch!")
                return False

            prob_train = model.predict_proba(train_rep)
            prob_val = model.predict_proba(val_rep)
            prob_test = model.predict_proba(test_rep)

            if (
                prob_train.shape != (len(train_rep), 2)
                or prob_val.shape != (len(val_rep), 2)
                or prob_test.shape != (len(test_rep), 2)
            ):
                print(f"  [-] FAIL: {name}/{rep_name} probability shape mismatch!")
                return False

            if not (
                np.isfinite(prob_train).all()
                and np.isfinite(prob_val).all()
                and np.isfinite(prob_test).all()
            ):
                print(f"  [-] FAIL: {name}/{rep_name} probabilities contain NaNs or Infs!")
                return False

            print(f"    - Reload, predict, and probability checks: PASS")
        
    print("  [+] Baseline models reload and execution check: PASS")
    return True

def test_sensor_fusion_gate():
    """
    Verifies that the fusion gate works correctly under normal and disturbed conditions.
    Checks shape consistency, sensor mapping, bounds, and no NaN/Inf values.
    """
    print("\n[Test 3] Verifying Sensor Fusion Gate...")
    
    # 1. correct sensor mapping
    try:
        feature_cols, solexs_cols, hel1os_cols, context_cols = detect_sensor_features()
    except Exception as e:
        print(f"  [-] FAIL: Feature detection threw exception: {e}")
        return False
        
    # Check counts
    if len(solexs_cols) != 10 or len(hel1os_cols) != 10 or len(context_cols) != 3:
        print(f"  [-] FAIL: Features split mismatch! SoLEXS={len(solexs_cols)}, HEL1OS={len(hel1os_cols)}, Context={len(context_cols)}")
        return False
        
    # 2. Gate forward passes and outputs check
    torch.manual_seed(42)  # For reproducibility
    gate = ContextAwareFusionGate(
        solexs_dim=len(solexs_cols),
        hel1os_dim=len(hel1os_cols),
        context_dim=len(context_cols),
        projection_dim=16
    )
    gate.eval()
    
    B, L = 4, 30
    x_sol = torch.randn(B, L, 10, requires_grad=True)
    x_hel = torch.randn(B, L, 10, requires_grad=True)
    x_ctx = torch.randn(B, L, 3, requires_grad=True)
    
    try:
        # Backward propagation check
        gate.train()
        fused, w_sol, w_hel = gate(x_sol, x_hel, x_ctx)
        loss = fused.sum()
        loss.backward()
        
        # Verify gradients exist and are finite
        for name, param in gate.named_parameters():
            if param.grad is None:
                print(f"  [-] FAIL: Parameter {name} has no gradients (backpropagation broken)!")
                return False
            if not torch.isfinite(param.grad).all():
                print(f"  [-] FAIL: Parameter {name} gradient contains NaNs or Infs!")
                return False
                
        # Forward pass evaluations check
        gate.eval()
        with torch.no_grad():
            fused, w_sol, w_hel = gate(x_sol, x_hel, x_ctx)
    except Exception as e:
        print(f"  [-] FAIL: Forward or backward pass raised exception: {e}")
        return False
        
    # Check outputs shape
    if fused.shape != (B, L, 16):
        print(f"  [-] FAIL: Fused output shape {fused.shape} != (4, 30, 16)")
        return False
    if w_sol.shape != (B, L, 1) or w_hel.shape != (B, L, 1):
        print(f"  [-] FAIL: Weights shape mismatch!")
        return False
        
    # Check no NaN / Inf
    if not (torch.isfinite(fused).all() and torch.isfinite(w_sol).all() and torch.isfinite(w_hel).all()):
        print(f"  [-] FAIL: Fusion outputs contain NaNs or Infs!")
        return False
        
    # Check bounds
    if (w_sol < 0.0).any() or (w_sol > 1.0).any() or (w_hel < 0.0).any() or (w_hel > 1.0).any():
        print(f"  [-] FAIL: Weighted gate outputs out of bounds [0, 1]!")
        return False
        
    # Check sum-to-1 constraint
    if not torch.allclose(w_sol + w_hel, torch.ones_like(w_sol)):
        print(f"  [-] FAIL: Sensor weights do not sum to 1!")
        return False

    # Check explicit input dimension validation
    try:
        gate(x_sol, x_hel[:, :, :9], x_ctx)
        print("  [-] FAIL: Input dimension mismatch did not raise an error!")
        return False
    except ValueError:
        pass
         
    print("  [+] Sensor Fusion Gate sanity and constraints check: PASS")
    return True

def run_all_validation():
    print("==================================================")
    print("KARN DAY 10 SYSTEM VALIDATION RUN")
    print("==================================================")
    
    step1 = test_dataset_integrity()
    step2 = test_baseline_models()
    step3 = test_sensor_fusion_gate()
    
    print("\n==================================================")
    if step1 and step2 and step3:
        print("ALL DAY 10 CORE PIPELINE VALIDATION TESTS: SUCCESS (PASS)")
        print("==================================================")
        return True
    else:
        print("DAY 10 CORE PIPELINE VALIDATION TESTS: FAILED (FAIL)")
        print("==================================================")
        return False

if __name__ == "__main__":
    success = run_all_validation()
    exit(0 if success else 1)
