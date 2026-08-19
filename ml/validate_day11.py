import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_stage1 import Stage1ConvNet
from train_fusion_stage import FusionStage1Model


ROOT = Path(r"C:\Users\DELL\OneDrive\Desktop\KARN")
WINDOWS_DIR = ROOT / "data" / "processed" / "training_windows"
MODELS_DIR = ROOT / "models"
FEATURES_PATH = ROOT / "data" / "processed" / "aditya_features.csv"
LABELS_PATH = ROOT / "data" / "processed" / "aditya_labels.csv"
RAW_DIR = ROOT / "data" / "raw"


def check(condition: bool, ok_msg: str, fail_msg: str):
    if condition:
        print(f"[PASS] {ok_msg}")
    else:
        print(f"[FAIL] {fail_msg}")
    return condition


def load_tensors():
    files = [
        "X_train.pt",
        "X_val.pt",
        "X_test.pt",
        "y_train_cls.pt",
        "y_val_cls.pt",
        "y_test_cls.pt",
    ]
    arrays = {}
    for f in files:
        p = WINDOWS_DIR / f
        if not p.exists():
            raise FileNotFoundError(f"Missing tensor: {p}")
        arrays[f] = torch.load(p, map_location="cpu").numpy()
    return arrays


def validate_stage1_inference(X_sample: torch.Tensor):
    ckpt_path = MODELS_DIR / "stage1_flare_detector.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = Stage1ConvNet(input_features=int(ckpt["input_features"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    logits = model(X_sample)
    probs = torch.sigmoid(logits)
    assert logits.shape == (X_sample.shape[0],), "Stage1 logits shape mismatch."
    assert probs.shape == (X_sample.shape[0],), "Stage1 prob shape mismatch."
    assert torch.isfinite(logits).all() and torch.isfinite(probs).all(), "Stage1 output contains NaN/Inf."
    assert (probs >= 0).all() and (probs <= 1).all(), "Stage1 probabilities are outside [0,1]."

    y_dummy = torch.randint(0, 2, (X_sample.shape[0],), dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y_dummy)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise ValueError(f"Stage1 gradient invalid at parameter {name}.")


def validate_fusion_inference(X_sample: torch.Tensor):
    ckpt_path = MODELS_DIR / "fusion_stage1.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = FusionStage1Model(
        solexs_indices=ckpt["solexs_indices"],
        hel1os_indices=ckpt["hel1os_indices"],
        context_indices=ckpt["context_indices"],
        projection_dim=ckpt["config"]["projection_dim"],
        temporal_hidden_dim=ckpt["config"]["temporal_hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    logits, w_sol, w_hel = model(X_sample)
    probs = torch.sigmoid(logits)
    assert logits.shape == (X_sample.shape[0],), "Fusion logits shape mismatch."
    assert probs.shape == (X_sample.shape[0],), "Fusion prob shape mismatch."
    assert w_sol.shape == (X_sample.shape[0], X_sample.shape[1], 1), "Fusion SoLEXS weight shape mismatch."
    assert w_hel.shape == (X_sample.shape[0], X_sample.shape[1], 1), "Fusion HEL1OS weight shape mismatch."
    assert torch.isfinite(logits).all() and torch.isfinite(probs).all(), "Fusion output contains NaN/Inf."
    assert torch.isfinite(w_sol).all() and torch.isfinite(w_hel).all(), "Fusion weights contain NaN/Inf."
    assert (w_sol >= 0).all() and (w_sol <= 1).all(), "Fusion SoLEXS weights out of [0,1]."
    assert (w_hel >= 0).all() and (w_hel <= 1).all(), "Fusion HEL1OS weights out of [0,1]."
    assert torch.allclose(w_sol + w_hel, torch.ones_like(w_sol), atol=1e-6), "Fusion weights do not sum to 1."

    y_dummy = torch.randint(0, 2, (X_sample.shape[0],), dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y_dummy)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise ValueError(f"Fusion gradient invalid at parameter {name}.")


def validate_chronology():
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], errors="coerce")

    if features["timestamp"].isna().any() or labels["timestamp"].isna().any():
        return False
    if features["timestamp"].duplicated().any() or labels["timestamp"].duplicated().any():
        return False
    if not features["timestamp"].is_monotonic_increasing:
        return False
    if not labels["timestamp"].is_monotonic_increasing:
        return False
    return True


def main():
    print("=== KARN DAY 11 VALIDATION ===")
    all_ok = True

    tensors = load_tensors()
    x_train = tensors["X_train.pt"]
    x_val = tensors["X_val.pt"]
    x_test = tensors["X_test.pt"]

    all_ok &= check(x_train.ndim == 3 and x_val.ndim == 3 and x_test.ndim == 3, "Tensor rank check passed", "Tensor rank check failed")
    all_ok &= check(
        x_train.shape[1:] == x_val.shape[1:] == x_test.shape[1:],
        f"Tensor feature/sequence shape consistent: {x_train.shape[1:]}",
        "Tensor feature/sequence shape mismatch across splits",
    )

    finite_ok = True
    for k, arr in tensors.items():
        if not np.isfinite(arr).all():
            finite_ok = False
            print(f"[FAIL] NaN/Inf in {k}")
    all_ok &= check(finite_ok, "No NaN/Inf in loaded tensors", "NaN/Inf detected in tensors")

    all_ok &= check(validate_chronology(), "Feature/label chronology checks passed", "Chronology validation failed")

    raw_fits_before = {p.name: p.stat().st_mtime for p in RAW_DIR.glob("*.fits")}
    sample = torch.from_numpy(x_train[:8].astype(np.float32))

    validate_stage1_inference(sample)
    print("[PASS] Stage-1 checkpoint reload + inference + gradient check")

    validate_fusion_inference(sample)
    print("[PASS] Fusion checkpoint reload + inference + weight constraints + gradient check")

    raw_fits_after = {p.name: p.stat().st_mtime for p in RAW_DIR.glob("*.fits")}
    all_ok &= check(raw_fits_before == raw_fits_after, "Raw FITS timestamps unchanged during validation", "Raw FITS metadata changed unexpectedly")

    # Exclude virtualenv and node_modules from large-file scan
    unexpected_large = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        p_str = str(p).lower()
        if "\\.venv\\" in p_str or "\\ml\\venv\\" in p_str or "\\node_modules\\" in p_str:
            continue
        if p.stat().st_size > 200 * 1024 * 1024:
            unexpected_large.append((str(p), p.stat().st_size))

    all_ok &= check(
        len(unexpected_large) == 0,
        "No unexpected >200MB generated files outside env folders",
        f"Found unexpected large files: {unexpected_large}",
    )

    if all_ok:
        print("DAY 11 VALIDATION: PASS")
    else:
        print("DAY 11 VALIDATION: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
