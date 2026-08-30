"""KARN training using only persisted real tensors."""
from __future__ import annotations
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from models_v3 import KarnMultiTaskEngine

ROOT = Path(__file__).resolve().parents[1]
TENSORS = ROOT / "data" / "processed" / "training_windows"
CHECKPOINT = ROOT / "ml" / "checkpoints" / "karn_real_weights_current.pt"
BINARY_CHECKPOINT = ROOT / "ml" / "checkpoints" / "karn_binary_flare_current.pt"
TEMPORAL_BINARY_CHECKPOINT = ROOT / "ml" / "checkpoints" / "karn_binary_flare_temporal_experiment.pt"

def load_split(split):
    x = torch.load(TENSORS / f"X_{split}.pt", weights_only=True).float()
    y = torch.load(TENSORS / f"y_{split}_cls.pt", weights_only=True).long()
    r = torch.load(TENSORS / f"y_{split}_reg.pt", weights_only=True).float().reshape(-1, 1)
    if x.ndim != 3 or x.shape[1] != 30 or len(x) != len(y) or len(x) != len(r): raise ValueError(f"Invalid {split} tensor shapes")
    if not torch.isfinite(x).all() or not torch.isfinite(r).all() or not torch.isin(y, torch.arange(4)).all(): raise ValueError(f"Invalid {split} tensor values")
    return x, y, r

def epoch(model, loader, opt, device, class_weights, max_batches=None, update=True):
    ce, mse = nn.CrossEntropyLoss(weight=class_weights), nn.MSELoss(); total = cls_total = reg_total = 0.; batches = 0
    for x, y, r in loader:
        if update: opt.zero_grad(set_to_none=True)
        logits, out = model(x.to(device)); lc, lr = ce(logits, y.to(device)), mse(out, r.to(device)); loss = lc + lr
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
        if update:
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("Non-finite gradients")
            opt.step()
        total += loss.item(); cls_total += lc.item(); reg_total += lr.item(); batches += 1
        if max_batches and batches >= max_batches: break
    return {"loss": total/batches, "classification_loss": cls_total/batches, "regression_loss": reg_total/batches, "batches": batches}

def run_ml_pipeline(smoke_only=False, epochs=3):
    train, val = load_split("train"), load_split("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KarnMultiTaskEngine(train[0].shape[2], 64, 4).to(device); opt = optim.Adam(model.parameters(), lr=1e-3)
    counts = torch.bincount(train[1], minlength=4).float()
    class_weights = torch.zeros(4, device=device)
    present = counts > 0
    class_weights[present] = counts[present].sum() / (present.sum() * counts[present])
    print(f"Training class weights (absent classes remain zero): {class_weights.cpu().tolist()}")
    loaders = [DataLoader(TensorDataset(*d), batch_size=512, shuffle=False) for d in (train, val)]
    history = []
    for n in range(1 if smoke_only else epochs):
        model.train(); tr = epoch(model, loaders[0], opt, device, class_weights, 3 if smoke_only else None)
        model.eval()
        with torch.no_grad(): va = epoch(model, loaders[1], opt, device, class_weights, 3 if smoke_only else None, update=False)
        history.append({"epoch": n + 1, "train": tr, "validation": va}); print(f"Epoch {n+1}: train={tr['loss']:.6f}; validation={va['loss']:.6f}")
    if smoke_only:
        print("SMOKE TEST: Forward pass = PASS; Backward pass = PASS; Loss finite = PASS; Gradients finite = PASS"); return None
    CHECKPOINT.parent.mkdir(exist_ok=True)
    validation = json.loads((TENSORS / "validation_report.json").read_text())
    provenance = {
        "feature_count": int(train[0].shape[2]),
        "feature_names": validation["feature_columns"],
        "label_mapping": {"0": "Quiet", "1": "C", "2": "M", "3": "X"},
        "training_tensor_shape": list(train[0].shape),
        "split_policy": validation["split_policy"],
        "split_ranges": {name: {"start": data["timestamp_start"], "end": data["timestamp_end"]} for name, data in validation["splits"].items()},
        "class_distributions": {name: data["class_distribution"] for name, data in validation["splits"].items()},
        "training_configuration": {"epochs": epochs, "batch_size": 512, "optimizer": "Adam", "learning_rate": 1e-3, "classification_loss": "class-weighted CrossEntropyLoss", "regression_loss": "MSELoss", "shuffle": False},
    }
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "history": history, "dataset_validation": validation, "provenance": provenance, "input_dim": train[0].shape[2], "num_classes": 4}, CHECKPOINT)
    print(f"REAL TRAINING COMPLETE\nCheckpoint saved: {CHECKPOINT}"); return CHECKPOINT

def load_binary_split(split):
    x = torch.load(TENSORS / f"X_{split}_binary.pt", weights_only=True).float()
    y = torch.load(TENSORS / f"y_{split}_flare.pt", weights_only=True).long()
    original = torch.load(TENSORS / f"y_{split}_original_cls_binary.pt", weights_only=True).long()
    if x.ndim != 3 or x.shape[1:] != (30, 21) or len(x) != len(y) or len(y) != len(original): raise ValueError(f"Invalid binary {split} tensors")
    if not torch.isfinite(x).all() or not torch.isin(y, torch.arange(2)).all() or not torch.equal(y, (original >= 2).long()): raise ValueError(f"Binary {split} integrity/traceability audit failed")
    return x, y, original

def binary_epoch(model, loader, optimizer, device, weights, update=True, max_batches=None):
    criterion = nn.CrossEntropyLoss(weight=weights); total = 0.0; batches = 0
    for x, y, _ in loader:
        if update: optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x.to(device)); loss = criterion(logits, y.to(device))
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite binary loss")
        if update:
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("Non-finite binary gradients")
            optimizer.step()
        total += loss.item(); batches += 1
        if max_batches and batches >= max_batches: break
    return {"classification_loss": total / batches, "batches": batches}

def binary_metrics(model, data, device):
    """Metrics over persisted real data; used only for training learning checks."""
    x, y, _ = data; predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            predictions.append(model(x[start:start + 512].to(device))[0].argmax(1).cpu())
    pred = torch.cat(predictions); truth = y.cpu()
    tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum()); tn = int(((pred == 0) & (truth == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {"confusion_matrix": [[tn, fp], [fn, tp]], "accuracy": (tn + tp) / len(truth), "flare_precision": precision, "flare_recall": recall, "flare_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "prediction_distribution": {"0": int((pred == 0).sum()), "1": int((pred == 1).sum())}}

def run_binary_flare_pipeline(smoke_only=False, epochs=3, seed=20260830):
    """Real binary Quiet-vs-(M/X) experiment; regression is intentionally disabled.

    time_to_peak_min is -1 for Quiet and therefore is not a scientifically valid
    regression target for this binary classification experiment.
    """
    torch.manual_seed(seed)
    train, val = load_binary_split("train"), load_binary_split("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KarnMultiTaskEngine(input_dim=21, hidden_dim=64, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    counts = torch.bincount(train[1], minlength=2).float(); weights = counts.sum() / (2 * counts)
    print(f"Binary training class counts: {counts.tolist()}; class weights: {weights.tolist()}")
    val_counts = torch.bincount(val[1], minlength=2).tolist()
    if val_counts[1] == 0: print("VALIDATION POSITIVE COVERAGE: INSUFFICIENT (loss only; no flare validation metric is claimed)")
    loaders = [DataLoader(TensorDataset(*d), batch_size=512, shuffle=False) for d in (train, val)]
    history = []
    for epoch_number in range(1 if smoke_only else epochs):
        model.train(); tr = binary_epoch(model, loaders[0], optimizer, device, weights.to(device), max_batches=3 if smoke_only else None)
        model.eval()
        with torch.no_grad(): va = binary_epoch(model, loaders[1], optimizer, device, weights.to(device), update=False, max_batches=3 if smoke_only else None)
        history.append({"epoch": epoch_number + 1, "train": tr, "validation": va}); print(f"Binary epoch {epoch_number+1}: train_loss={tr['classification_loss']:.6f}; validation_loss={va['classification_loss']:.6f}")
    if smoke_only:
        print("BINARY SMOKE TEST: real forward/backward pass = PASS; finite loss/gradients = PASS"); return None
    validation = json.loads((TENSORS / "binary_validation_report.json").read_text())
    provenance = {"experiment": "binary flare detection", "seed": seed, "feature_count": 21, "feature_names": validation["feature_columns"], "sequence_length": 30, "label_mapping": validation["label_mapping"], "dataset_paths": {split: {"X": str(TENSORS / f"X_{split}_binary.pt"), "y_flare": str(TENSORS / f"y_{split}_flare.pt"), "original_labels": str(TENSORS / f"y_{split}_original_cls_binary.pt")} for split in ("train", "val", "test")}, "split_ranges": {k: {"start": v["timestamp_start"], "end": v["timestamp_end"]} for k,v in validation["splits"].items()}, "class_distributions": {k: v["binary_distribution"] for k,v in validation["splits"].items()}, "training_configuration": {"epochs": epochs, "batch_size": 512, "optimizer": "Adam", "learning_rate": 1e-3, "loss": "class-weighted CrossEntropyLoss", "class_weights": weights.tolist(), "shuffle": False, "regression": "disabled: target undefined for Quiet rows"}, "validation_positive_coverage": int(val_counts[1])}
    BINARY_CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history, "input_dim": 21, "num_classes": 2, "provenance": provenance}, BINARY_CHECKPOINT)
    print(f"BINARY REAL TRAINING COMPLETE\nCheckpoint saved: {BINARY_CHECKPOINT}"); return BINARY_CHECKPOINT

def run_binary_temporal_experiment(epochs=8, seed=20260830):
    """Fixed, reproducible temporal generalization experiment.

    Shuffling is limited to ordering existing July 30 training windows for SGD. It
    does not move any row between splits or use August 1 test data for training,
    model selection, or threshold calibration. The decision rule is argmax.
    """
    torch.manual_seed(seed)
    train, val = load_binary_split("train"), load_binary_split("val")
    counts = torch.bincount(train[1], minlength=2).float()
    weights = counts.sum() / (2 * counts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KarnMultiTaskEngine(input_dim=21, hidden_dim=64, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*train), batch_size=512, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(*val), batch_size=512, shuffle=False)
    positive_batches = sum(int((y == 1).any()) for _, y, _ in loader)
    print(f"TEMPORAL EXPERIMENT: fixed epochs={epochs}, seed={seed}, batch_size=512, argmax decision rule")
    print(f"Training class counts={counts.tolist()}; class weights={weights.tolist()}; batches={len(loader)}; batches containing flare={positive_batches}/{len(loader)} ({100 * positive_batches / len(loader):.1f}%)")
    # Recreate loader after the audit traversal so its seeded epoch order is explicit.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*train), batch_size=512, shuffle=True, generator=generator)
    initial = binary_metrics(model, train, device)
    print(f"Initial train metrics: {initial}")
    history = []
    for epoch_number in range(epochs):
        model.train(); train_loss = binary_epoch(model, loader, optimizer, device, weights.to(device))
        model.eval()
        with torch.no_grad(): validation_loss = binary_epoch(model, val_loader, optimizer, device, weights.to(device), update=False)
        train_metric = binary_metrics(model, train, device)
        record = {"epoch": epoch_number + 1, "train_loss": train_loss, "validation_loss_quiet_only": validation_loss, "train_metrics": train_metric}
        history.append(record)
        print(f"Epoch {epoch_number + 1}/{epochs}: train_loss={train_loss['classification_loss']:.6f}; validation_loss(quiet-only)={validation_loss['classification_loss']:.6f}; train_accuracy={train_metric['accuracy']:.6f}; train_flare_recall={train_metric['flare_recall']:.6f}; train_flare_precision={train_metric['flare_precision']:.6f}; train_flare_f1={train_metric['flare_f1']:.6f}")
    validation = json.loads((TENSORS / "binary_validation_report.json").read_text())
    provenance = {"experiment": "temporal generalization experiment: July 30 Quiet+M training to untouched August 1 Quiet+X test", "trained_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "feature_count": 21, "feature_names": validation["feature_columns"], "sequence_length": 30, "label_mapping": validation["label_mapping"], "training_tensor_shape": list(train[0].shape), "training_class_distribution": {"0": int(counts[0]), "1": int(counts[1])}, "test_period": validation["splits"]["test"], "split_policy": validation["split_policy"], "training_configuration": {"epochs": epochs, "batch_size": 512, "shuffle_within_training_set_only": True, "optimizer": "Adam", "learning_rate": 1e-3, "loss": "class-weighted CrossEntropyLoss", "class_weights": weights.tolist(), "decision_rule": "argmax logits; no threshold tuning", "regression": "disabled: time-to-peak undefined for Quiet"}, "positive_validation_available": False, "model_selection": "fixed configuration; no validation early stopping or test-driven selection"}
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history, "initial_train_metrics": initial, "final_train_metrics": history[-1]["train_metrics"], "input_dim": 21, "num_classes": 2, "provenance": provenance}, TEMPORAL_BINARY_CHECKPOINT)
    print(f"TEMPORAL EXPERIMENT TRAINING COMPLETE\nCheckpoint saved: {TEMPORAL_BINARY_CHECKPOINT}")
    return TEMPORAL_BINARY_CHECKPOINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binary", action="store_true"); parser.add_argument("--temporal-experiment", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--epochs", type=int, default=3); args = parser.parse_args()
    if args.temporal_experiment: run_binary_temporal_experiment(epochs=args.epochs)
    else: (run_binary_flare_pipeline if args.binary else run_ml_pipeline)(smoke_only=args.smoke, epochs=args.epochs)
