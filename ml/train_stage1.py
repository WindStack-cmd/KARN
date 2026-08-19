import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


WINDOWS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\training_windows"
MODELS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\models"
RESULTS_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\results"
MODEL_PATH = os.path.join(MODELS_DIR, "stage1_flare_detector.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "stage1_metrics.json")


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    device: str = "cpu"


class Stage1ConvNet(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> conv expects (B, F, L)
        z = x.transpose(1, 2)
        z = self.conv(z)
        z = self.pool(z).squeeze(-1)
        z = self.dropout(z)
        return self.head(z).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_split_tensors():
    def _load(name: str):
        return torch.load(os.path.join(WINDOWS_DIR, name), map_location="cpu").numpy()

    X_train = _load("X_train.pt").astype(np.float32)
    X_val = _load("X_val.pt").astype(np.float32)
    X_test = _load("X_test.pt").astype(np.float32)

    y_train_cls = _load("y_train_cls.pt").astype(np.int64)
    y_val_cls = _load("y_val_cls.pt").astype(np.int64)
    y_test_cls = _load("y_test_cls.pt").astype(np.int64)

    # Binary target proxy from existing flare target semantics: quiet=0, flare classes >0
    y_train = (y_train_cls > 0).astype(np.float32)
    y_val = (y_val_cls > 0).astype(np.float32)
    y_test = (y_test_cls > 0).astype(np.float32)

    for name, arr in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_val", y_val),
        ("y_test", y_test),
    ]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf values.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = {
        "num_samples": int(len(y_true)),
        "num_negative": int((y_true == 0).sum()),
        "num_positive": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        metrics["roc_auc"] = "undefined because only one class is present"
        metrics["pr_auc"] = "undefined because only one class is present"
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return metrics


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    losses = []
    all_y = []
    all_p = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            losses.append(loss.item())
            all_y.append(yb.cpu().numpy())
            all_p.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y).astype(np.int64)
    y_prob = np.concatenate(all_p).astype(np.float64)
    return float(np.mean(losses)), compute_binary_metrics(y_true, y_prob)


def train_stage1():
    config = TrainConfig()
    set_seed(config.seed)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_split_tensors()

    device = torch.device(config.device)
    model = Stage1ConvNet(input_features=X_train.shape[-1]).to(device)
    param_count = count_parameters(model)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if pos <= 0:
        raise ValueError("Training set has zero positive samples; cannot train Stage-1 detector.")
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_loader = build_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = build_loader(X_val, y_val, config.batch_size, shuffle=False)
    test_loader = build_loader(X_test, y_test, config.batch_size, shuffle=False)

    print("=== DAY 11 PHASE 10 STAGE-1 TRAINING ===")
    print(f"Device: {device}")
    print(f"Dataset sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Class distribution (train): negatives={int(neg)}, positives={int(pos)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epoch limit: {config.max_epochs}")
    print(f"Model parameters: {param_count}")

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    history = []
    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_metrics["f1_score"],
            }
        )
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_f1={val_metrics['f1_score']:.4f}")

        if (best_val_loss - val_loss) > config.early_stopping_min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "input_features": int(X_train.shape[-1]),
                "parameter_count": int(param_count),
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "train_class_distribution": {"negative": int(neg), "positive": int(pos)},
            }
            torch.save(checkpoint, MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    elapsed = time.time() - start

    saved = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(saved["model_state_dict"])

    train_loss, train_metrics = evaluate(model, train_loader, device, criterion)
    val_loss, val_metrics = evaluate(model, val_loader, device, criterion)
    test_loss, test_metrics = evaluate(model, test_loader, device, criterion)

    report = {
        "evaluation_status": "PROTOTYPE_ONLY",
        "scientific_validity": "NOT_VALID_FOR_REAL_SOLAR_FLARE_PERFORMANCE",
        "warning": "Synthetic/prototype labels are for software pipeline validation only.",
        "model_name": "Stage1ConvNet",
        "checkpoint_path": MODEL_PATH,
        "device": str(device),
        "parameter_count": int(param_count),
        "training_config": config.__dict__,
        "training_runtime_seconds": float(elapsed),
        "best_epoch": int(saved["best_epoch"]),
        "best_val_loss": float(saved["best_val_loss"]),
        "splits": {
            "train": {"loss": train_loss, **train_metrics},
            "validation": {"loss": val_loss, **val_metrics},
            "test": {"loss": test_loss, **test_metrics},
        },
        "history": history,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Saved Stage-1 checkpoint: {MODEL_PATH}")
    print(f"[OK] Saved Stage-1 metrics: {METRICS_PATH}")


if __name__ == "__main__":
    train_stage1()
