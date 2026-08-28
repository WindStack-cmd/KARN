"""KARN training using only persisted real tensors."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from models_v3 import KarnMultiTaskEngine

ROOT = Path(__file__).resolve().parents[1]
TENSORS = ROOT / "data" / "processed" / "training_windows"
CHECKPOINT = ROOT / "ml" / "checkpoints" / "karn_real_weights_current.pt"

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
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "history": history, "dataset_validation": json.loads((TENSORS / "validation_report.json").read_text()), "input_dim": train[0].shape[2], "num_classes": 4}, CHECKPOINT)
    print(f"REAL TRAINING COMPLETE\nCheckpoint saved: {CHECKPOINT}"); return CHECKPOINT

if __name__ == "__main__":
    run_ml_pipeline(smoke_only=True)
    run_ml_pipeline(smoke_only=False)
