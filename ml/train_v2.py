import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, recall_score
import numpy as np
from dataset_v2 import SolarFlareDatasetV2

RAW_DIR  = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
PROC_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed"

# ── Model ─────────────────────────────────────────────────────────────────────
class FlareRecurrentLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Q, C, M, X
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ── Severity Score ─────────────────────────────────────────────────────────────
def severity_score(probs):
    return (1 * probs[1]) + (10 * probs[2]) + (100 * probs[3])

def alert_status(probs):
    p_x = probs[3]
    p_m = probs[2]
    p_c = probs[1]
    score = severity_score(probs)
    if p_x >= 0.3 or score >= 30.0:
        return "🔴 CRITICAL — X-CLASS IMMINENT"
    elif p_m >= 0.5 or score >= 5.0:
        return "🟡 WARNING  — M-CLASS EXPECTED"
    elif p_c >= 0.5 or score >= 1.0:
        return "🔵 ADVISORY — C-CLASS ACTIVITY"
    else:
        return "🟢 GREEN    — QUIET SUN"


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    catalog_path = os.path.join(PROC_DIR, "full_catalog_v2.csv")
    flux_path    = os.path.join(RAW_DIR,  "goes_flux_combined.csv")

    print("Loading dataset...")
    dataset = SolarFlareDatasetV2(catalog_path, flux_path)
    print(f"Total: {len(dataset)} windows")

    # Time-based split
    split     = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split))
    val_set   = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

    model = FlareRecurrentLSTM()

    # Class weights — penalize missing M and X heavily
    # Q=1.0, C=2.0, M=1.0, X=8.0
    # M gets 1.0 because it's the majority class now
    class_weights = torch.tensor([2.0, 3.0, 1.0, 8.0])
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("\nTraining KARN LSTM v2...")
    print("=" * 65)

    best_val_loss    = float('inf')
    patience_counter = 0
    PATIENCE         = 8
    EPOCHS           = 50

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_set)

        # Validate
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_true  = []

        with torch.no_grad():
            for x, y in val_loader:
                out      = model(x)
                loss     = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds    = torch.argmax(out, dim=1)
                all_preds.extend(preds.numpy())
                all_true.extend(y.numpy())

        val_loss /= len(val_set)
        scheduler.step()

        # Recall per class
        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)

        recalls = {}
        class_names = ['Q', 'C', 'M', 'X']
        for i, name in enumerate(class_names):
            mask = all_true == i
            if mask.sum() > 0:
                recalls[name] = (all_preds[mask] == i).mean()
            else:
                recalls[name] = 0.0

        recall_str = " | ".join([f"{n}:{v:.0%}" for n, v in recalls.items()])

        print(f"Ep {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {recall_str}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "karn_lstm_v2.pth")
            print(f"       ✓ Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\n" + "=" * 65)
    print("Training complete — loading best model for evaluation")
    print("=" * 65)

    # Final evaluation
    model.load_state_dict(torch.load("karn_lstm_v2.pth", weights_only=True))
    model.eval()

    all_preds = []
    all_true  = []

    with torch.no_grad():
        for x, y in val_loader:
            out   = model(x)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.numpy())
            all_true.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    print("\nFinal Classification Report:")
    labels_present = sorted(np.unique(np.concatenate([all_true, all_preds])))
    names_present  = [['Quiet','C-class','M-class','X-class'][i] for i in labels_present]

    print(classification_report(
        all_true, all_preds,
        labels=labels_present,
        target_names=names_present,
        zero_division=0
    ))

    print("\nFlare Detection Summary:")
    class_names_full = ['Quiet', 'C-class', 'M-class', 'X-class']
    for i in labels_present:
        mask    = all_true == i
        correct = (all_preds[mask] == i).sum()
        total   = mask.sum()
        print(f"  {class_names_full[i]:10s}: {correct:3d}/{total:3d} | Recall: {correct/total:.1%}")

    # Live inference demo
    print("\n" + "=" * 65)
    print("LIVE INFERENCE — 5 Sample Windows")
    print("=" * 65)

    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            x, true_label = dataset[i]
            out   = model(x.unsqueeze(0))
            probs = torch.softmax(out, dim=1).squeeze().tolist()
            pred  = int(torch.argmax(out).item())

            true_name = class_names_full[true_label.item()]
            pred_name = class_names_full[pred]
            score     = severity_score(probs)
            status    = alert_status(probs)

            correct = "✓" if pred == true_label.item() else "✗"
            print(f"\nSample {i+1} {correct} | True: {true_name} | Pred: {pred_name}")
            print(f"  Q:{probs[0]:.1%} C:{probs[1]:.1%} M:{probs[2]:.1%} X:{probs[3]:.1%}")
            print(f"  Score: {score:.2f} | {status}")


if __name__ == "__main__":
    train()