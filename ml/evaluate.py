import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_v2 import SolarFlareDatasetV2
from train_v2 import FlareRecurrentLSTM

def run_evaluation():
    base_dir  = r"C:\Users\DELL\OneDrive\Desktop\KARN\data"
    catalog_path = os.path.join(base_dir, "processed", "full_catalog.csv")
    flux_path    = os.path.join(base_dir, "raw",       "goes_flux_combined.csv")
    model_path = "karn_lstm_v2.pth"
    PROC_DIR     = os.path.join(base_dir, "processed")

    print("Loading dataset...")
    dataset = SolarFlareDatasetV2(catalog_path, flux_path)
    print(f"Total samples: {len(dataset)}")

    # Time-based split — NEVER shuffle
    split     = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split))
    val_set   = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Load model
    model = FlareRecurrentLSTM()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Run inference
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=-1)
            preds   = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.squeeze().long().numpy())
            all_probs.extend(probs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # Class names
    class_names = ['Quiet', 'C-class', 'M-class', 'X-class']

    print("\n" + "=" * 60)
    print("KARN LSTM — EVALUATION RESULTS")
    print("=" * 60)

    # Classification report
    print("\nClassification Report:")
    labels_in_val = sorted(list(set(all_labels.tolist() + all_preds.tolist())))
    names_in_val  = [class_names[i] for i in labels_in_val]

    print(classification_report(
        all_labels, all_preds,
        labels=labels_in_val,
        target_names=names_in_val,
        zero_division=0
    ))

    # Key metrics
    from sklearn.metrics import recall_score, precision_score, f1_score
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Macro Recall:    {recall:.3f}")
    print(f"Macro Precision: {precision:.3f}")
    print(f"Macro F1:        {f1:.3f}")

    # Most important — flare detection recall
    labels_present = np.unique(np.concatenate([all_labels, all_preds]))
    print("\n--- FLARE DETECTION SUMMARY ---")
    for i, name in enumerate(class_names):
        if i not in labels_present:
            continue
        mask         = all_labels == i
        correct      = np.sum(all_preds[mask] == i)
        total        = np.sum(mask)
        class_recall = correct / total if total > 0 else 0
        print(f"{name:10s}: {correct}/{total} detected | Recall: {class_recall:.2%}")

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    present_names = [class_names[i] for i in labels_present]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0A0A14')
    ax.set_facecolor('#0F0F1E')

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title('KARN LSTM — Confusion Matrix',
                 color='#FFB347', fontsize=13, pad=12)
    ax.set_xlabel('Predicted', color='#888')
    ax.set_ylabel('Actual',    color='#888')
    ax.tick_params(colors='#888')

    plt.tight_layout()
    cm_path = os.path.join(PROC_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150,
                bbox_inches='tight', facecolor='#0A0A14')
    print(f"\nConfusion matrix saved: {cm_path}")
    plt.show()

    # Probability distribution plot
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0A0A14')

    for ax in axes:
        ax.set_facecolor('#0F0F1E')
        ax.tick_params(colors='#666')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # M-class probability distribution
    m_probs = all_probs[:, 2]
    axes[0].hist(m_probs[all_labels == 0], bins=20,
                 alpha=0.7, color='#4A9EFF', label='Quiet windows')
    axes[0].hist(m_probs[all_labels >= 2], bins=20,
                 alpha=0.7, color='#FF8C00', label='Flare windows')
    axes[0].set_title('M-class Probability Distribution',
                      color='#FFB347', fontsize=11)
    axes[0].set_xlabel('P(M-class)', color='#888')
    axes[0].set_ylabel('Count',      color='#888')
    axes[0].legend(facecolor='#1A1A2E', labelcolor='#CCC')
    axes[0].axvline(x=0.5, color='#FF3B3B',
                    linestyle='--', alpha=0.8, label='Threshold')

    # Severity score distribution
    severity_scores = []
    for prob in all_probs:
        score = (1 * prob[1]) + (10 * prob[2]) + (100 * prob[3])
        severity_scores.append(score)

    severity_scores = np.array(severity_scores)
    axes[1].hist(severity_scores[all_labels == 0], bins=20,
                 alpha=0.7, color='#4A9EFF', label='Quiet')
    axes[1].hist(severity_scores[all_labels >= 1], bins=20,
                 alpha=0.7, color='#FF8C00', label='Flare')
    axes[1].set_title('Consequence Severity Score Distribution',
                      color='#FFB347', fontsize=11)
    axes[1].set_xlabel('Severity Score', color='#888')
    axes[1].set_ylabel('Count',          color='#888')
    axes[1].legend(facecolor='#1A1A2E', labelcolor='#CCC')
    axes[1].axvline(x=5.0, color='#FF8C00',
                    linestyle='--', alpha=0.8, label='M threshold')
    axes[1].axvline(x=30.0, color='#FF3B3B',
                    linestyle='--', alpha=0.8, label='X threshold')

    plt.tight_layout()
    dist_path = os.path.join(PROC_DIR, 'probability_distributions.png')
    plt.savefig(dist_path, dpi=150,
                bbox_inches='tight', facecolor='#0A0A14')
    print(f"Distribution plot saved: {dist_path}")
    plt.show()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()