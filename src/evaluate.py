import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels):
    """Returns dict with accuracy, precision, recall, F1 (macro average)."""
    return {
        "accuracy":  round(accuracy_score(labels, preds) * 100, 2),
        "precision": round(precision_score(labels, preds, average="macro", zero_division=0) * 100, 2),
        "recall":    round(recall_score(labels, preds, average="macro", zero_division=0) * 100, 2),
        "f1":        round(f1_score(labels, preds, average="macro", zero_division=0) * 100, 2),
    }


def plot_confusion_matrix(preds, labels, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def evaluate_checkpoint(model, checkpoint_path, test_loader, class_names, device, figures_dir, experiment_name):
    """Load best checkpoint and compute full test metrics."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    preds, labels = get_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, labels)

    print(f"\n=== {experiment_name} — Test Results ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.2f}%")

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    cm_path = f"{figures_dir}/{experiment_name}_confusion_matrix.png"
    plot_confusion_matrix(preds, labels, class_names, cm_path)

    return metrics
