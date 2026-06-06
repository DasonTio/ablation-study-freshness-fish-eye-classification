"""
Generate paper-ready figures for the flat 24-class Swin-Tiny baseline experiment.

Produces:
  - 24-class confusion matrix (flat model)
  - Grad-CAM per freshness level (Highly Fresh / Fresh / Not Fresh)
  - Updated benchmark comparison bar chart (all models: benchmark → SOTA → flat → ordinal → hybrid)

Designed to run on the same Vast.ai instance immediately after training completes.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from src.dataset import load_config
from src.hierarchical import (
    HierarchicalClassifier,
    RANK_TO_FRESHNESS,
    get_hierarchical_dataloaders,
)
from src.recipe import IMAGENET_MEAN, IMAGENET_STD


BACKBONE_TAG = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
CHECKPOINT = "results/checkpoints/flat_swin_tiny_flat_BEST.pth"
RESULTS_CSV = "results/flat_swin_baseline_results.csv"
OUT_DIR = Path("results/figures/flat_baseline")

# Fixed results from prior completed experiments (do not change)
PRASETYO_ACC = 78.82
HOANG_ACC = 85.99
HIER_NEURAL_MEAN = 88.46
HIER_NEURAL_STD = 0.97
HIER_ET_BEST = 89.98


class ClassHeadWrapper(torch.nn.Module):
    """Expose only the 24-class logits so Grad-CAM targets the classification head."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["class"]


def swin_reshape_transform(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        size = int(tokens ** 0.5)
        return tensor.reshape(batch, size, size, channels).permute(0, 3, 1, 2)
    return tensor


def load_model(metadata, device):
    model = HierarchicalClassifier(
        backbone_tag=BACKBONE_TAG,
        num_classes=metadata.num_classes,
        num_species=metadata.num_species,
        pretrained=False,
        dropout=0.3,
        drop_path=0.1,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    class_probs, class_preds, class_labels = [], [], []
    fresh_labels_all = []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        probs = outputs["class"].softmax(1)
        class_probs.append(probs.cpu().numpy())
        class_preds.append(probs.argmax(1).cpu().numpy())
        class_labels.append(targets["class"].numpy())
        fresh_labels_all.append(targets["freshness"].numpy())
    return {
        "class_probs": np.concatenate(class_probs),
        "class_preds": np.concatenate(class_preds),
        "class_labels": np.concatenate(class_labels),
        "fresh_labels": np.concatenate(fresh_labels_all),
    }


def plot_class_confusion(preds, metadata):
    cm = confusion_matrix(preds["class_labels"], preds["class_preds"], labels=np.arange(metadata.num_classes))
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm,
        cmap="Oranges",
        xticklabels=metadata.class_names,
        yticklabels=metadata.class_names,
        cbar_kws={"label": "Images"},
        ax=ax,
    )
    acc = (preds["class_preds"] == preds["class_labels"]).mean() * 100
    ax.set_title(f"Flat Swin-Tiny 24-Class Confusion Matrix  (test acc {acc:.2f}%)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flat_baseline_confusion_matrix.png", dpi=220)
    plt.close(fig)
    print(f"  Confusion matrix saved (test acc {acc:.2f}%)")


def plot_benchmark_comparison():
    """Bar chart covering full model progression — key journal figure."""
    flat_df = pd.read_csv(RESULTS_CSV)
    flat_mean = flat_df["test_class_acc"].mean()
    flat_std = flat_df["test_class_acc"].std(ddof=1) if len(flat_df) > 1 else 0.0

    labels = [
        "ResNet50\n[Prasetyo 2022]",
        "Swin-T + ET + LGBM\n[Hoang 2025 SOTA]",
        f"Flat Swin-Tiny\n[ours, {flat_mean:.2f}±{flat_std:.2f}%]",
        f"Hierarchical Ordinal\n[ours, {HIER_NEURAL_MEAN:.2f}±{HIER_NEURAL_STD:.2f}%]",
        f"Hierarchical Ordinal + ET\n[ours, {HIER_ET_BEST:.2f}%]",
    ]
    values = [PRASETYO_ACC, HOANG_ACC, flat_mean, HIER_NEURAL_MEAN, HIER_ET_BEST]
    # Error bars: std for our multi-seed results only
    xerr = [0, 0, flat_std, HIER_NEURAL_STD, 0]
    colors = ["#7a8793", "#9eaab5", "#c45c2a", "#2166ac", "#1b7837"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, xerr=xerr, color=colors, capsize=4, height=0.55)
    ax.set_xlim(70, 93)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("FFE Fish-Eye Freshness Classification — Full Model Comparison", fontsize=11, pad=10)
    for bar, val in zip(bars, values):
        ax.text(val + 0.25, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9)
    ax.axvline(HOANG_ACC, color="#9eaab5", linestyle="--", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flat_vs_hierarchical_benchmark_comparison.png", dpi=220)
    plt.close(fig)
    print(f"  Benchmark comparison chart saved (flat mean {flat_mean:.2f}%)")

    # Also write CSV for the paper comparison table
    table_rows = [
        {"Model": "ResNet50 [Prasetyo 2022, benchmark]", "Accuracy (%)": f"{PRASETYO_ACC:.2f}"},
        {"Model": "Swin-T + ExtraTrees + LGBM [Hoang 2025, SOTA]", "Accuracy (%)": f"{HOANG_ACC:.2f}"},
        {"Model": "Flat Swin-Tiny 24-class CE [ours]", "Accuracy (%)": f"{flat_mean:.2f} ± {flat_std:.2f}"},
        {"Model": "Hierarchical Ordinal Swin-Tiny [ours, neural]", "Accuracy (%)": f"{HIER_NEURAL_MEAN:.2f} ± {HIER_NEURAL_STD:.2f}"},
        {"Model": "Hierarchical Ordinal + ExtraTrees [ours, best]", "Accuracy (%)": f"{HIER_ET_BEST:.2f}"},
    ]
    pd.DataFrame(table_rows).to_csv(OUT_DIR / "full_comparison_table.csv", index=False)


def preprocess_for_gradcam(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    rgb = np.asarray(image).astype(np.float32) / 255.0
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tensor = normalize(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    return tensor, rgb


def select_gradcam_samples(preds, samples):
    """Pick highest-confidence correctly-classified sample per freshness level."""
    selected = {}
    confidence = preds["class_probs"].max(axis=1)
    order = np.argsort(-confidence)
    for idx in order:
        fresh = int(preds["fresh_labels"][idx])
        if fresh in selected:
            continue
        if preds["class_labels"][idx] == preds["class_preds"][idx]:
            selected[fresh] = idx
        if len(selected) == 3:
            break
    # fallback: any sample for missing freshness levels
    for idx in order:
        fresh = int(preds["fresh_labels"][idx])
        selected.setdefault(fresh, idx)
        if len(selected) == 3:
            break
    return [selected[rank] for rank in sorted(selected)]


def generate_gradcam_figures(model, preds, samples, metadata, device, image_size):
    wrapped = ClassHeadWrapper(model).to(device).eval()
    target_layers = [wrapped.model.backbone.layers[-1].blocks[-1].norm2]
    sample_indices = select_gradcam_samples(preds, samples)
    panel_images, titles = [], []

    for idx in sample_indices:
        sample = samples[idx]
        target_class = int(preds["class_preds"][idx])
        tensor, rgb = preprocess_for_gradcam(sample.path, image_size)
        with GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=swin_reshape_transform) as cam:
            mask = cam(input_tensor=tensor.to(device), targets=[ClassifierOutputTarget(target_class)])[0]
        overlay = show_cam_on_image(rgb, mask, use_rgb=True)
        fresh = RANK_TO_FRESHNESS[int(preds["fresh_labels"][idx])]
        conf = float(preds["class_probs"][idx, target_class])
        panel_images.append((rgb, overlay))
        titles.append(f"{fresh}  (conf {conf:.2f})\n{metadata.class_names[target_class]}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM (flat CE)")
        axes[1].axis("off")
        fig.suptitle(f"Flat 24-class Swin-Tiny — {fresh}", fontsize=10)
        fig.tight_layout()
        safe = fresh.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"gradcam_{safe}.png", dpi=220)
        plt.close(fig)
        print(f"  Grad-CAM saved: {fresh}")

    # 3-row panel
    fig, axes = plt.subplots(len(panel_images), 2, figsize=(8, 4 * len(panel_images)))
    if len(panel_images) == 1:
        axes = np.asarray([axes])
    for row, ((rgb, overlay), title) in enumerate(zip(panel_images, titles)):
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{title}\nOriginal", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title("Grad-CAM (flat CE)", fontsize=8)
        axes[row, 1].axis("off")
    fig.suptitle("Flat 24-class Swin-Tiny — Grad-CAM per Freshness Level", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flat_baseline_gradcam_panel.png", dpi=220)
    plt.close(fig)
    print("  Grad-CAM panel saved.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader, metadata = get_hierarchical_dataloaders(
        cfg,
        split_seed=42,
        batch_size=64,
        num_workers=4,
    )
    model = load_model(metadata, device)
    print("Model loaded.")

    preds = collect_predictions(model, test_loader, device)
    samples = test_loader.dataset.samples

    print("Generating figures...")
    plot_class_confusion(preds, metadata)
    plot_benchmark_comparison()
    generate_gradcam_figures(model, preds, samples, metadata, device, cfg["dataset"]["image_size"])

    print(f"\nAll flat baseline figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
