"""Generate paper-ready figures for the hierarchical ordinal FFE experiment."""
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
    predict_coral_rank,
)
from src.recipe import IMAGENET_MEAN, IMAGENET_STD


BACKBONE_TAG = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
CHECKPOINT = "results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth"
OUT_DIR = Path("results/figures/hierarchical_ordinal")


class ClassHeadWrapper(torch.nn.Module):
    """Expose only the 24-class logits for Grad-CAM targets."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["class"]


def swin_reshape_transform(tensor):
    """Convert timm Swin activations from NHWC/token layouts to NCHW for Grad-CAM."""
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        size = int(tokens ** 0.5)
        if size * size != tokens:
            raise ValueError(f"Cannot reshape {tokens} Swin tokens into a square grid.")
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
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    class_probs, class_preds, class_labels = [], [], []
    fresh_preds, fresh_labels = [], []
    for images, targets in loader:
        images = images.to(device)
        outputs = model(images)
        probs = outputs["class"].softmax(1)
        class_probs.append(probs.cpu().numpy())
        class_preds.append(probs.argmax(1).cpu().numpy())
        class_labels.append(targets["class"].numpy())
        fresh_preds.append(predict_coral_rank(outputs["freshness"]).cpu().numpy())
        fresh_labels.append(targets["freshness"].numpy())
    return {
        "class_probs": np.concatenate(class_probs),
        "class_preds": np.concatenate(class_preds),
        "class_labels": np.concatenate(class_labels),
        "fresh_preds": np.concatenate(fresh_preds),
        "fresh_labels": np.concatenate(fresh_labels),
    }


def save_prediction_csv(preds, samples, metadata):
    rows = []
    for i, sample in enumerate(samples):
        true_class = metadata.class_names[preds["class_labels"][i]]
        pred_class = metadata.class_names[preds["class_preds"][i]]
        rows.append({
            "path": sample.path,
            "true_class": true_class,
            "pred_class": pred_class,
            "true_freshness": RANK_TO_FRESHNESS[int(preds["fresh_labels"][i])],
            "pred_freshness": RANK_TO_FRESHNESS[int(preds["fresh_preds"][i])],
            "class_correct": int(preds["class_labels"][i] == preds["class_preds"][i]),
            "freshness_abs_error": int(abs(preds["fresh_labels"][i] - preds["fresh_preds"][i])),
            "confidence": float(preds["class_probs"][i, preds["class_preds"][i]]),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "hierarchical_ordinal_test_predictions.csv", index=False)


def plot_class_confusion(preds, metadata):
    cm = confusion_matrix(preds["class_labels"], preds["class_preds"], labels=np.arange(metadata.num_classes))
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=metadata.class_names,
        yticklabels=metadata.class_names,
        cbar_kws={"label": "Images"},
        ax=ax,
    )
    ax.set_title("Hierarchical Ordinal Swin-Tiny: 24-Class Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hierarchical_ordinal_24class_confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_freshness_confusion(preds):
    labels = [0, 1, 2]
    names = [RANK_TO_FRESHNESS[i] for i in labels]
    cm = confusion_matrix(preds["fresh_labels"], preds["fresh_preds"], labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title("Ordered Freshness Confusion Matrix")
    ax.set_xlabel("Predicted freshness")
    ax.set_ylabel("True freshness")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hierarchical_ordinal_freshness_confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_ordinal_errors(preds):
    abs_errors = np.abs(preds["fresh_labels"] - preds["fresh_preds"])
    counts = pd.Series(abs_errors).value_counts().reindex([0, 1, 2], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Correct (0)", "Adjacent (1)", "Severe (2)"], counts.values, color=["#2d7f5e", "#d69f32", "#bd3f32"])
    ax.set_title("Freshness Ordinal Error Distribution")
    ax.set_ylabel("Test images")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hierarchical_ordinal_error_distribution.png", dpi=220)
    plt.close(fig)


def plot_comparison_table():
    table = pd.read_csv("results/hierarchical_ordinal_comparison_table.csv")
    rows = [
        ("Prasetyo ResNet50", 78.82),
        ("Hoang 2025 SOTA", 85.99),
        ("Ours neural", 88.46),
        ("Ours + ExtraTrees", 89.98),
        ("Ours + LightGBM", 89.41),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, values = zip(*rows)
    colors = ["#7a8793", "#7a8793", "#2166ac", "#1b7837", "#5aae61"]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(70, 92)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("FFE Benchmark Comparison")
    for bar, value in zip(bars, values):
        ax.text(value + 0.2, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hierarchical_ordinal_benchmark_comparison.png", dpi=220)
    table.to_csv(OUT_DIR / "hierarchical_ordinal_paper_comparison_table.csv", index=False)
    plt.close(fig)


def preprocess_for_gradcam(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    rgb = np.asarray(image).astype(np.float32) / 255.0
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tensor = normalize(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    return tensor, rgb


def select_gradcam_samples(preds, samples):
    selected = {}
    confidence = preds["class_probs"].max(axis=1)
    order = np.argsort(-confidence)
    for idx in order:
        true_fresh = int(preds["fresh_labels"][idx])
        if true_fresh in selected:
            continue
        if preds["class_labels"][idx] == preds["class_preds"][idx] and preds["fresh_labels"][idx] == preds["fresh_preds"][idx]:
            selected[true_fresh] = idx
        if len(selected) == 3:
            break
    for idx in order:
        true_fresh = int(preds["fresh_labels"][idx])
        selected.setdefault(true_fresh, idx)
        if len(selected) == 3:
            break
    return [selected[rank] for rank in sorted(selected)]


def generate_gradcam_figures(model, preds, samples, metadata, device, image_size):
    wrapped = ClassHeadWrapper(model).to(device).eval()
    target_layers = [wrapped.model.backbone.layers[-1].blocks[-1].norm2]
    sample_indices = select_gradcam_samples(preds, samples)
    panel_images = []
    titles = []

    for idx in sample_indices:
        sample = samples[idx]
        target_class = int(preds["class_preds"][idx])
        tensor, rgb = preprocess_for_gradcam(sample.path, image_size)
        with GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=swin_reshape_transform) as cam:
            mask = cam(input_tensor=tensor.to(device), targets=[ClassifierOutputTarget(target_class)])[0]
        overlay = show_cam_on_image(rgb, mask, use_rgb=True)
        fresh = RANK_TO_FRESHNESS[int(preds["fresh_labels"][idx])]
        panel_images.append((rgb, overlay))
        titles.append(f"{fresh}\nPred: {metadata.class_names[target_class]}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        fig.suptitle(titles[-1], fontsize=10)
        fig.tight_layout()
        safe = fresh.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"gradcam_{safe}.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(len(panel_images), 2, figsize=(8, 4 * len(panel_images)))
    if len(panel_images) == 1:
        axes = np.asarray([axes])
    for row, ((rgb, overlay), title) in enumerate(zip(panel_images, titles)):
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{title}\nOriginal", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title("Grad-CAM", fontsize=9)
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hierarchical_ordinal_gradcam_panel.png", dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, metadata = get_hierarchical_dataloaders(
        cfg,
        split_seed=42,
        batch_size=64,
        num_workers=0,
    )
    model = load_model(metadata, device)
    preds = collect_predictions(model, test_loader, device)
    samples = test_loader.dataset.samples

    save_prediction_csv(preds, samples, metadata)
    plot_class_confusion(preds, metadata)
    plot_freshness_confusion(preds)
    plot_ordinal_errors(preds)
    plot_comparison_table()
    generate_gradcam_figures(model, preds, samples, metadata, device, cfg["dataset"]["image_size"])
    print(f"Saved hierarchical ordinal journal figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
