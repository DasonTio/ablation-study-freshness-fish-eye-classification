"""
Revision figures:
  1. Legible 24-class confusion matrix (abbreviated labels + cell counts) from the
     saved per-sample predictions CSV (no model load).
  2. Per-class correct-vs-incorrect Grad-CAM xAI panel for the hierarchical model,
     in the style of one-correct / one-incorrect comparison per class.
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.hierarchical import HierarchicalClassifier, get_hierarchical_dataloaders, parse_ffe_class_name
from src.recipe import IMAGENET_MEAN, IMAGENET_STD

OUT = Path("results/figures/hierarchical_ordinal"); OUT.mkdir(parents=True, exist_ok=True)
CSV = "results/figures/hierarchical_ordinal/hierarchical_ordinal_test_predictions.csv"
HIER_CKPT = "results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth"
BACKBONE = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
SP_CODE = {"Chanos Chanos": "CC", "Johnius Trachycephalus": "JT", "Nibea Albiflora": "NA",
           "Rastrelliger Faughni": "RF", "Upeneus Moluccensis": "UM",
           "Eleutheronema Tetradactylum": "ET", "Oreochromis Mossambicus": "OM",
           "Oreochromis Niloticus": "ON"}
FR_CODE = {"Highly Fresh": "HF", "Fresh": "F", "Not Fresh": "NF"}


def short(name):
    p = parse_ffe_class_name(name)
    return f"{SP_CODE[p.species]}-{FR_CODE[p.freshness]}"


def confusion_figure():
    df = pd.read_csv(CSV)
    labels = sorted(df["true_class"].unique(), key=lambda c: short(c))
    cm = confusion_matrix(df["true_class"], df["pred_class"], labels=labels)
    short_labels = [short(c) for c in labels]
    annot = np.where(cm > 0, cm.astype(str), "")
    fig, ax = plt.subplots(figsize=(15, 12.5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", square=True,
                xticklabels=short_labels, yticklabels=short_labels,
                annot_kws={"size": 9}, linewidths=0.4, linecolor="#dddddd",
                cbar_kws={"label": "Jumlah citra", "shrink": 0.7}, ax=ax)
    acc = (df["true_class"] == df["pred_class"]).mean() * 100
    ax.set_title(f"Matriks Konfusi 24-Kelas — Model Ordinal Hierarkis (akurasi uji {acc:.2f}%)",
                 fontsize=14, pad=12)
    ax.set_xlabel("Kelas prediksi", fontsize=12); ax.set_ylabel("Kelas sebenarnya", fontsize=12)
    ax.tick_params(axis="x", labelrotation=90, labelsize=11)
    ax.tick_params(axis="y", labelrotation=0, labelsize=11)
    legend = "Kode spesies: " + "  ".join(f"{v}={k}" for k, v in SP_CODE.items()) + \
             "   |   HF=Highly Fresh, F=Fresh, NF=Not Fresh"
    fig.text(0.5, 0.01, legend, ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT / "hierarchical_ordinal_24class_confusion_matrix_readable.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("CM saved:", out)


def swin_reshape(t):
    if t.ndim == 4:
        return t.permute(0, 3, 1, 2)
    if t.ndim == 3:
        b, n, c = t.shape; s = int(n ** 0.5)
        return t.reshape(b, s, s, c).permute(0, 3, 1, 2)
    return t


class ClassHead(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, x): return self.model(x)["class"]


def preprocess(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size))
    rgb = np.asarray(img).astype(np.float32) / 255.0
    t = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    return t, rgb


def cam_mask(model, tensor, target):
    w = ClassHead(model).eval()
    layers = [w.model.backbone.layers[-1].blocks[-1].norm2]
    with GradCAM(model=w, target_layers=layers, reshape_transform=swin_reshape) as cam:
        return cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target)])[0]


def xai_figure():
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    _, _, _, meta = get_hierarchical_dataloaders(cfg, split_seed=42, batch_size=32, num_workers=0)
    name_to_idx = {n: i for i, n in enumerate(meta.class_names)}
    model = HierarchicalClassifier(BACKBONE, meta.num_classes, meta.num_species, pretrained=False)
    model.load_state_dict(torch.load(HIER_CKPT, map_location="cpu", weights_only=True))
    model.eval()

    df = pd.read_csv(CSV)
    # classes that have both a correct and an incorrect test sample; prefer freshness mistakes
    chosen = []
    for cls in df["true_class"].unique():
        sub = df[df["true_class"] == cls]
        cor = sub[sub["class_correct"] == 1]
        inc = sub[sub["class_correct"] == 0]
        if len(cor) and len(inc):
            fresh_err = inc[inc["true_freshness"] != inc["pred_freshness"]]
            inc_row = (fresh_err.sort_values("confidence", ascending=False).iloc[0]
                       if len(fresh_err) else inc.sort_values("confidence", ascending=False).iloc[0])
            cor_row = cor.sort_values("confidence", ascending=False).iloc[0]
            chosen.append((cls, cor_row, inc_row, len(fresh_err)))
    # prefer distinct species, prioritise freshness-mistake examples
    chosen.sort(key=lambda x: (-x[3]))
    picked, seen = [], set()
    for c in chosen:
        sp = parse_ffe_class_name(c[0]).species
        if sp not in seen:
            picked.append(c); seen.add(sp)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        picked = chosen[:3]

    nrows = len(picked)
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 3.5 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])
    for ci, ct in enumerate(["Asli (benar)", "Grad-CAM (benar)", "Asli (salah)", "Grad-CAM (salah)"]):
        axes[0, ci].set_title(ct, fontsize=11, fontweight="bold", pad=8)

    for r, (cls, cor, inc, _) in enumerate(picked):
        for off, row in ((0, cor), (2, inc)):
            tensor, rgb = preprocess(row["path"])
            target = name_to_idx[row["pred_class"]]
            mask = cam_mask(model, tensor, target)
            overlay = show_cam_on_image(rgb, mask, use_rgb=True)
            axes[r, off].imshow(rgb); axes[r, off].axis("off")
            axes[r, off + 1].imshow(overlay); axes[r, off + 1].axis("off")
            tag = "BENAR" if off == 0 else "SALAH"
            axes[r, off + 1].set_xlabel(
                f"{tag}\npred: {short(row['pred_class'])}  (conf {row['confidence']:.2f})", fontsize=8.5)
            axes[r, off + 1].axis("on"); axes[r, off + 1].set_xticks([]); axes[r, off + 1].set_yticks([])
        axes[r, 0].set_ylabel(f"Kelas sebenarnya:\n{short(cls)}", fontsize=9.5)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

    fig.suptitle("Grad-CAM Explainable AI — Perbandingan Prediksi Benar vs Salah (Model Ordinal Hierarkis)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(h_pad=3.0)
    out = OUT / "xai_correct_vs_incorrect.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("xAI figure saved:", out, "| classes:", [short(c[0]) for c in picked])


if __name__ == "__main__":
    confusion_figure()
    xai_figure()
