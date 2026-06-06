#!/usr/bin/env python3
"""Generate JuTISI paper figures (300 dpi, solid-fill, B/W-safe).

Figure 1 — FFE sample grid across freshness levels.
Figure 2 — CLAHE ablation (grouped bars, per backbone, with seed SD).
Figure 3 — Pooling-operator comparison (per-seed points + mean bar, A/B/C/D).
Figure 4 — Final GAP Swin-Tiny context against published FFE results.
"""
from __future__ import annotations
from pathlib import Path
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "jutisi"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "savefig.dpi": 300})

# solid, B/W-distinguishable fills
C_NO, C_YES = "#1f3b73", "#c0392b"      # dark blue / brick red
GREY = "#888888"


def fig1_ffe_samples():
    """Show representative examples from two species across three freshness levels."""
    data_dir = ROOT / "data" / "FFE"
    species = ["Chanos Chanos", "Oreochromis Niloticus"]
    freshness = ["Highly Fresh", "Fresh", "Not Fresh"]
    fig, axes = plt.subplots(len(species), len(freshness), figsize=(7.2, 4.4))
    rng = random.Random(42)

    for row, sp in enumerate(species):
        for col, fr in enumerate(freshness):
            ax = axes[row, col]
            class_dir = data_dir / f"{sp} - {fr}"
            images = sorted(class_dir.glob("*.jpg"))
            if not images:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
                continue
            img_path = images[rng.randrange(len(images))]
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(fr if row == 0 else "", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(sp.replace(" ", "\n"), fontsize=10)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

    fig.suptitle("Representative FFE Eye Images", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "figure1_ffe_samples.png")
    plt.close(fig)
    print("wrote", OUT / "figure1_ffe_samples.png")


def fig2_clahe():
    df = pd.read_csv(ROOT / "results" / "multiseed_summary.csv")
    order = ["resnet50", "efficientnetv2s", "convnext_small"]
    labels = ["ResNet50", "EfficientNetV2-S", "ConvNeXt-Small"]
    no = df[df.clahe == False].set_index("backbone")
    yes = df[df.clahe == True].set_index("backbone")
    x = np.arange(len(order)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, [no.loc[b, "acc_mean"] for b in order], w,
                yerr=[no.loc[b, "acc_std"] for b in order], capsize=4,
                color=C_NO, label="No CLAHE", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, [yes.loc[b, "acc_mean"] for b in order], w,
                yerr=[yes.loc[b, "acc_std"] for b in order], capsize=4,
                color=C_YES, label="With CLAHE", edgecolor="black", linewidth=0.6,
                hatch="///")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.15,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(70, 84)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#dddddd")
    fig.tight_layout(); fig.savefig(OUT / "figure2_clahe_ablation.png"); plt.close(fig)
    print("wrote", OUT / "figure2_clahe_ablation.png")


def fig3_pooling():
    df = pd.read_csv(ROOT / "results" / "secondorder_merged" / "secondorder_results.csv")
    order = ["A_gap", "B_raw_bilinear", "C_gap_raw_bilinear", "D_centered_cov"]
    labels = ["A: GAP\n(baseline)", "B: mean-pres.\n2nd-order", "C: GAP +\n2nd-order", "D: centered\ncov. (control)"]
    colors = [C_NO, C_YES, C_YES, GREY]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(order):
        vals = df[df.arm == arm]["test_class_acc"].to_numpy()
        m, s = vals.mean(), vals.std(ddof=1)
        ax.bar(i, m, 0.55, color=colors[i], edgecolor="black", linewidth=0.6,
               alpha=0.85, hatch=("" if arm == "A_gap" else "///" if colors[i] == C_YES else ".."))
        ax.errorbar(i, m, yerr=s, color="black", capsize=5, lw=1)
        jit = (rng.random(len(vals)) - 0.5) * 0.22
        ax.scatter(np.full(len(vals), i) + jit, vals, s=26, color="black", zorder=3)
        ax.text(i, m + s + 0.2, f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(df[df.arm == "A_gap"]["test_class_acc"].mean(), color=C_NO, ls="--", lw=0.9, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test 24-class accuracy (%)"); ax.set_ylim(83, 90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#dddddd")
    fig.tight_layout(); fig.savefig(OUT / "figure3_pooling_comparison.png"); plt.close(fig)
    print("wrote", OUT / "figure3_pooling_comparison.png")


def fig4_final_model_context():
    labels = [
        "Prasetyo\nResNet50",
        "Yildiz\nVGG19+ANN",
        "Hoang\nSwin+RF/LGBM",
        "This study\nGAP Swin-Tiny",
    ]
    values = np.array([78.82, 77.30, 85.99, 88.53])
    errors = np.array([0.0, 0.0, 0.0, 0.75])
    colors = ["#888888", "#888888", "#c0392b", "#1f3b73"]
    hatches = ["", "", "///", ""]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.7)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.55,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Reported FFE accuracy (%)")
    ax.set_ylim(70, 91)
    ax.text(3, 72.2, "5 seeds", ha="center", fontsize=8)
    ax.text(2, 72.2, "published\nsingle run", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#dddddd")
    fig.tight_layout()
    fig.savefig(OUT / "figure4_final_model_context.png")
    plt.close(fig)
    print("wrote", OUT / "figure4_final_model_context.png")


if __name__ == "__main__":
    fig1_ffe_samples()
    fig2_clahe()
    fig3_pooling()
    fig4_final_model_context()
