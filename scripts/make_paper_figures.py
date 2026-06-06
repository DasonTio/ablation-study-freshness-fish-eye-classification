"""Generate publication-ready results table and training curves. Run AFTER ablation experiments complete."""
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/figures").mkdir(parents=True, exist_ok=True)

# --- Table: Ablation + benchmark comparison ---
try:
    df = pd.read_csv("results/ablation_results.csv")
    display_df = df[["experiment", "params_M", "accuracy", "precision", "recall", "f1"]].copy()
    display_df.columns = ["Model", "Params (M)", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]
except FileNotFoundError:
    print("WARNING: results/ablation_results.csv not found. Run run_ablation.py first.")
    display_df = pd.DataFrame()

# Add benchmark rows from paper (Prasetyo et al. 2022, Table 2 — FFE dataset)
benchmark = pd.DataFrame([
    {"Model": "MobileNetV1 [paper]", "Params (M)": 3.22,  "Accuracy (%)": 59.11, "Precision (%)": 59.74, "Recall (%)": 57.98, "F1 (%)": 58.85},
    {"Model": "MobileNetV2 [paper]", "Params (M)": 2.25,  "Accuracy (%)": 53.87, "Precision (%)": 52.10, "Recall (%)": 50.95, "F1 (%)": 51.53},
    {"Model": "ResNet50 [paper]",    "Params (M)": 23.59, "Accuracy (%)": 78.82, "Precision (%)": 79.14, "Recall (%)": 77.70, "F1 (%)": 78.41},
    {"Model": "DenseNet121 [paper]", "Params (M)": 7.04,  "Accuracy (%)": 42.37, "Precision (%)": 42.50, "Recall (%)": 38.41, "F1 (%)": 40.35},
    {"Model": "VGG16 [paper]",       "Params (M)": 14.71, "Accuracy (%)": 43.85, "Precision (%)": 45.81, "Recall (%)": 41.38, "F1 (%)": 43.48},
    {"Model": "NASNet Mobile [paper]","Params (M)": 4.27,  "Accuracy (%)": 37.24, "Precision (%)": 33.66, "Recall (%)": 30.61, "F1 (%)": 33.37},
    {"Model": "MB-BE [paper]",        "Params (M)": 3.16,  "Accuracy (%)": 60.02, "Precision (%)": 58.41, "Recall (%)": 58.06, "F1 (%)": 58.24},
])

full_table = pd.concat([benchmark, display_df], ignore_index=True) if not display_df.empty else benchmark
full_table.to_csv("results/final_comparison_table.csv", index=False)
print("=== FULL COMPARISON TABLE ===")
print(full_table.to_string(index=False))
print(f"\nSaved: results/final_comparison_table.csv")

# --- Figure: Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

CURVES = [
    ("exp_A_resnet50_no_clahe",    "ResNet50 (no CLAHE)",       "steelblue"),
    ("exp_B_resnet50_clahe",       "ResNet50 + CLAHE",          "royalblue"),
    ("exp_C_v2s_no_clahe",         "EfficientNetV2-S (no CLAHE)","darkorange"),
    ("exp_D_v2s_clahe_proposed",   "EfficientNetV2-S + CLAHE",  "green"),
    ("convnext_no_clahe",          "ConvNeXt-S (no CLAHE)",     "purple"),
    ("convnext_clahe",             "ConvNeXt-S + CLAHE",        "crimson"),
]

for exp_name, label, color in CURVES:
    log_path = f"results/logs/{exp_name}.csv"
    try:
        log = pd.read_csv(log_path)
        log["train_acc"] = log["train_acc"].astype(float) * 100
        log["val_acc"]   = log["val_acc"].astype(float) * 100
        axes[0].plot(log["epoch"], log["train_acc"], label=f"{label} (train)", color=color, linestyle="--", alpha=0.7)
        axes[0].plot(log["epoch"], log["val_acc"],   label=f"{label} (val)",   color=color)
        axes[1].plot(log["epoch"], log["train_loss"].astype(float), label=f"{label} (train)", color=color, linestyle="--", alpha=0.7)
        axes[1].plot(log["epoch"], log["val_loss"].astype(float),   label=f"{label} (val)",   color=color)
    except FileNotFoundError:
        print(f"Log not found: {log_path} — run ablation first")

axes[0].set_title("Training & Validation Accuracy", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].set_title("Training & Validation Loss", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/training_curves.png", dpi=150, bbox_inches="tight")
print("Training curves saved: results/figures/training_curves.png")
