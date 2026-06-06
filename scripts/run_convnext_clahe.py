"""ConvNeXt-Small ablation: no-CLAHE vs CLAHE under the final optimization protocol.

ConvNeXt-Small uses 7x7 depthwise conv throughout — larger local receptive field
preserves spatial contrast gradients that CLAHE amplifies, unlike ResNet50/EfficientNetV2-S
which rely on aggressive spatial pooling.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd
import torch

from src.dataset import get_dataloaders, load_config
from src.evaluate import evaluate_checkpoint
from src.final_optimization import make_optimized_config
from src.models import build_convnext_small, count_parameters, freeze_backbone
from src.train import run_experiment


EXPERIMENTS = [
    {"name": "convnext_no_clahe", "use_clahe": False},
    {"name": "convnext_clahe",    "use_clahe": True},
]


def main():
    cfg = make_optimized_config(load_config())
    # channels_last can conflict with ConvNeXt LayerNorm; disable for safety.
    cfg["training"]["channels_last"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("ConvNeXt config:")
    for key in ["epochs", "patience", "learning_rate", "unfreeze_lr", "dropout", "label_smoothing", "amp"]:
        print(f"  {key}: {cfg['training'].get(key)}")

    results = []
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"CONVNEXT EXPERIMENT: {exp['name']}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader, class_names = get_dataloaders(
            cfg, use_clahe=exp["use_clahe"]
        )
        model = build_convnext_small(
            num_classes=cfg["dataset"]["num_classes"],
            dropout=cfg["training"]["dropout"],
        )
        freeze_backbone(model, "convnext_small")
        total_params, trainable_params = count_parameters(model)
        print(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable during warmup")
        model = model.to(device)

        best_val_acc, ckpt_path = run_experiment(
            experiment_name=exp["name"],
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            model_name="convnext_small",
        )
        metrics = evaluate_checkpoint(
            model=model,
            checkpoint_path=ckpt_path,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            figures_dir=cfg["paths"]["figures"],
            experiment_name=exp["name"],
        )

        results.append({
            "experiment": exp["name"],
            "clahe": exp["use_clahe"],
            "backbone": "convnext_small",
            "params_M": round(total_params / 1e6, 2),
            "best_val_acc": round(best_val_acc * 100, 2),
            **metrics,
        })

    Path("results").mkdir(exist_ok=True)
    out_path = "results/convnext_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print("\n=== CONVNEXT RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")

    _run_gradcam(cfg, device)
    _update_comparison_table(df)

    # Touch CONVNEXT_DONE here so local watcher can start rsyncing,
    # then sleep to give it time before bash fires vastai stop.
    import time
    Path("results/CONVNEXT_DONE").touch()
    print("\n[done] CONVNEXT_DONE written. Sleeping 10 min for local rsync before instance stops.")
    time.sleep(600)


def _run_gradcam(cfg, device):
    """Generate Grad-CAM comparison for ConvNeXt no-CLAHE vs CLAHE."""
    print("\n[ConvNeXt Grad-CAM] Generating figures...")
    from src.dataset import build_class_map
    from src.gradcam import generate_gradcam_comparison
    from src.models import build_convnext_small

    ckpt_no   = "results/checkpoints/convnext_no_clahe_best.pth"
    ckpt_with = "results/checkpoints/convnext_clahe_best.pth"
    if not (Path(ckpt_no).exists() and Path(ckpt_with).exists()):
        print("[ConvNeXt Grad-CAM] Checkpoints not found, skipping.")
        return

    num_classes = cfg["dataset"]["num_classes"]
    dropout     = cfg["training"]["dropout"]

    model_no = build_convnext_small(num_classes=num_classes, dropout=dropout).to(device)
    model_no.load_state_dict(torch.load(ckpt_no, map_location=device))

    model_with = build_convnext_small(num_classes=num_classes, dropout=dropout).to(device)
    model_with.load_state_dict(torch.load(ckpt_with, map_location=device))

    samples, class_names = build_class_map(cfg["dataset"]["root"])
    Path("results/figures/gradcam").mkdir(parents=True, exist_ok=True)

    freshness_levels = ["Highly Fresh", "Fresh", "Not Fresh"]
    generated = []
    for path, label_idx in samples:
        class_name = class_names[label_idx]
        freshness = next((f for f in freshness_levels if f in class_name), None)
        if freshness and freshness not in generated:
            safe_name = freshness.lower().replace(" ", "_")
            generate_gradcam_comparison(
                image_path=path,
                model_no_clahe=model_no,
                model_with_clahe=model_with,
                backbone_name="convnext_small",
                target_class=label_idx,
                device=device,
                save_path=f"results/figures/gradcam/convnext_gradcam_{safe_name}.png",
            )
            generated.append(freshness)
        if len(generated) == 3:
            break
    print("[ConvNeXt Grad-CAM] Done.")


def _update_comparison_table(convnext_df: pd.DataFrame):
    """Append ConvNeXt rows to the existing final_comparison_table.csv."""
    table_path = Path("results/final_comparison_table.csv")
    if not table_path.exists():
        print("[Table] final_comparison_table.csv not found, skipping merge.")
        return

    existing = pd.read_csv(table_path)
    rows = []
    for _, row in convnext_df.iterrows():
        label = f"ConvNeXt-S {'+ CLAHE' if row['clahe'] else '(no CLAHE)'} [ours]"
        rows.append({
            "Model":          label,
            "Params (M)":     row["params_M"],
            "Accuracy (%)":   row["accuracy"],
            "Precision (%)":  row["precision"],
            "Recall (%)":     row["recall"],
            "F1 (%)":         row["f1"],
        })
    updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    updated.to_csv(table_path, index=False)
    print(f"[Table] Updated {table_path} with ConvNeXt rows.")
    print(updated.to_string(index=False))


if __name__ == "__main__":
    main()
