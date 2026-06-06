"""Run a stronger final-training pass for all four publication scenarios."""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd
import torch

from src.dataset import get_dataloaders, load_config
from src.evaluate import evaluate_checkpoint
from src.final_optimization import build_final_experiments, make_optimized_config
from src.models import (
    build_efficientnetv2s,
    build_resnet50,
    count_parameters,
    freeze_backbone,
)
from src.train import run_experiment


def build_model(exp, cfg):
    if exp["backbone"] == "resnet50":
        return build_resnet50(
            num_classes=cfg["dataset"]["num_classes"],
            pretrained=cfg["models"]["resnet50"]["pretrained"],
            dropout=cfg["training"]["dropout"],
        )
    return build_efficientnetv2s(
        num_classes=cfg["dataset"]["num_classes"],
        pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"],
        dropout=cfg["training"]["dropout"],
    )


def main():
    cfg = make_optimized_config(load_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Final optimization config:")
    for key in ["epochs", "patience", "learning_rate", "unfreeze_lr", "dropout", "label_smoothing", "amp", "channels_last"]:
        print(f"  {key}: {cfg['training'].get(key)}")

    results = []
    for exp in build_final_experiments():
        print(f"\n{'=' * 60}")
        print(f"FINAL OPTIMIZATION: {exp['name']}")
        print(f"{'=' * 60}")

        train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg, use_clahe=exp["use_clahe"])
        model = build_model(exp, cfg)
        freeze_backbone(model, exp["model_name"])
        total_params, trainable_params = count_parameters(model)
        print(f"Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable during warmup")
        model = model.to(device)

        best_val_acc, ckpt_path = run_experiment(
            experiment_name=exp["name"],
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            model_name=exp["model_name"],
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
            "backbone": exp["backbone"],
            "params_M": round(total_params / 1e6, 2),
            "best_val_acc": round(best_val_acc * 100, 2),
            **metrics,
        })

    Path("results").mkdir(exist_ok=True)
    out_path = "results/final_optimization_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print("\n=== FINAL OPTIMIZATION RESULTS ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
