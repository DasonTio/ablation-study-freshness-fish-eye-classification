"""Run all 4 ablation experiments sequentially. Results saved to results/."""
import sys
sys.path.insert(0, ".")

import torch
import pandas as pd
from pathlib import Path

from src.dataset import load_config, get_dataloaders
from src.models import build_resnet50, build_efficientnetv2s, freeze_backbone, count_parameters
from src.train import run_experiment
from src.evaluate import evaluate_checkpoint

cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

EXPERIMENTS = [
    {"name": "exp_A_resnet50_no_clahe",   "backbone": "resnet50",        "model_name": "resnet50",       "use_clahe": False},
    {"name": "exp_B_resnet50_clahe",      "backbone": "resnet50",        "model_name": "resnet50",       "use_clahe": True},
    {"name": "exp_C_v2s_no_clahe",        "backbone": "efficientnetv2s", "model_name": "efficientnetv2s","use_clahe": False},
    {"name": "exp_D_v2s_clahe_proposed",  "backbone": "efficientnetv2s", "model_name": "efficientnetv2s","use_clahe": True},
]

results_table = []

for exp in EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp['name']}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg, use_clahe=exp["use_clahe"])

    if exp["backbone"] == "resnet50":
        model = build_resnet50(
            num_classes=cfg["dataset"]["num_classes"],
            pretrained=cfg["models"]["resnet50"]["pretrained"],
            dropout=cfg["training"]["dropout"]
        )
    else:
        model = build_efficientnetv2s(
            num_classes=cfg["dataset"]["num_classes"],
            pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"],
            dropout=cfg["training"]["dropout"]
        )

    freeze_backbone(model, exp["model_name"])
    total_params, _ = count_parameters(model)
    print(f"Parameters: {total_params/1e6:.2f}M")
    model = model.to(device)

    best_val_acc, ckpt_path = run_experiment(
        experiment_name=exp["name"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        model_name=exp["model_name"]
    )

    metrics = evaluate_checkpoint(
        model=model,
        checkpoint_path=ckpt_path,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        figures_dir=cfg["paths"]["figures"],
        experiment_name=exp["name"]
    )

    results_table.append({
        "experiment": exp["name"],
        "clahe": exp["use_clahe"],
        "backbone": exp["backbone"],
        "params_M": round(total_params / 1e6, 2),
        **metrics
    })

df = pd.DataFrame(results_table)
Path("results").mkdir(exist_ok=True)
table_path = "results/ablation_results.csv"
df.to_csv(table_path, index=False)
print(f"\n\n=== ABLATION RESULTS ===")
print(df.to_string(index=False))
print(f"\nSaved: {table_path}")
