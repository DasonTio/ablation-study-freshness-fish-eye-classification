"""
Full statistical study: 6 configs x 3 seeds = 18 runs, with mean +/- std error bars.

Configs:  ResNet50 / EfficientNetV2-S / ConvNeXt-Small, each with and without CLAHE.
Seeds:    42, 123, 2024 (vary data split + init + augmentation).

Runs everything in ONE process (no chained tmux sessions). Resilient: writes results
incrementally, wraps each run in try/except, and only touches results/ALL_DONE at the
very end. The local watcher destroys the instance after it confirms the download.

Outputs:
  results/multiseed_results.csv     raw 18 rows
  results/multiseed_summary.csv     6 rows, mean/std per config
  results/final_comparison_table.csv  benchmark + our rows (mean +/- std)
  results/checkpoints/<key>_BEST.pth  best-seed checkpoint per config
  results/figures/gradcam/*           Grad-CAM for ConvNeXt + EfficientNetV2-S
  results/ALL_DONE                    final marker
"""
import sys
sys.path.insert(0, ".")

import shutil
import traceback
from pathlib import Path

import pandas as pd
import torch

from src.dataset import get_dataloaders, load_config
from src.evaluate import evaluate_checkpoint
from src.final_optimization import make_optimized_config
from src.models import (
    build_convnext_small,
    build_efficientnetv2s,
    build_resnet50,
    count_parameters,
    freeze_backbone,
)
from src.seed import set_seed
from src.train import run_experiment

SEEDS = [42, 123, 2024]

CONFIGS = [
    {"key": "resnet50_no_clahe", "backbone": "resnet50",        "model_name": "resnet50",        "use_clahe": False},
    {"key": "resnet50_clahe",    "backbone": "resnet50",        "model_name": "resnet50",        "use_clahe": True},
    {"key": "v2s_no_clahe",      "backbone": "efficientnetv2s", "model_name": "efficientnetv2s", "use_clahe": False},
    {"key": "v2s_clahe",         "backbone": "efficientnetv2s", "model_name": "efficientnetv2s", "use_clahe": True},
    {"key": "convnext_no_clahe", "backbone": "convnext_small",  "model_name": "convnext_small",  "use_clahe": False},
    {"key": "convnext_clahe",    "backbone": "convnext_small",  "model_name": "convnext_small",  "use_clahe": True},
]

BENCHMARK_ROWS = [
    {"Model": "MobileNetV1 [paper]",  "Params (M)": 3.22,  "Accuracy (%)": "59.11", "Precision (%)": "59.74", "Recall (%)": "57.98", "F1 (%)": "58.85"},
    {"Model": "MobileNetV2 [paper]",  "Params (M)": 2.25,  "Accuracy (%)": "53.87", "Precision (%)": "52.10", "Recall (%)": "50.95", "F1 (%)": "51.53"},
    {"Model": "ResNet50 [paper]",     "Params (M)": 23.59, "Accuracy (%)": "78.82", "Precision (%)": "79.14", "Recall (%)": "77.70", "F1 (%)": "78.41"},
    {"Model": "DenseNet121 [paper]",  "Params (M)": 7.04,  "Accuracy (%)": "42.37", "Precision (%)": "42.50", "Recall (%)": "38.41", "F1 (%)": "40.35"},
    {"Model": "VGG16 [paper]",        "Params (M)": 14.71, "Accuracy (%)": "43.85", "Precision (%)": "45.81", "Recall (%)": "41.38", "F1 (%)": "43.48"},
    {"Model": "NASNet Mobile [paper]","Params (M)": 4.27,  "Accuracy (%)": "37.24", "Precision (%)": "33.66", "Recall (%)": "30.61", "F1 (%)": "33.37"},
    {"Model": "MB-BE [paper]",        "Params (M)": 3.16,  "Accuracy (%)": "60.02", "Precision (%)": "58.41", "Recall (%)": "58.06", "F1 (%)": "58.24"},
]


def build_model(backbone, cfg):
    if backbone == "resnet50":
        return build_resnet50(
            num_classes=cfg["dataset"]["num_classes"],
            pretrained=cfg["models"]["resnet50"]["pretrained"],
            dropout=cfg["training"]["dropout"],
        )
    if backbone == "efficientnetv2s":
        return build_efficientnetv2s(
            num_classes=cfg["dataset"]["num_classes"],
            pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"],
            dropout=cfg["training"]["dropout"],
        )
    if backbone == "convnext_small":
        return build_convnext_small(
            num_classes=cfg["dataset"]["num_classes"],
            dropout=cfg["training"]["dropout"],
        )
    raise ValueError(backbone)


def run_one(cfg_base, config, seed, device):
    """One (config, seed) run. Returns a result dict or None on failure."""
    cfg = make_optimized_config(cfg_base)
    # ConvNeXt LayerNorm can conflict with channels_last; disable for it.
    if config["backbone"] == "convnext_small":
        cfg["training"]["channels_last"] = False

    set_seed(seed)
    exp_name = f"{config['key']}_seed{seed}"
    print(f"\n{'='*64}\nRUN: {exp_name}\n{'='*64}", flush=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        cfg, use_clahe=config["use_clahe"], seed=seed
    )
    model = build_model(config["backbone"], cfg)
    freeze_backbone(model, config["model_name"])
    total_params, trainable = count_parameters(model)
    print(f"Params: {total_params/1e6:.2f}M total, {trainable/1e6:.2f}M trainable warmup", flush=True)
    model = model.to(device)

    best_val_acc, ckpt_path = run_experiment(
        experiment_name=exp_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        model_name=config["model_name"],
    )
    metrics = evaluate_checkpoint(
        model=model,
        checkpoint_path=ckpt_path,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        figures_dir=cfg["paths"]["figures"],
        experiment_name=exp_name,
    )
    return {
        "config": config["key"],
        "seed": seed,
        "backbone": config["backbone"],
        "clahe": config["use_clahe"],
        "params_M": round(total_params / 1e6, 2),
        "best_val_acc": round(best_val_acc * 100, 2),
        "checkpoint": ckpt_path,
        **metrics,
    }


def aggregate(results_df):
    """Mean/std per config across seeds."""
    rows = []
    for key, grp in results_df.groupby("config", sort=False):
        rows.append({
            "config": key,
            "backbone": grp["backbone"].iloc[0],
            "clahe": grp["clahe"].iloc[0],
            "params_M": grp["params_M"].iloc[0],
            "n_seeds": len(grp),
            "acc_mean": round(grp["accuracy"].mean(), 2),
            "acc_std": round(grp["accuracy"].std(ddof=1), 2) if len(grp) > 1 else 0.0,
            "prec_mean": round(grp["precision"].mean(), 2),
            "prec_std": round(grp["precision"].std(ddof=1), 2) if len(grp) > 1 else 0.0,
            "rec_mean": round(grp["recall"].mean(), 2),
            "rec_std": round(grp["recall"].std(ddof=1), 2) if len(grp) > 1 else 0.0,
            "f1_mean": round(grp["f1"].mean(), 2),
            "f1_std": round(grp["f1"].std(ddof=1), 2) if len(grp) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def copy_best_checkpoints(results_df, ckpt_dir):
    """For each config, copy the highest-test-accuracy seed's checkpoint to <key>_BEST.pth."""
    best_paths = {}
    for key, grp in results_df.groupby("config", sort=False):
        best = grp.loc[grp["accuracy"].idxmax()]
        src = Path(best["checkpoint"])
        if src.exists():
            dst = ckpt_dir / f"{key}_BEST.pth"
            shutil.copy2(src, dst)
            best_paths[key] = dst
            print(f"BEST {key}: seed {best['seed']} acc {best['accuracy']:.2f}% -> {dst.name}", flush=True)
    return best_paths


LABELS = {
    "resnet50_no_clahe": "ResNet50 [ours]",
    "resnet50_clahe":    "ResNet50 + CLAHE [ours]",
    "v2s_no_clahe":      "EfficientNetV2-S [ours]",
    "v2s_clahe":         "EfficientNetV2-S + CLAHE [ours]",
    "convnext_no_clahe": "ConvNeXt-S [ours]",
    "convnext_clahe":    "ConvNeXt-S + CLAHE [ours]",
}


def build_comparison_table(summary_df):
    def cell(m, s):
        return f"{m:.2f} ± {s:.2f}"
    our_rows = []
    for _, r in summary_df.iterrows():
        our_rows.append({
            "Model": LABELS.get(r["config"], r["config"]),
            "Params (M)": r["params_M"],
            "Accuracy (%)": cell(r["acc_mean"], r["acc_std"]),
            "Precision (%)": cell(r["prec_mean"], r["prec_std"]),
            "Recall (%)": cell(r["rec_mean"], r["rec_std"]),
            "F1 (%)": cell(r["f1_mean"], r["f1_std"]),
        })
    return pd.concat([pd.DataFrame(BENCHMARK_ROWS), pd.DataFrame(our_rows)], ignore_index=True)


def run_gradcam_best(cfg, device, best_paths):
    """Grad-CAM comparisons (no-CLAHE vs CLAHE) for ConvNeXt and EfficientNetV2-S best models."""
    from src.dataset import build_class_map
    from src.gradcam import generate_gradcam_comparison

    samples, class_names = build_class_map(cfg["dataset"]["root"])
    out_dir = Path("results/figures/gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)
    freshness_levels = ["Highly Fresh", "Fresh", "Not Fresh"]

    pairs = [
        ("convnext_small", "convnext_no_clahe", "convnext_clahe", build_convnext_small),
        ("efficientnetv2s", "v2s_no_clahe", "v2s_clahe",
         lambda **kw: build_efficientnetv2s(pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"], **kw)),
    ]
    for backbone, key_no, key_yes, builder in pairs:
        if key_no not in best_paths or key_yes not in best_paths:
            print(f"[gradcam] missing best ckpt for {backbone}, skip", flush=True)
            continue
        try:
            m_no = builder(num_classes=cfg["dataset"]["num_classes"], dropout=cfg["training"]["dropout"]).to(device)
            m_no.load_state_dict(torch.load(best_paths[key_no], map_location=device))
            m_yes = builder(num_classes=cfg["dataset"]["num_classes"], dropout=cfg["training"]["dropout"]).to(device)
            m_yes.load_state_dict(torch.load(best_paths[key_yes], map_location=device))

            generated = []
            for path, label_idx in samples:
                fresh = next((f for f in freshness_levels if f in class_names[label_idx]), None)
                if fresh and fresh not in generated:
                    safe = fresh.lower().replace(" ", "_")
                    generate_gradcam_comparison(
                        image_path=path, model_no_clahe=m_no, model_with_clahe=m_yes,
                        backbone_name=backbone, target_class=label_idx, device=device,
                        save_path=str(out_dir / f"{backbone}_gradcam_{safe}.png"),
                    )
                    generated.append(fresh)
                if len(generated) == 3:
                    break
        except Exception:
            print(f"[gradcam] {backbone} failed:\n{traceback.format_exc()}", flush=True)


def main():
    cfg_base = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  {len(CONFIGS)} configs x {len(SEEDS)} seeds = {len(CONFIGS)*len(SEEDS)} runs", flush=True)

    Path("results").mkdir(exist_ok=True)
    ckpt_dir = Path(cfg_base["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path("results/multiseed_results.csv")

    results = []
    for config in CONFIGS:
        for seed in SEEDS:
            try:
                row = run_one(cfg_base, config, seed, device)
                if row:
                    results.append(row)
                    # Incremental save so partial progress survives any later failure.
                    pd.DataFrame(results).to_csv(raw_path, index=False)
            except Exception:
                print(f"[run failed] {config['key']} seed {seed}:\n{traceback.format_exc()}", flush=True)

    if not results:
        print("No successful runs; aborting aggregation.", flush=True)
        return

    df = pd.DataFrame(results)
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {raw_path}", flush=True)

    summary = aggregate(df)
    summary.to_csv("results/multiseed_summary.csv", index=False)
    print("\n=== SUMMARY (mean +/- std) ===")
    print(summary.to_string(index=False), flush=True)

    best_paths = copy_best_checkpoints(df, ckpt_dir)

    table = build_comparison_table(summary)
    table.to_csv("results/final_comparison_table.csv", index=False)
    print("\n=== COMPARISON TABLE ===")
    print(table.to_string(index=False), flush=True)

    run_gradcam_best(make_optimized_config(cfg_base), device, best_paths)

    Path("results/ALL_DONE").touch()
    print("\n[ALL_DONE] Study complete. Marker written.", flush=True)


if __name__ == "__main__":
    main()
