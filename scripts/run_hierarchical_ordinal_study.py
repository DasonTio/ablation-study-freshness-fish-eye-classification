"""
Hierarchical ordinal FFE study.

Default target: one strong backbone x 3 training seeds within a 6-7 hour Vast.ai
window. The benchmark target remains 24-class accuracy, while species and ordered
freshness labels provide auxiliary supervision.
"""
import argparse
import copy
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")

from src.dataset import load_config
from src.evaluate import compute_metrics
from src.hierarchical import (
    HierarchicalClassifier,
    evaluate_hierarchical,
    extract_hierarchical_features,
    get_hierarchical_dataloaders,
    train_hierarchical,
)
from src.seed import set_seed


BENCHMARK_ROWS = [
    {"Model": "ResNet50 [Prasetyo 2022, benchmark]", "Accuracy (%)": "78.82"},
    {"Model": "Swin-T + ExtraTrees + LGBM [Hoang 2025, SOTA]", "Accuracy (%)": "85.99"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run hierarchical ordinal FFE study.")
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k")
    parser.add_argument("--backbone-key", default="swin_tiny_hier_ord")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--species-weight", type=float, default=0.3)
    parser.add_argument("--freshness-weight", type=float, default=0.7)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def hybrid_head_acc(model, train_loader, val_loader, test_loader, device, amp=True):
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.preprocessing import StandardScaler

        Xtr, ytr = extract_hierarchical_features(model, train_loader, device, amp=amp)
        Xva, yva = extract_hierarchical_features(model, val_loader, device, amp=amp)
        Xte, yte = extract_hierarchical_features(model, test_loader, device, amp=amp)
        Xtr = np.concatenate([Xtr, Xva])
        ytr = np.concatenate([ytr, yva])

        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

        out = {}
        et = ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=0)
        et.fit(Xtr, ytr)
        out["hybrid_extratrees"] = compute_metrics(et.predict(Xte), yte)["accuracy"]
        try:
            from lightgbm import LGBMClassifier

            lgbm = LGBMClassifier(n_estimators=600, n_jobs=-1, random_state=0, verbose=-1)
            lgbm.fit(Xtr, ytr)
            out["hybrid_lgbm"] = compute_metrics(lgbm.predict(Xte), yte)["accuracy"]
        except Exception:
            print("[hybrid] LightGBM unavailable; ExtraTrees completed.", flush=True)
        return out
    except Exception:
        print(f"[hybrid] failed:\n{traceback.format_exc()}", flush=True)
        return {}


def mean_std(series):
    if len(series) <= 1:
        return f"{series.mean():.2f}"
    return f"{series.mean():.2f} +/- {series.std(ddof=1):.2f}"


def main():
    args = parse_args()
    cfg = load_config()
    if args.batch_size is None:
        args.batch_size = cfg["training"]["batch_size"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir or cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "hierarchical_ordinal_results.csv"
    summary_path = out_dir / "hierarchical_ordinal_summary.csv"
    table_path = out_dir / "hierarchical_ordinal_comparison_table.csv"
    marker_path = out_dir / "HIER_ORD_DONE"
    if marker_path.exists():
        marker_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device} backbone={args.backbone} seeds={args.seeds} "
        f"epochs={args.epochs} batch={args.batch_size}",
        flush=True,
    )

    rows = []
    best_state, best_val, best_seed = None, -1.0, None
    for seed in args.seeds:
        run_name = f"{args.backbone_key}_seed{seed}"
        print(f"\n{'=' * 72}\n{run_name}\n{'=' * 72}", flush=True)
        try:
            set_seed(seed)
            train_loader, val_loader, test_loader, metadata = get_hierarchical_dataloaders(
                cfg,
                split_seed=args.split_seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
            )
            model = HierarchicalClassifier(
                backbone_tag=args.backbone,
                num_classes=metadata.num_classes,
                num_species=metadata.num_species,
                pretrained=not args.no_pretrained,
                dropout=args.dropout,
                drop_path=args.drop_path,
            )
            state, val_metrics = train_hierarchical(
                model,
                train_loader,
                val_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                species_weight=args.species_weight,
                freshness_weight=args.freshness_weight,
                patience=args.patience,
                amp=args.amp,
            )
            model.load_state_dict(state)
            model = model.to(device)
            test_metrics = evaluate_hierarchical(model, test_loader, device, amp=args.amp)
            hyb = {} if args.no_hybrid else hybrid_head_acc(model, train_loader, val_loader, test_loader, device, amp=args.amp)

            row = {
                "backbone": args.backbone_key,
                "seed": seed,
                "best_val_class_acc": val_metrics.get("class_acc", 0.0),
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **hyb,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(f"RESULT {run_name}: {row}", flush=True)

            if val_metrics.get("class_acc", 0.0) > best_val:
                best_val = val_metrics.get("class_acc", 0.0)
                best_seed = seed
                best_state = copy.deepcopy(state)
        except Exception:
            print(f"[run failed] {run_name}:\n{traceback.format_exc()}", flush=True)

    if not rows:
        raise RuntimeError("No hierarchical ordinal runs completed.")

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    summary = {
        "backbone": args.backbone_key,
        "n": len(df),
        "class_acc_mean": round(df["test_class_acc"].mean(), 2),
        "class_acc_std": round(df["test_class_acc"].std(ddof=1), 2) if len(df) > 1 else 0.0,
        "freshness_acc_mean": round(df["test_freshness_acc"].mean(), 2),
        "freshness_mae_mean": round(df["test_freshness_mae"].mean(), 4),
        "best_seed": best_seed,
        "best_val_class_acc": best_val,
    }
    if "hybrid_extratrees" in df:
        summary["hybrid_et_max"] = round(df["hybrid_extratrees"].max(), 2)
    if "hybrid_lgbm" in df:
        summary["hybrid_lgbm_max"] = round(df["hybrid_lgbm"].max(), 2)

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    table_rows = BENCHMARK_ROWS + [
        {"Model": f"{args.backbone_key} 24-class (ours)", "Accuracy (%)": mean_std(df["test_class_acc"])},
        {"Model": f"{args.backbone_key} freshness-only (ours)", "Accuracy (%)": mean_std(df["test_freshness_acc"])},
    ]
    if "hybrid_extratrees" in df:
        table_rows.append({"Model": f"{args.backbone_key} + ExtraTrees hybrid (ours)", "Accuracy (%)": f"{df['hybrid_extratrees'].max():.2f}"})
    if "hybrid_lgbm" in df:
        table_rows.append({"Model": f"{args.backbone_key} + LightGBM hybrid (ours)", "Accuracy (%)": f"{df['hybrid_lgbm'].max():.2f}"})
    pd.DataFrame(table_rows).to_csv(table_path, index=False)

    if best_state is not None:
        torch.save(best_state, ckpt_dir / f"hierarchical_ordinal_{args.backbone_key}_BEST.pth")

    marker_path.touch()
    print("\n=== SUMMARY ===", flush=True)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print(f"\n[HIER_ORD_DONE] Wrote {marker_path}", flush=True)


if __name__ == "__main__":
    main()
