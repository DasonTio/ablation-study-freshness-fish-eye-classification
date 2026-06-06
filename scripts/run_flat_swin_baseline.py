"""
Flat 24-class Swin-Tiny baseline — identical protocol to the hierarchical ordinal study
(same backbone, splits, optimizer, schedule, augmentation, epochs, patience) with
species_weight=freshness_weight=0 so the loss reduces to pure 24-class CrossEntropy.

Controls for all training factors; only the multi-task objective differs.
Run BEFORE writing the ablation section — this is the apples-to-apples comparison
that justifies the hierarchical ordinal accuracy gain.
"""
import argparse
import copy
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, ".")

from src.dataset import load_config
from src.hierarchical import (
    HierarchicalClassifier,
    evaluate_hierarchical,
    get_hierarchical_dataloaders,
    train_hierarchical,
)
from src.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Flat 24-class Swin-Tiny baseline study.")
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k")
    parser.add_argument("--backbone-key", default="swin_tiny_flat")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def mean_std(series):
    if len(series) <= 1:
        return f"{series.mean():.2f}"
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"


def main():
    args = parse_args()
    cfg = load_config()
    if args.batch_size is None:
        args.batch_size = cfg["training"]["batch_size"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir or cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "flat_swin_baseline_results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device} backbone={args.backbone} seeds={args.seeds} "
        f"epochs={args.epochs} [FLAT 24-class baseline]",
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
            )
            model = HierarchicalClassifier(
                backbone_tag=args.backbone,
                num_classes=metadata.num_classes,
                num_species=metadata.num_species,
                pretrained=True,
                dropout=args.dropout,
                drop_path=args.drop_path,
            )
            # species_weight=freshness_weight=0 → loss = CrossEntropy(class) only
            state, val_metrics = train_hierarchical(
                model,
                train_loader,
                val_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                species_weight=0.0,
                freshness_weight=0.0,
                patience=args.patience,
                amp=args.amp,
            )
            model.load_state_dict(state)
            model = model.to(device)
            test_metrics = evaluate_hierarchical(model, test_loader, device, amp=args.amp)

            row = {
                "backbone": args.backbone_key,
                "seed": seed,
                "best_val_class_acc": val_metrics.get("class_acc", 0.0),
                **{f"test_{key}": value for key, value in test_metrics.items()},
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
        raise RuntimeError("No flat baseline runs completed.")

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)

    print("\n=== FLAT BASELINE SUMMARY ===", flush=True)
    print(f"24-class test acc: {mean_std(df['test_class_acc'])}%", flush=True)
    print(f"Results → {result_path}", flush=True)

    if best_state is not None:
        ckpt_path = ckpt_dir / f"flat_{args.backbone_key}_BEST.pth"
        torch.save(best_state, ckpt_path)
        print(f"Best checkpoint → {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
