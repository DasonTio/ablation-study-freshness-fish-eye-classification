#!/usr/bin/env python3
"""Mean-preserving second-order pooling ablation on FFE.

Primary scientific question: does raw bilinear / uncentered second-order moment
pooling improve over GAP on fish-eye freshness classification?

The runner writes both aggregate metrics and per-sample predictions. Freshness
metrics for flat arms are derived from the 24-class prediction, not from the
untrained CORAL head, avoiding the earlier severe-error artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, ".")

from src.dataset import load_config
from src.hierarchical import (
    FRESHNESS_TO_RANK,
    get_hierarchical_dataloaders,
    parse_ffe_class_name,
    predict_coral_rank,
    train_hierarchical,
)
from src.secondorder_model import PooledHierarchicalClassifier
from src.seed import set_seed


ARMS = {
    "A_gap": {
        "pooling": "gap",
        "species_weight": 0.0,
        "freshness_weight": 0.0,
        "role": "same-run GAP baseline",
    },
    "B_raw_bilinear": {
        "pooling": "raw_bilinear",
        "species_weight": 0.0,
        "freshness_weight": 0.0,
        "role": "primary mean-preserving second-order method",
    },
    "C_gap_raw_bilinear": {
        "pooling": "gap_raw_bilinear",
        "species_weight": 0.0,
        "freshness_weight": 0.0,
        "role": "mean plus second-order fusion",
    },
    "D_centered_cov": {
        "pooling": "centered_cov",
        "species_weight": 0.0,
        "freshness_weight": 0.0,
        "role": "control: centered covariance removes mean opacity",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run GAP vs second-order FFE study.")
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024, 7, 2025])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--output-dir", default="results/secondorder")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def class_idx_to_freshness_rank(class_names: list[str]) -> np.ndarray:
    return np.array([
        FRESHNESS_TO_RANK[parse_ffe_class_name(class_name).freshness]
        for class_name in class_names
    ])


def _acc(true, pred) -> float:
    return round(float(accuracy_score(true, pred) * 100), 2)


def compute_secondorder_metrics(
    class_preds: np.ndarray,
    class_labels: np.ndarray,
    true_freshness: np.ndarray,
    class_to_freshness: np.ndarray,
    head_freshness_preds: np.ndarray,
) -> dict[str, float | int]:
    pred_fresh_class = class_to_freshness[class_preds]
    class_abs = np.abs(pred_fresh_class - true_freshness)
    head_abs = np.abs(head_freshness_preds - true_freshness)
    return {
        "class_acc": _acc(class_labels, class_preds),
        "class_precision": round(float(precision_score(class_labels, class_preds, average="macro", zero_division=0) * 100), 2),
        "class_recall": round(float(recall_score(class_labels, class_preds, average="macro", zero_division=0) * 100), 2),
        "class_f1": round(float(f1_score(class_labels, class_preds, average="macro", zero_division=0) * 100), 2),
        "freshness_acc_classderived": _acc(true_freshness, pred_fresh_class),
        "freshness_mae_classderived": round(float(class_abs.mean()), 4),
        "freshness_adjacent_classderived": int((class_abs == 1).sum()),
        "freshness_severe_classderived": int((class_abs >= 2).sum()),
        "freshness_acc_head": _acc(true_freshness, head_freshness_preds),
        "freshness_mae_head": round(float(head_abs.mean()), 4),
        "freshness_adjacent_head": int((head_abs == 1).sum()),
        "freshness_severe_head": int((head_abs >= 2).sum()),
    }


@torch.no_grad()
def evaluate_with_predictions(model, loader, metadata, device, amp=True) -> tuple[dict, pd.DataFrame]:
    model.eval()
    amp = amp and device.type == "cuda"
    class_to_freshness = class_idx_to_freshness_rank(metadata.class_names)
    class_preds, class_labels, true_freshness, head_freshness = [], [], [], []
    paths = []
    samples = loader.dataset.samples
    offset = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)
        pred_cls = outputs["class"].argmax(1).cpu().numpy()
        pred_head = predict_coral_rank(outputs["freshness"]).cpu().numpy()
        class_preds.append(pred_cls)
        class_labels.append(targets["class"].numpy())
        true_freshness.append(targets["freshness"].numpy())
        head_freshness.append(pred_head)
        for i in range(len(pred_cls)):
            paths.append(samples[offset + i].path)
        offset += len(pred_cls)

    class_preds = np.concatenate(class_preds)
    class_labels = np.concatenate(class_labels)
    true_freshness = np.concatenate(true_freshness)
    head_freshness = np.concatenate(head_freshness)
    metrics = compute_secondorder_metrics(
        class_preds=class_preds,
        class_labels=class_labels,
        true_freshness=true_freshness,
        class_to_freshness=class_to_freshness,
        head_freshness_preds=head_freshness,
    )
    pred_fresh_class = class_to_freshness[class_preds]
    pred_df = pd.DataFrame({
        "path": paths,
        "true_class_idx": class_labels,
        "pred_class_idx": class_preds,
        "true_class": [metadata.class_names[i] for i in class_labels],
        "pred_class": [metadata.class_names[i] for i in class_preds],
        "true_freshness_rank": true_freshness,
        "pred_freshness_classderived": pred_fresh_class,
        "pred_freshness_head": head_freshness,
        "class_correct": (class_preds == class_labels).astype(int),
        "severe_classderived": (np.abs(pred_fresh_class - true_freshness) >= 2).astype(int),
        "severe_head": (np.abs(head_freshness - true_freshness) >= 2).astype(int),
    })
    return metrics, pred_df


def mean_std(values: pd.Series) -> dict[str, float]:
    return {
        "mean": round(float(values.mean()), 2),
        "std": round(float(values.std(ddof=1)), 2) if len(values) > 1 else 0.0,
    }


def write_progress(out_dir: Path, rows: list[dict], total_runs: int, started_at: float):
    elapsed = time.time() - started_at
    completed = len(rows)
    avg = elapsed / completed if completed else None
    remaining = total_runs - completed
    payload = {
        "completed_runs": completed,
        "total_runs": total_runs,
        "elapsed_hours": round(elapsed / 3600, 2),
        "estimated_remaining_hours": round((avg * remaining) / 3600, 2) if avg else None,
        "rows": rows,
    }
    (out_dir / "secondorder_progress.json").write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    unknown = [arm for arm in args.arms if arm not in ARMS]
    if unknown:
        raise ValueError(f"Unknown arm(s): {unknown}. Known arms: {list(ARMS)}")

    cfg = load_config()
    cfg["dataset"]["train_split"] = 0.64
    cfg["dataset"]["val_split"] = 0.16
    cfg["dataset"]["test_split"] = 0.20

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    marker_path = out_dir / "SECONDORDER_STUDY_DONE"
    failed_path = out_dir / "SECONDORDER_FAILED"
    for marker in (marker_path, failed_path):
        if marker.exists():
            marker.unlink()

    result_path = out_dir / "secondorder_results.csv"
    pred_path = out_dir / "secondorder_predictions.csv"
    summary_path = out_dir / "secondorder_summary.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_runs = len(args.arms) * len(args.seeds)
    print(
        f"Device={device} backbone={args.backbone} arms={args.arms} seeds={args.seeds} "
        f"split=64/16/20 epochs={args.epochs} patience={args.patience} total_runs={total_runs}",
        flush=True,
    )

    rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    best_by_arm: dict[str, tuple[float, dict]] = {}
    started_at = time.time()
    try:
        for arm in args.arms:
            spec = ARMS[arm]
            for seed in args.seeds:
                run_started = time.time()
                run_name = f"{arm}_seed{seed}"
                print(f"\n{'=' * 80}\n{run_name}: {spec['role']}\n{'=' * 80}", flush=True)
                set_seed(seed)
                train_loader, val_loader, test_loader, metadata = get_hierarchical_dataloaders(
                    cfg,
                    split_seed=args.split_seed,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_samples=args.max_samples,
                )
                model = PooledHierarchicalClassifier(
                    backbone_tag=args.backbone,
                    num_classes=metadata.num_classes,
                    num_species=metadata.num_species,
                    pooling=spec["pooling"],
                    proj_dim=args.proj_dim,
                    pretrained=not args.no_pretrained,
                    dropout=args.dropout,
                    drop_path=args.drop_path,
                )
                if hasattr(torch, "compile"):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        print(f"[compile] torch.compile enabled for {run_name}.", flush=True)
                    except Exception as exc:
                        print(f"[compile] torch.compile failed ({exc}); using eager.", flush=True)
                state, val_metrics = train_hierarchical(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing,
                    species_weight=spec["species_weight"],
                    freshness_weight=spec["freshness_weight"],
                    patience=args.patience,
                    amp=args.amp,
                )
                model.load_state_dict(state)
                model = model.to(device)
                metrics, pred_df = evaluate_with_predictions(model, test_loader, metadata, device, amp=args.amp)
                pred_df.insert(0, "seed", seed)
                pred_df.insert(0, "arm", arm)
                pred_frames.append(pred_df)

                val_acc = float(val_metrics.get("class_acc", 0.0))
                elapsed_h = (time.time() - run_started) / 3600
                row = {
                    "arm": arm,
                    "pooling": spec["pooling"],
                    "seed": seed,
                    "best_val_class_acc": val_acc,
                    "species_weight": spec["species_weight"],
                    "freshness_weight": spec["freshness_weight"],
                    "run_hours": round(elapsed_h, 3),
                    **{f"test_{key}": value for key, value in metrics.items()},
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(result_path, index=False)
                pd.concat(pred_frames, ignore_index=True).to_csv(pred_path, index=False)
                write_progress(out_dir, rows, total_runs, started_at)
                print(f"RESULT {run_name}: {row}", flush=True)

                if arm not in best_by_arm or val_acc > best_by_arm[arm][0]:
                    best_by_arm[arm] = (val_acc, copy.deepcopy(state))
                    torch.save(state, ckpt_dir / f"{arm}_BEST.pth")

        df = pd.DataFrame(rows)
        summary_rows = []
        for arm, g in df.groupby("arm", sort=False):
            acc = mean_std(g["test_class_acc"])
            fresh = mean_std(g["test_freshness_acc_classderived"])
            summary_rows.append({
                "arm": arm,
                "pooling": ARMS[arm]["pooling"],
                "n": len(g),
                "class_acc_mean": acc["mean"],
                "class_acc_std": acc["std"],
                "freshness_acc_classderived_mean": fresh["mean"],
                "freshness_acc_classderived_std": fresh["std"],
                "freshness_mae_classderived_mean": round(float(g["test_freshness_mae_classderived"].mean()), 4),
                "severe_classderived_mean": round(float(g["test_freshness_severe_classderived"].mean()), 2),
                "run_hours_total": round(float(g["run_hours"].sum()), 2),
            })
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        marker_path.write_text(json.dumps({
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "runs": len(rows),
            "total_hours": round((time.time() - started_at) / 3600, 2),
        }, indent=2))
        print("\n=== SECONDORDER SUMMARY ===", flush=True)
        print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
        print(f"\n[SECONDORDER_STUDY_DONE] Wrote {marker_path}", flush=True)
    except Exception:
        failed_path.write_text(traceback.format_exc())
        print(f"[SECONDORDER_FAILED]\n{traceback.format_exc()}", flush=True)
        raise


if __name__ == "__main__":
    main()
