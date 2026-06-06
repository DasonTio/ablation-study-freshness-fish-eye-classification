"""
Run evaluation on all saved checkpoints using local dataset.

Works on macOS (CPU or MPS on Apple Silicon). No GPU required.
Run after results/checkpoints/ has been synced from the server.

Usage:
    python scripts/evaluate_local.py
    python scripts/evaluate_local.py --checkpoint results/checkpoints/final_D_v2s_clahe_proposed_best.pth
"""
import sys
sys.path.insert(0, ".")

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dataset import get_dataloaders, load_config
from src.evaluate import evaluate_checkpoint
from src.models import build_convnext_small, build_efficientnetv2s, build_resnet50


# ── Device selection ──────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Checkpoint → model/CLAHE inference from filename ─────────────────────────
def parse_checkpoint(path: Path, cfg: dict):
    name = path.stem.lower()
    num_classes = cfg["dataset"]["num_classes"]
    dropout = cfg["training"]["dropout"]

    if "resnet50" in name:
        model = build_resnet50(num_classes=num_classes, pretrained=False, dropout=dropout)
        backbone = "resnet50"
    elif "convnext" in name:
        model = build_convnext_small(num_classes=num_classes, dropout=dropout)
        backbone = "convnext_small"
    elif "v2s" in name or "efficientnetv2s" in name:
        model = build_efficientnetv2s(
            num_classes=num_classes,
            pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"],
            dropout=dropout,
        )
        backbone = "efficientnetv2s"
    else:
        print(f"  [skip] Cannot infer model from filename: {path.name}")
        return None, None, None

    # "no_clahe" must be checked before "clahe"
    use_clahe = ("clahe" in name) and ("no_clahe" not in name)
    return model, backbone, use_clahe


def evaluate_one(ckpt_path: Path, cfg: dict, device: torch.device) -> dict | None:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path.name}")

    model, backbone, use_clahe = parse_checkpoint(ckpt_path, cfg)
    if model is None:
        return None

    print(f"  backbone={backbone}  clahe={use_clahe}  device={device}")

    _, _, test_loader, class_names = get_dataloaders(cfg, use_clahe=use_clahe)
    model = model.to(device)

    metrics = evaluate_checkpoint(
        model=model,
        checkpoint_path=str(ckpt_path),
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        figures_dir=cfg["paths"]["figures"],
        experiment_name=ckpt_path.stem,
    )

    return {
        "checkpoint": ckpt_path.stem,
        "backbone": backbone,
        "clahe": use_clahe,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate a single checkpoint. Omit to evaluate all in results/checkpoints/.",
    )
    args = parser.parse_args()

    cfg = load_config()
    device = get_device()
    print(f"Device: {device}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    if not ckpt_dir.exists():
        sys.exit(f"Checkpoint directory not found: {ckpt_dir}\nRun rsync from server first.")

    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        checkpoints = sorted(ckpt_dir.glob("*_best.pth"))
        if not checkpoints:
            sys.exit(f"No *_best.pth files in {ckpt_dir}. Sync from server first.")
        print(f"Found {len(checkpoints)} checkpoints.")

    rows = []
    for ckpt in checkpoints:
        result = evaluate_one(ckpt, cfg, device)
        if result:
            rows.append(result)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    print("\n\n=== LOCAL EVALUATION SUMMARY ===")
    print(df.to_string(index=False))

    out = Path("results/local_evaluation_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
