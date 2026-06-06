#!/usr/bin/env python3
"""Build SEPARATED Grad-CAM panels for the JuTISI manuscript.

Two 3x3 figures, both with the same evidence-chain columns
(CLAHE EfficientNetV2-S, second-order Swin-Tiny, final GAP Swin-Tiny) and the
three ordered freshness levels as rows:

  * figure8_xai_correct_3x3.png   - one correctly classified example per cell
  * figure9_xai_incorrect_3x3.png - one misclassified example per cell

Separating correct from incorrect makes the visual contrast explicit: where each
model attends when it succeeds versus where it attends when it fails. Examples are
drawn from a held-out split and the chosen file paths are written to a manifest for
reproducibility. The figures are qualitative support; the tables carry the evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_class_map, load_config
from src.hierarchical import build_hierarchical_metadata
from scripts.make_stage_xai_figure import (
    ClassOutputWrapper,
    StageSpec,
    load_stage_model,
    preprocess,
    stage_specs,
    target_layer_and_reshape,
)

FRESH_LEVELS = ["Highly Fresh", "Fresh", "Not Fresh"]
# Each model's own held-out ratio so chosen examples are genuinely held out.
SPLIT_RATIOS = {"clahe": (0.60, 0.20), "second_order": (0.64, 0.16), "final_gap": (0.64, 0.16)}
SPLIT_SEED = 42
COL_TITLES = {
    "clahe": "CLAHE stage\nEfficientNetV2-S + CLAHE",
    "second_order": "Second-order stage\nSwin-Tiny + raw bilinear",
    "final_gap": "Final model\nGAP Swin-Tiny",
}
OUT_CORRECT = ROOT / "results" / "figures" / "jutisi" / "figure8_xai_correct_3x3.png"
OUT_INCORRECT = ROOT / "results" / "figures" / "jutisi" / "figure9_xai_incorrect_3x3.png"
OUT_MANIFEST = ROOT / "results" / "figures" / "jutisi" / "xai_split_manifest.json"


def freshness_of(class_name: str) -> str:
    return class_name.split(" - ")[-1].strip()


def species_short(class_name: str) -> str:
    return " ".join(class_name.split(" - ")[0].split()[:2])


def test_pool(root: str, train_split: float, val_split: float) -> list[tuple[str, int]]:
    samples, _ = build_class_map(root)
    n = len(samples)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    _, _, test_s = torch.utils.data.random_split(samples, [n_train, n_val, n_test], generator=gen)
    return [samples[i] for i in test_s.indices]


def sweep_predictions(spec, model, items, class_names, image_size, device):
    """Predict 24-class for every pool image; return per-record dicts."""
    wrapper = ClassOutputWrapper(model).to(device).eval()
    records = []
    with torch.no_grad():
        for path, label in items:
            tensor, _ = preprocess(Path(path), image_size, use_clahe=spec.use_clahe)
            logits = wrapper(tensor.to(device))
            pred = int(logits.argmax(1).item())
            conf = float(logits.softmax(1)[0, pred].item())
            records.append(
                {
                    "path": path,
                    "true": label,
                    "pred": pred,
                    "conf": conf,
                    "correct": pred == label,
                    "true_fresh": freshness_of(class_names[label]),
                }
            )
    return records


def pick_cells(records):
    """For each freshness level pick the most confident correct and incorrect record."""
    correct, incorrect = {}, {}
    for fresh in FRESH_LEVELS:
        c = [r for r in records if r["correct"] and r["true_fresh"] == fresh]
        w = [r for r in records if not r["correct"] and r["true_fresh"] == fresh]
        correct[fresh] = max(c, key=lambda r: r["conf"]) if c else None
        # Most confident error is the most instructive failure to show.
        incorrect[fresh] = max(w, key=lambda r: r["conf"]) if w else None
    return correct, incorrect


def gradcam_overlay(spec, model, path, class_for_cam, image_size, device):
    tensor, rgb = preprocess(Path(path), image_size, use_clahe=spec.use_clahe)
    wrapper = ClassOutputWrapper(model).to(device).eval()
    layers, reshape = target_layer_and_reshape(spec, model)
    with GradCAM(model=wrapper, target_layers=layers, reshape_transform=reshape) as cam:
        mask = cam(input_tensor=tensor.to(device), targets=[ClassifierOutputTarget(class_for_cam)])[0]
    return show_cam_on_image(rgb, mask, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.55)


def render_panel(kind, picks_by_spec, specs, class_names, image_size, cam_device, output, suptitle):
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 9.0))
    for col, spec in enumerate(specs):
        axes[0, col].set_title(COL_TITLES[spec.key], fontsize=9.5, fontweight="bold", pad=8)
    for row, fresh in enumerate(FRESH_LEVELS):
        for col, spec in enumerate(specs):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(fresh, fontsize=10, fontweight="bold")
            rec = picks_by_spec[spec.key][kind][fresh]
            if rec is None:
                ax.text(0.5, 0.5, "no example", ha="center", va="center", fontsize=8)
                ax.set_facecolor("#f0f0f0")
                continue
            overlay = gradcam_overlay(spec, picks_by_spec[spec.key]["model"], rec["path"],
                                      rec["pred"], image_size, cam_device)
            ax.imshow(overlay)
            true_f = freshness_of(class_names[rec["true"]])
            pred_f = freshness_of(class_names[rec["pred"]])
            if kind == "correct":
                ax.set_xlabel(f"{species_short(class_names[rec['true']])} / {true_f}\nconf {rec['conf']:.2f}",
                              fontsize=7, color="#0a7d2c")
            else:
                ax.set_xlabel(f"true {true_f} -> pred {pred_f}\nconf {rec['conf']:.2f}",
                              fontsize=7, color="#b00000")
            edge = "#0a7d2c" if kind == "correct" else "#b00000"
            for spine in ax.spines.values():
                spine.set_color(edge); spine.set_linewidth(1.2)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def build(skip_if_exists: bool = False):
    if skip_if_exists and OUT_CORRECT.exists() and OUT_INCORRECT.exists():
        print("skip existing split xai figures")
        return
    cfg = load_config()
    root = cfg["dataset"]["root"]
    image_size = int(cfg["dataset"]["image_size"])
    _, class_names = build_class_map(root)
    _, metadata = build_hierarchical_metadata(root)

    sweep_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cam_device = torch.device("cpu")  # grad-cam is most robust on CPU
    specs = stage_specs()

    picks_by_spec, manifest = {}, {}
    for spec in specs:
        model = load_stage_model(spec, metadata, cfg, sweep_device)
        pool = test_pool(root, *SPLIT_RATIOS[spec.key])
        print(f"[{spec.key}] sweeping {len(pool)} held-out images on {sweep_device}...")
        try:
            records = sweep_predictions(spec, model, pool, class_names, image_size, sweep_device)
        except Exception as exc:  # MPS fallback
            print(f"  {sweep_device} failed ({exc}); retrying on cpu")
            model = model.to("cpu")
            records = sweep_predictions(spec, model, pool, class_names, image_size, torch.device("cpu"))
        acc = 100.0 * sum(r["correct"] for r in records) / len(records)
        print(f"  pool accuracy {acc:.2f}%")
        correct, incorrect = pick_cells(records)
        model = model.to(cam_device)
        picks_by_spec[spec.key] = {"model": model, "correct": correct, "incorrect": incorrect}
        manifest[spec.key] = {
            "pool_accuracy": round(acc, 2),
            "correct": {f: (correct[f]["path"] if correct[f] else None) for f in FRESH_LEVELS},
            "incorrect": {
                f: ({"path": incorrect[f]["path"],
                     "true": class_names[incorrect[f]["true"]],
                     "pred": class_names[incorrect[f]["pred"]]} if incorrect[f] else None)
                for f in FRESH_LEVELS
            },
        }

    render_panel("correct", picks_by_spec, specs, class_names, image_size, cam_device,
                 OUT_CORRECT, "Correctly Classified Samples: Grad-CAM Across the Model Chain")
    render_panel("incorrect", picks_by_spec, specs, class_names, image_size, cam_device,
                 OUT_INCORRECT, "Misclassified Samples: Grad-CAM Across the Model Chain")
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT_MANIFEST}")


def main():
    parser = argparse.ArgumentParser(description="Generate separated correct/incorrect Grad-CAM 3x3 figures.")
    parser.add_argument("--skip-if-exists", action="store_true")
    build(parser.parse_args().skip_if_exists)


if __name__ == "__main__":
    main()
