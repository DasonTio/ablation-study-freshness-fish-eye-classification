#!/usr/bin/env python3
"""Build a 3x3 stage-wise Grad-CAM panel for the JuTISI manuscript.

Columns are representative trained arms from the evidence chain:
CLAHE preprocessing, second-order pooling, and the final GAP Swin-Tiny model.
Rows are matched FFE samples from the three ordered freshness levels.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLAHETransform, load_config
from src.hierarchical import build_hierarchical_metadata
from src.recipe import IMAGENET_MEAN, IMAGENET_STD
from src.secondorder_model import PooledHierarchicalClassifier

BACKBONE_TAG = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
OUT_PATH = ROOT / "results" / "figures" / "jutisi" / "figure8_stage_xai_3x3.png"

HANDPICKED = [
    ("Highly Fresh", ROOT / "data/FFE/Johnius Trachycephalus - Highly Fresh/IMG_20191014_071619.jpg"),
    ("Fresh", ROOT / "data/FFE/Rastrelliger Faughni - Fresh/IMG_20191016_064941.jpg"),
    ("Not Fresh", ROOT / "data/FFE/Chanos Chanos - Not Fresh/IMG_20191004_062950.jpg"),
]


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    checkpoint: str
    family: str
    pooling: str | None = None
    use_clahe: bool = False


def stage_specs() -> list[StageSpec]:
    return [
        StageSpec(
            key="clahe",
            title="CLAHE Stage\nEfficientNetV2-S + CLAHE",
            checkpoint="results/checkpoints/exp_D_v2s_clahe_proposed_best.pth",
            family="efficientnetv2s",
            use_clahe=True,
        ),
        StageSpec(
            key="second_order",
            title="Second-Order Candidate\nSwin-Tiny + raw bilinear",
            checkpoint="results/secondorder_merged/checkpoints/B_raw_bilinear_BEST.pth",
            family="pooled_swin",
            pooling="raw_bilinear",
        ),
        StageSpec(
            key="final_gap",
            title="Final GAP Swin-Tiny\n24-class head",
            checkpoint="results/secondorder_merged/checkpoints/A_gap_BEST.pth",
            family="pooled_swin",
            pooling="gap",
        ),
    ]


def strip_compile_prefix(state: dict) -> dict:
    return {
        key.replace("_orig_mod.", "", 1) if key.startswith("_orig_mod.") else key: value
        for key, value in state.items()
    }


class ClassOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["class"]
        return out


def swin_reshape(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        b, tokens, channels = tensor.shape
        side = int(tokens ** 0.5)
        return tensor.reshape(b, side, side, channels).permute(0, 3, 1, 2)
    return tensor


def build_efficientnetv2s_for_xai(num_classes: int, dropout: float) -> nn.Module:
    model = timm.create_model(
        "tf_efficientnetv2_s.in21k",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.num_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )
    return model


def load_stage_model(spec: StageSpec, metadata, cfg, device: torch.device) -> nn.Module:
    checkpoint = ROOT / spec.checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = strip_compile_prefix(state)
    if spec.family == "efficientnetv2s":
        model = build_efficientnetv2s_for_xai(
            num_classes=metadata.num_classes,
            dropout=cfg["training"]["dropout"],
        )
    elif spec.family == "pooled_swin":
        if spec.pooling is None:
            raise ValueError(f"Missing pooling mode for {spec.key}")
        model = PooledHierarchicalClassifier(
            backbone_tag=BACKBONE_TAG,
            num_classes=metadata.num_classes,
            num_species=metadata.num_species,
            pooling=spec.pooling,
            proj_dim=128,
            pretrained=False,
            dropout=0.3,
            drop_path=0.1,
        )
    else:
        raise ValueError(f"Unknown stage family: {spec.family}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def target_layer_and_reshape(spec: StageSpec, model: nn.Module):
    if spec.family == "efficientnetv2s":
        return [model.blocks[-1]], None
    if spec.family == "pooled_swin":
        return [model.backbone.layers[-1].blocks[-1].norm2], swin_reshape
    raise ValueError(f"Unknown stage family: {spec.family}")


def preprocess(path: Path, image_size: int, use_clahe: bool) -> tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB")
    if use_clahe:
        img = CLAHETransform()(img)
    img = img.resize((image_size, image_size))
    rgb = np.asarray(img).astype(np.float32) / 255.0
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tensor = norm(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    return tensor, rgb


def class_short(name: str) -> str:
    species, freshness = [part.strip() for part in name.split(" - ", 1)]
    species_short = " ".join(species.split()[:2])
    return f"{species_short} / {freshness}"


def gradcam_overlay(spec: StageSpec, model: nn.Module, path: Path, metadata, image_size: int, device: torch.device):
    tensor, rgb = preprocess(path, image_size, use_clahe=spec.use_clahe)
    wrapper = ClassOutputWrapper(model).eval()
    with torch.no_grad():
        logits = wrapper(tensor.to(device))
        pred = int(logits.argmax(1).item())
        conf = float(logits.softmax(1)[0, pred].item())
    layers, reshape = target_layer_and_reshape(spec, model)
    with GradCAM(model=wrapper, target_layers=layers, reshape_transform=reshape) as cam:
        mask = cam(input_tensor=tensor.to(device), targets=[ClassifierOutputTarget(pred)])[0]
    overlay = show_cam_on_image(
        rgb,
        mask,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=0.55,
    )
    return overlay, class_short(metadata.class_names[pred]), conf


def build_stage_xai_figure(output: Path = OUT_PATH, skip_if_exists: bool = False) -> Path:
    if skip_if_exists and output.exists():
        print(f"skip existing {output}")
        return output

    cfg = load_config()
    _, metadata = build_hierarchical_metadata(cfg["dataset"]["root"])
    image_size = int(cfg["dataset"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = stage_specs()
    models = [load_stage_model(spec, metadata, cfg, device) for spec in specs]

    entries = [(label, path) for label, path in HANDPICKED if path.exists()]
    if len(entries) != 3:
        missing = [str(path) for _, path in HANDPICKED if not path.exists()]
        raise FileNotFoundError(f"Expected 3 handpicked images; missing: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 8.4))
    for col, spec in enumerate(specs):
        axes[0, col].set_title(spec.title, fontsize=9.5, fontweight="bold", pad=8)

    for row, (freshness, path) in enumerate(entries):
        for col, (spec, model) in enumerate(zip(specs, models)):
            overlay, pred, conf = gradcam_overlay(spec, model, path, metadata, image_size, device)
            ax = axes[row, col]
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(freshness, fontsize=9, fontweight="bold")
            ax.set_xlabel(f"Pred: {pred}\nconf={conf:.2f}", fontsize=7)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.6)

    fig.suptitle(
        "Stage-Wise xAI: Rejected Add-Ons vs Final GAP Swin-Tiny",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-wise 3x3 Grad-CAM figure.")
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--skip-if-exists", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_stage_xai_figure(Path(args.output), skip_if_exists=args.skip_if_exists)


if __name__ == "__main__":
    main()
