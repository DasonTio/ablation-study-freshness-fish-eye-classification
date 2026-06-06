from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

from src.dataset import CLAHETransform
from src.hierarchical import (
    HierarchicalClassifier,
    RANK_TO_FRESHNESS,
    build_hierarchical_metadata,
    predict_coral_rank,
)
from src.recipe import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    checkpoint_path: str
    architecture: str
    kind: str = "flat"
    use_clahe: bool = False
    xai_supported: bool = True


class HierarchicalClassWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["class"]


MODEL_SPECS = [
    ModelSpec("resnet50_no_clahe", "ResNet50", "results/checkpoints/resnet50_no_clahe_BEST.pth", "resnet50"),
    ModelSpec("resnet50_clahe", "ResNet50 + CLAHE", "results/checkpoints/resnet50_clahe_BEST.pth", "resnet50", use_clahe=True),
    ModelSpec("v2s_no_clahe", "EfficientNetV2-S", "results/checkpoints/v2s_no_clahe_BEST.pth", "efficientnetv2s"),
    ModelSpec("v2s_clahe", "EfficientNetV2-S + CLAHE", "results/checkpoints/v2s_clahe_BEST.pth", "efficientnetv2s", use_clahe=True),
    ModelSpec("convnext_no_clahe", "ConvNeXt-Small", "results/checkpoints/convnext_no_clahe_BEST.pth", "convnext_small"),
    ModelSpec("convnext_clahe", "ConvNeXt-Small + CLAHE", "results/checkpoints/convnext_clahe_BEST.pth", "convnext_small", use_clahe=True),
    ModelSpec("recipe_swin_tiny", "Recipe Swin-Tiny", "results/checkpoints/recipe_swin_tiny_BEST.pth", "recipe_swin_tiny"),
    ModelSpec("recipe_convnext_tiny", "Recipe ConvNeXt-Tiny", "results/checkpoints/recipe_convnext_tiny_BEST.pth", "recipe_convnext_tiny"),
    ModelSpec("flat_swin_tiny", "Flat Swin-Tiny (ablation baseline)", "results/checkpoints/flat_swin_tiny_flat_BEST.pth", "hierarchical_swin_tiny", kind="hierarchical"),
    ModelSpec("hierarchical_swin_tiny", "Hierarchical Ordinal Swin-Tiny", "results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth", "hierarchical_swin_tiny", kind="hierarchical"),
]


def available_model_specs() -> list[ModelSpec]:
    return [spec for spec in MODEL_SPECS if Path(spec.checkpoint_path).exists()]


def get_model_spec(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown model key: {key}")


def _mlp_head(in_features: int, num_classes: int, dropout: float):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def build_flat_model(architecture: str, num_classes: int):
    if architecture == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = _mlp_head(model.fc.in_features, num_classes, dropout=0.4)
        return model
    if architecture == "efficientnetv2s":
        model = timm.create_model("tf_efficientnetv2_s.in21k", pretrained=False, num_classes=0, global_pool="avg")
        model.classifier = _mlp_head(model.num_features, num_classes, dropout=0.4)
        return model
    if architecture == "convnext_small":
        model = timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False)
        model.head.fc = _mlp_head(model.head.fc.in_features, num_classes, dropout=0.4)
        return model
    if architecture == "recipe_swin_tiny":
        return timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.1,
        )
    if architecture == "recipe_convnext_tiny":
        return timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.1,
        )
    raise ValueError(f"Unsupported flat architecture: {architecture}")


def load_model(spec: ModelSpec, device: torch.device, dataset_root="data/FFE"):
    _, metadata = build_hierarchical_metadata(dataset_root)
    if spec.kind == "hierarchical":
        model = HierarchicalClassifier(
            "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
            metadata.num_classes,
            metadata.num_species,
            pretrained=False,
            dropout=0.3,
            drop_path=0.1,
        )
    else:
        model = build_flat_model(spec.architecture, metadata.num_classes)

    state = torch.load(spec.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, metadata


def preprocess_pil(image: Image.Image, use_clahe: bool, image_size: int = 224):
    image = image.convert("RGB")
    if use_clahe:
        image = CLAHETransform()(image)
    resized = image.resize((image_size, image_size))
    rgb_float = np.asarray(resized).astype(np.float32) / 255.0
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tensor = normalize(torch.from_numpy(rgb_float).permute(2, 0, 1)).unsqueeze(0)
    return tensor, rgb_float


@torch.no_grad()
def predict_image(model, spec: ModelSpec, image: Image.Image, metadata, device: torch.device, topk: int = 5):
    tensor, _ = preprocess_pil(image, spec.use_clahe)
    tensor = tensor.to(device)
    outputs = model(tensor)
    if spec.kind == "hierarchical":
        class_logits = outputs["class"]
        fresh_rank = int(predict_coral_rank(outputs["freshness"])[0].item())
        species_idx = int(outputs["species"].argmax(1)[0].item())
        extra = {
            "freshness": RANK_TO_FRESHNESS[fresh_rank],
            "species": metadata.species_names[species_idx],
        }
    else:
        class_logits = outputs
        extra = {}

    probs = class_logits.softmax(1)[0].detach().cpu()
    values, indices = torch.topk(probs, k=min(topk, len(metadata.class_names)))
    top_predictions = [
        {
            "class": metadata.class_names[int(idx)],
            "confidence": float(value),
        }
        for value, idx in zip(values, indices)
    ]
    return {
        "top_predictions": top_predictions,
        "predicted_class_idx": int(indices[0]),
        "extra": extra,
    }


def _swin_reshape_transform(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        size = int(tokens ** 0.5)
        return tensor.reshape(batch, size, size, channels).permute(0, 3, 1, 2)
    return tensor


def gradcam_target(spec: ModelSpec, model):
    reshape_transform: Callable | None = None
    cam_model = model
    if spec.kind == "hierarchical":
        cam_model = HierarchicalClassWrapper(model)
        target_layer = cam_model.model.backbone.layers[-1].blocks[-1].norm2
        reshape_transform = _swin_reshape_transform
    elif spec.architecture == "resnet50":
        target_layer = model.layer4[-1]
    elif spec.architecture == "efficientnetv2s":
        target_layer = model.blocks[-1]
    elif spec.architecture in {"convnext_small", "recipe_convnext_tiny"}:
        target_layer = model.stages[-1].blocks[-1]
    elif spec.architecture == "recipe_swin_tiny":
        target_layer = model.layers[-1].blocks[-1].norm2
        reshape_transform = _swin_reshape_transform
    else:
        raise ValueError(f"Grad-CAM not supported for {spec.architecture}")
    return cam_model, [target_layer], reshape_transform


def generate_gradcam(model, spec: ModelSpec, image: Image.Image, target_class_idx: int, device: torch.device):
    tensor, rgb_float = preprocess_pil(image, spec.use_clahe)
    cam_model, target_layers, reshape_transform = gradcam_target(spec, model)
    with GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        mask = cam(input_tensor=tensor.to(device), targets=[ClassifierOutputTarget(target_class_idx)])[0]
    overlay = show_cam_on_image(rgb_float, mask, use_rgb=True)
    return Image.fromarray(overlay)


def draw_label(image: Image.Image, text: str):
    canvas = np.asarray(image.convert("RGB")).copy()
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    return Image.fromarray(canvas)
