import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model, backbone_name):
    """Return last conv layer for Grad-CAM."""
    if backbone_name == "resnet50":
        return [model.layer4[-1]]
    elif backbone_name == "efficientnetv2s":
        return [model.blocks[-1]]
    elif backbone_name == "convnext_small":
        # Last ConvNeXt block in the last stage outputs (B, C, H, W) spatial features
        return [model.stages[-1].blocks[-1]]
    raise ValueError(f"Unknown backbone: {backbone_name}")


def preprocess_image(image_path, use_clahe, image_size=224, clip_limit=2.0):
    """Load, optionally apply CLAHE, return (tensor, rgb_float_array)."""
    from src.dataset import CLAHETransform
    img = Image.open(image_path).convert("RGB")

    if use_clahe:
        clahe_fn = CLAHETransform(clip_limit=clip_limit)
        img = clahe_fn(img)

    img_np = np.array(img.resize((image_size, image_size))).astype(np.float32) / 255.0

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor = normalize(torch.from_numpy(img_np).permute(2, 0, 1)).unsqueeze(0)
    return tensor, img_np


def generate_gradcam_comparison(
    image_path,
    model_no_clahe,
    model_with_clahe,
    backbone_name,
    target_class,
    device,
    save_path
):
    """
    4-panel figure: original | Grad-CAM no CLAHE | CLAHE enhanced | Grad-CAM with CLAHE.
    """
    target_layer_no   = get_target_layer(model_no_clahe, backbone_name)
    target_layer_with = get_target_layer(model_with_clahe, backbone_name)

    tensor_raw,   img_raw   = preprocess_image(image_path, use_clahe=False)
    tensor_clahe, img_clahe = preprocess_image(image_path, use_clahe=True)

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model_no_clahe, target_layers=target_layer_no) as cam:
        mask_raw = cam(input_tensor=tensor_raw.to(device), targets=targets)[0]
    with GradCAM(model=model_with_clahe, target_layers=target_layer_with) as cam:
        mask_clahe = cam(input_tensor=tensor_clahe.to(device), targets=targets)[0]

    overlay_raw   = show_cam_on_image(img_raw,   mask_raw,   use_rgb=True)
    overlay_clahe = show_cam_on_image(img_clahe, mask_clahe, use_rgb=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_raw);       axes[0].set_title("Original");             axes[0].axis("off")
    axes[1].imshow(overlay_raw);   axes[1].set_title("Grad-CAM (No CLAHE)");  axes[1].axis("off")
    axes[2].imshow(img_clahe);     axes[2].set_title("CLAHE Enhanced");       axes[2].axis("off")
    axes[3].imshow(overlay_clahe); axes[3].set_title("Grad-CAM (With CLAHE)");axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM comparison: {save_path}")
