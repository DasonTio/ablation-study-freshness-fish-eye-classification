import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def apply_clahe_lab(img_pil, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to L-channel only in LAB space. Returns PIL RGB image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l_eq = clahe.apply(l)
    img_lab_eq = cv2.merge([l_eq, a, b])
    img_bgr_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    img_rgb_eq = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_eq)


def visualize_clahe_effect(image_path, clip_limit=2.0, tile_grid_size=(8, 8), save_path=None):
    """Show side-by-side original vs CLAHE-enhanced fish eye image."""
    original = Image.open(image_path).convert("RGB")
    enhanced = apply_clahe_lab(original, clip_limit, tile_grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(enhanced)
    axes[1].set_title(f"CLAHE (clip={clip_limit}, grid={tile_grid_size})", fontsize=14)
    axes[1].axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
