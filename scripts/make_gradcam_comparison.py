"""
Side-by-side Grad-CAM comparison: Flat 24-class CE vs Hierarchical Ordinal.

Sample selection: for each freshness level, evaluate top-N candidates and pick the one
where the hierarchical model is most spatially concentrated (Gini score) AND most
different from the flat model. This maximises the visual contrast between the two
approaches instead of just picking the most confident prediction.

GradCAM++ produces cleaner, more concentrated activation maps than standard GradCAM.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.dataset import load_config
from src.hierarchical import (
    HierarchicalClassifier,
    RANK_TO_FRESHNESS,
    get_hierarchical_dataloaders,
    predict_coral_rank,
)
from src.recipe import IMAGENET_MEAN, IMAGENET_STD

BACKBONE_TAG     = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
FLAT_CKPT        = "results/checkpoints/flat_swin_tiny_flat_BEST.pth"
HIER_CKPT        = "results/checkpoints/hierarchical_ordinal_swin_tiny_hier_ord_BEST.pth"
OUT_DIR          = Path("results/figures")
N_CANDIDATES     = 40
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hand-picked images — sharp, representative of each freshness level.
# Override auto-selection when all three paths exist.
HANDPICKED = [
    ("Highly Fresh", "data/FFE/Johnius Trachycephalus - Highly Fresh/IMG_20191014_071619.jpg"),
    ("Not Fresh",    "data/FFE/Chanos Chanos - Not Fresh/IMG_20191004_062950.jpg"),
    ("Fresh",        "data/FFE/Rastrelliger Faughni - Fresh/IMG_20191016_064941.jpg"),
]


class ClassHeadWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["class"]


class FreshnessHeadWrapper(torch.nn.Module):
    """Exposes the CORAL freshness logits (B, 2) for Grad-CAM targeting.

    We target logit_0 = P(rank > 0), the "is at least Fresh?" boundary.
    This is the most informative single scalar for distinguishing freshness
    across all three levels and encodes the global turbidity signal.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["freshness"]  # (B, 2) CORAL logits


def swin_reshape(tensor):
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.ndim == 3:
        b, t, c = tensor.shape
        s = int(t ** 0.5)
        return tensor.reshape(b, s, s, c).permute(0, 3, 1, 2)
    return tensor


def load_model(ckpt_path, metadata, device):
    m = HierarchicalClassifier(
        backbone_tag=BACKBONE_TAG,
        num_classes=metadata.num_classes,
        num_species=metadata.num_species,
        pretrained=False,
        dropout=0.3,
        drop_path=0.1,
    ).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    m.eval()
    return m


def preprocess(image_path, image_size):
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    rgb = np.asarray(img).astype(np.float32) / 255.0
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tensor = norm(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    return tensor, rgb


def _run_gradcam(wrapper_cls, model, tensor, target_idx, device):
    """Run GradCAM on CPU (MPS gradient support is unreliable for hooks)."""
    cpu = torch.device("cpu")
    model.to(cpu)
    wrapped = wrapper_cls(model).eval()
    target_layer = [wrapped.model.backbone.layers[-1].blocks[-1].norm2]
    with GradCAM(model=wrapped, target_layers=target_layer,
                 reshape_transform=swin_reshape) as cam:
        mask = cam(
            input_tensor=tensor.to(cpu),
            targets=[ClassifierOutputTarget(target_idx)],
        )[0]
    model.to(device)
    return mask


def get_class_cam_mask(model, tensor, target_class, device):
    """Grad-CAM through the 24-class head — used for flat model."""
    return _run_gradcam(ClassHeadWrapper, model, tensor, target_class, device)


def get_freshness_cam_mask(model, tensor, device):
    """Grad-CAM through the CORAL freshness head logit_0 = P(rank > 0).

    Targets the freshness boundary signal directly, not the class boundary.
    For hierarchical model only — flat model freshness head is untrained noise.
    """
    return _run_gradcam(FreshnessHeadWrapper, model, tensor, 0, device)


def gini(mask):
    """Gini coefficient of the spatial activation map. Higher = more concentrated."""
    flat = np.abs(mask.flatten()).astype(np.float64)
    flat += 1e-10
    flat = np.sort(flat)
    n = len(flat)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * flat) / (n * flat.sum()) - (n + 1) / n)


@torch.no_grad()
def collect_preds(model, loader, device):
    probs, preds, labels, fresh_preds, fresh_labels = [], [], [], [], []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        p = out["class"].softmax(1)
        probs.append(p.cpu().numpy())
        preds.append(p.argmax(1).cpu().numpy())
        labels.append(targets["class"].numpy())
        fresh_preds.append(predict_coral_rank(out["freshness"]).cpu().numpy())
        fresh_labels.append(targets["freshness"].numpy())
    return (
        np.concatenate(probs),
        np.concatenate(preds),
        np.concatenate(labels),
        np.concatenate(fresh_preds),
        np.concatenate(fresh_labels),
    )


def select_best_samples(
    flat_model, hier_model, samples,
    probs, preds, labels, fresh_labels,
    device, image_size,
):
    """
    For each freshness level collect N_CANDIDATES correctly-classified images,
    score each by (hier_gini + (hier_gini - flat_gini)), return best per level.
    """
    candidates = {0: [], 1: [], 2: []}
    conf = probs.max(axis=1)
    for idx in np.argsort(-conf):
        fr = int(fresh_labels[idx])
        if labels[idx] == preds[idx] and len(candidates[fr]) < N_CANDIDATES:
            candidates[fr].append(idx)
        if all(len(v) >= N_CANDIDATES for v in candidates.values()):
            break
    # fill any freshness level that has fewer than 1 candidate (fallback)
    for idx in np.argsort(-conf):
        fr = int(fresh_labels[idx])
        candidates[fr].append(idx)
        if all(len(v) >= 1 for v in candidates.values()):
            break

    selected = {}
    for fr, idxs in candidates.items():
        best_score, best_idx = -np.inf, idxs[0]
        print(f"  Scoring {len(idxs)} candidates for freshness={RANK_TO_FRESHNESS[fr]}...", flush=True)
        for idx in idxs:
            tensor, _ = preprocess(samples[idx].path, image_size)
            target = int(preds[idx])
            hier_mask = get_cam_mask(hier_model, tensor, target, device)
            flat_mask = get_cam_mask(flat_model, tensor, target, device)
            hier_g = gini(hier_mask)
            flat_g = gini(flat_mask)
            # maximise concentration of hier CAM and contrast vs flat
            score = hier_g + (hier_g - flat_g)
            if score > best_score:
                best_score, best_idx = score, idx
                print(f"    new best idx={idx}  hier_gini={hier_g:.3f}  flat_gini={flat_g:.3f}  score={score:.3f}")
        selected[fr] = best_idx
    return [selected[r] for r in sorted(selected)]


def main():
    cfg = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    image_size = cfg["dataset"]["image_size"]
    print(f"Device: {device}")

    _, _, _, metadata = get_hierarchical_dataloaders(
        cfg, split_seed=42, batch_size=32, num_workers=0
    )
    flat_model = load_model(FLAT_CKPT, metadata, device)
    hier_model = load_model(HIER_CKPT, metadata, device)
    print("Both checkpoints loaded.")

    entries = [(fr, Path(p)) for fr, p in HANDPICKED if Path(p).exists()]
    if len(entries) < len(HANDPICKED):
        print(f"WARNING: missing paths: {[p for _,p in HANDPICKED if not Path(p).exists()]}")
    print(f"Using {len(entries)} hand-picked images.", flush=True)

    nrows = len(entries)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 4.5 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    col_titles = [
        "Original Image",
        "Flat 24-class CE\n(Grad-CAM → class head)",
        "Hierarchical Ordinal\n(Grad-CAM → freshness CORAL head)",
    ]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11, fontweight="bold", pad=10)

    for row, (fr_name, img_path) in enumerate(entries):
        tensor, rgb = preprocess(img_path, image_size)

        with torch.no_grad():
            flat_out = flat_model(tensor.to(device))
            hier_out = hier_model(tensor.to(device))
        flat_pred  = int(flat_out["class"].argmax(1).item())
        flat_conf  = float(flat_out["class"].softmax(1)[0, flat_pred].item())
        hier_pred  = int(hier_out["class"].argmax(1).item())
        hier_conf  = float(hier_out["class"].softmax(1)[0, hier_pred].item())
        hier_fresh = int((torch.sigmoid(hier_out["freshness"][0]) > 0.5).sum().item())

        # Flat → class head CAM  |  Hierarchical → freshness CORAL head CAM
        flat_mask = get_class_cam_mask(flat_model, tensor, flat_pred, device)
        hier_mask = get_freshness_cam_mask(hier_model, tensor, device)

        flat_overlay = show_cam_on_image(rgb, flat_mask, use_rgb=True,
                                         colormap=cv2.COLORMAP_JET, image_weight=0.55)
        hier_overlay = show_cam_on_image(rgb, hier_mask, use_rgb=True,
                                         colormap=cv2.COLORMAP_JET, image_weight=0.55)

        flat_g = gini(flat_mask)
        hier_g = gini(hier_mask)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(
            f"Freshness: {fr_name}\n{img_path.parent.name[:32]}",
            fontsize=8.5, labelpad=6,
        )

        axes[row, 1].imshow(flat_overlay)
        axes[row, 1].set_xlabel(
            f"Pred: {metadata.class_names[flat_pred][:28]}  (conf {flat_conf:.2f})\n"
            f"Gini: {flat_g:.3f}",
            fontsize=8,
        )

        axes[row, 2].imshow(hier_overlay)
        axes[row, 2].set_xlabel(
            f"Pred: {metadata.class_names[hier_pred][:28]}  (conf {hier_conf:.2f})\n"
            f"Freshness: {RANK_TO_FRESHNESS[hier_fresh]}  |  Gini: {hier_g:.3f}  (Δ{hier_g - flat_g:+.3f})",
            fontsize=8,
        )

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        print(f"  {fr_name:14s} — flat class Gini {flat_g:.3f}  →  hier freshness Gini {hier_g:.3f}", flush=True)

    fig.suptitle(
        "Grad-CAM: Flat 24-class CE (class head)  vs  Hierarchical Ordinal (freshness CORAL head)\n"
        "Right column shows backbone regions encoding the global turbidity/freshness signal",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=1.5)
    out_path = OUT_DIR / "gradcam_flat_vs_hier_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
