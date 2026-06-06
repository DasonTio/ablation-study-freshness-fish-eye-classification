"""Generate Grad-CAM comparisons for paper figures. Run AFTER ablation experiments complete."""
import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from src.dataset import load_config, build_class_map
from src.models import build_efficientnetv2s
from src.gradcam import generate_gradcam_comparison

cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build class map to find valid sample images automatically
samples, class_names = build_class_map(cfg["dataset"]["root"])
print(f"Classes: {class_names}")

# Load EfficientNetV2-S without CLAHE (Exp C)
model_no_clahe = build_efficientnetv2s(
    num_classes=cfg["dataset"]["num_classes"],
    pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"],
    dropout=cfg["training"]["dropout"]
).to(device)
model_no_clahe.load_state_dict(
    torch.load("results/checkpoints/exp_C_v2s_no_clahe_best.pth", map_location=device)
)

# Load EfficientNetV2-S with CLAHE (Exp D — proposed)
model_with_clahe = build_efficientnetv2s(
    num_classes=cfg["dataset"]["num_classes"],
    pretrained_tag=cfg["models"]["efficientnetv2s"]["pretrained"],
    dropout=cfg["training"]["dropout"]
).to(device)
model_with_clahe.load_state_dict(
    torch.load("results/checkpoints/exp_D_v2s_clahe_proposed_best.pth", map_location=device)
)

Path("results/figures/gradcam").mkdir(parents=True, exist_ok=True)

# Pick one sample per freshness level from class_names
freshness_levels = ["Highly Fresh", "Fresh", "Not Fresh"]
generated = []

for path, label_idx in samples:
    class_name = class_names[label_idx]
    freshness = next((f for f in freshness_levels if f in class_name), None)
    if freshness and freshness not in generated:
        safe_name = freshness.lower().replace(" ", "_")
        generate_gradcam_comparison(
            image_path=path,
            model_no_clahe=model_no_clahe,
            model_with_clahe=model_with_clahe,
            backbone_name="efficientnetv2s",
            target_class=label_idx,
            device=device,
            save_path=f"results/figures/gradcam/gradcam_{safe_name}.png"
        )
        generated.append(freshness)
    if len(generated) == 3:
        break

print("Grad-CAM generation complete.")
