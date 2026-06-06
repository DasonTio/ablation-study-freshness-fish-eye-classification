import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_class_map(root):
    """
    FFE dataset uses flat folders named 'Species - Freshness'.
    Returns (samples, class_names) where samples = list of (path, label_idx).
    """
    root = Path(root)
    # Each subdirectory is one class; strip whitespace to handle 'Nibea Albiflora -  Highly Fresh'
    class_names = sorted([
        d.name.strip()
        for d in root.iterdir()
        if d.is_dir()
    ])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    samples = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        label = class_to_idx[d.name.strip()]
        for img_file in d.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((str(img_file), label))
    return samples, class_names


class CLAHETransform:
    """Apply CLAHE to L-channel of LAB image only. Preserves color, enhances contrast."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img_pil):
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        l_eq = self.clahe.apply(l)
        img_lab_eq = cv2.merge([l_eq, a, b])
        img_bgr_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
        img_rgb_eq = cv2.cvtColor(img_bgr_eq, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb_eq)


class FFEDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size, use_clahe, clahe_cfg, split="train"):
    clahe = CLAHETransform(
        clip_limit=clahe_cfg["clip_limit"],
        tile_grid_size=tuple(clahe_cfg["tile_grid_size"])
    ) if use_clahe else None

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ] if split == "train" else []

    steps = []
    if clahe:
        steps.append(clahe)
    steps += [
        transforms.Resize((image_size, image_size)),
        *aug,
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(steps)


def get_dataloaders(cfg, use_clahe, seed=None):
    samples, class_names = build_class_map(cfg["dataset"]["root"])

    seed = cfg["dataset"]["random_seed"] if seed is None else seed
    n = len(samples)
    n_train = int(n * cfg["dataset"]["train_split"])
    n_val = int(n * cfg["dataset"]["val_split"])
    n_test = n - n_train - n_val

    import torch
    gen = torch.Generator().manual_seed(seed)
    train_s, val_s, test_s = random_split(samples, [n_train, n_val, n_test], generator=gen)

    img_size = cfg["dataset"]["image_size"]
    clahe_cfg = cfg["clahe"]

    train_ds = FFEDataset(list(train_s), get_transforms(img_size, use_clahe, clahe_cfg, "train"))
    val_ds   = FFEDataset(list(val_s),   get_transforms(img_size, use_clahe, clahe_cfg, "val"))
    test_ds  = FFEDataset(list(test_s),  get_transforms(img_size, use_clahe, clahe_cfg, "test"))

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, class_names
