import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.recipe import IMAGENET_MEAN, IMAGENET_STD


FRESHNESS_TO_RANK = {
    "Not Fresh": 0,
    "Fresh": 1,
    "Highly Fresh": 2,
}
RANK_TO_FRESHNESS = {v: k for k, v in FRESHNESS_TO_RANK.items()}


@dataclass(frozen=True)
class ParsedFFEClass:
    species: str
    freshness: str
    freshness_rank: int


@dataclass(frozen=True)
class HierarchicalSample:
    path: str
    class_idx: int
    species_idx: int
    freshness_rank: int


@dataclass(frozen=True)
class HierarchicalMetadata:
    class_names: list[str]
    species_names: list[str]
    species_to_idx: dict[str, int]
    num_classes: int
    num_species: int
    num_freshness: int = 3


def parse_ffe_class_name(class_name: str) -> ParsedFFEClass:
    parts = [part.strip() for part in class_name.strip().split(" - ") if part.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid FFE class name: {class_name!r}")
    species = " - ".join(parts[:-1])
    freshness = parts[-1]
    if freshness not in FRESHNESS_TO_RANK:
        raise ValueError(f"Unknown FFE freshness label: {freshness!r}")
    return ParsedFFEClass(
        species=species,
        freshness=freshness,
        freshness_rank=FRESHNESS_TO_RANK[freshness],
    )


def coral_targets(ranks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=ranks.device)
    return (ranks.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, ranks: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    targets = coral_targets(ranks, num_classes=num_classes).to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets)


def predict_coral_rank(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


def build_hierarchical_metadata(root) -> tuple[list[HierarchicalSample], HierarchicalMetadata]:
    root = Path(root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.strip())
    class_names = [d.name.strip() for d in class_dirs]
    parsed_by_class = [parse_ffe_class_name(name) for name in class_names]
    species_names = sorted({parsed.species for parsed in parsed_by_class})
    species_to_idx = {name: idx for idx, name in enumerate(species_names)}
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples = []
    for class_dir in class_dirs:
        class_name = class_dir.name.strip()
        parsed = parse_ffe_class_name(class_name)
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append(HierarchicalSample(
                    path=str(img_file),
                    class_idx=class_to_idx[class_name],
                    species_idx=species_to_idx[parsed.species],
                    freshness_rank=parsed.freshness_rank,
                ))

    metadata = HierarchicalMetadata(
        class_names=class_names,
        species_names=species_names,
        species_to_idx=species_to_idx,
        num_classes=len(class_names),
        num_species=len(species_names),
    )
    return samples, metadata


class HierarchicalFFEDataset(Dataset):
    def __init__(self, samples: list[HierarchicalSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = {
            "class": torch.tensor(sample.class_idx, dtype=torch.long),
            "species": torch.tensor(sample.species_idx, dtype=torch.long),
            "freshness": torch.tensor(sample.freshness_rank, dtype=torch.long),
        }
        return image, target


def get_hierarchical_transforms(image_size: int, split: str):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def get_hierarchical_dataloaders(
    cfg,
    split_seed: int,
    batch_size: int,
    num_workers: int = 8,
    max_samples: int | None = None,
):
    samples, metadata = build_hierarchical_metadata(cfg["dataset"]["root"])
    if max_samples is not None:
        samples = samples[:max_samples]
    n = len(samples)
    n_train = int(n * cfg["dataset"]["train_split"])
    n_val = int(n * cfg["dataset"]["val_split"])
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(split_seed)
    train_s, val_s, test_s = random_split(samples, [n_train, n_val, n_test], generator=gen)

    image_size = cfg["dataset"]["image_size"]
    train_ds = HierarchicalFFEDataset(list(train_s), get_hierarchical_transforms(image_size, "train"))
    val_ds = HierarchicalFFEDataset(list(val_s), get_hierarchical_transforms(image_size, "val"))
    test_ds = HierarchicalFFEDataset(list(test_s), get_hierarchical_transforms(image_size, "test"))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        common["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, metadata


class HierarchicalClassifier(nn.Module):
    def __init__(
        self,
        backbone_tag: str,
        num_classes: int,
        num_species: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_tag,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=dropout,
            drop_path_rate=drop_path,
        )
        feature_dim = self.backbone.num_features
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.species_head = nn.Linear(feature_dim, num_species)
        self.freshness_head = nn.Linear(feature_dim, 2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.extract_features(x)
        return {
            "class": self.class_head(features),
            "species": self.species_head(features),
            "freshness": self.freshness_head(features),
        }


def hierarchical_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    species_weight: float = 0.3,
    freshness_weight: float = 0.7,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    class_loss = F.cross_entropy(outputs["class"], targets["class"], label_smoothing=label_smoothing)
    species_loss = F.cross_entropy(outputs["species"], targets["species"], label_smoothing=label_smoothing)
    freshness_loss = coral_loss(outputs["freshness"], targets["freshness"], num_classes=3)
    return class_loss + species_weight * species_loss + freshness_weight * freshness_loss


def _targets_to_device(targets: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) for key, value in targets.items()}


@torch.no_grad()
def evaluate_hierarchical(model, loader, device, amp=True):
    model.eval()
    amp = amp and device.type == "cuda"
    class_preds, class_labels = [], []
    freshness_preds, freshness_labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = _targets_to_device(targets, device)
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)
        class_preds.append(outputs["class"].argmax(1).cpu().numpy())
        class_labels.append(targets["class"].cpu().numpy())
        freshness_preds.append(predict_coral_rank(outputs["freshness"]).cpu().numpy())
        freshness_labels.append(targets["freshness"].cpu().numpy())

    return compute_hierarchical_metrics(
        np.concatenate(class_preds),
        np.concatenate(class_labels),
        np.concatenate(freshness_preds),
        np.concatenate(freshness_labels),
    )


def compute_hierarchical_metrics(
    class_preds,
    class_labels,
    freshness_preds,
    freshness_labels,
) -> dict[str, float]:
    class_preds = np.asarray(class_preds)
    class_labels = np.asarray(class_labels)
    freshness_preds = np.asarray(freshness_preds)
    freshness_labels = np.asarray(freshness_labels)
    freshness_abs = np.abs(freshness_preds - freshness_labels)
    return {
        "class_acc": round(accuracy_score(class_labels, class_preds) * 100, 2),
        "class_precision": round(precision_score(class_labels, class_preds, average="macro", zero_division=0) * 100, 2),
        "class_recall": round(recall_score(class_labels, class_preds, average="macro", zero_division=0) * 100, 2),
        "class_f1": round(f1_score(class_labels, class_preds, average="macro", zero_division=0) * 100, 2),
        "freshness_acc": round(accuracy_score(freshness_labels, freshness_preds) * 100, 2),
        "freshness_mae": round(float(freshness_abs.mean()), 4),
        "freshness_adjacent_errors": int((freshness_abs == 1).sum()),
        "freshness_severe_errors": int((freshness_abs >= 2).sum()),
    }


def train_hierarchical(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 90,
    warmup: int = 5,
    lr: float = 2e-4,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    species_weight: float = 0.3,
    freshness_weight: float = 0.7,
    patience: int = 18,
    amp: bool = True,
):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[warmup])
    amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = {}
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = _targets_to_device(targets, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images)
                loss = hierarchical_loss(
                    outputs,
                    targets,
                    species_weight=species_weight,
                    freshness_weight=freshness_weight,
                    label_smoothing=label_smoothing,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        metrics = evaluate_hierarchical(model, val_loader, device, amp=amp)
        val_acc = metrics["class_acc"] / 100.0
        print(
            f"Epoch {epoch:3d} | Val class={metrics['class_acc']:.2f}% "
            f"fresh={metrics['freshness_acc']:.2f}% mae={metrics['freshness_mae']:.3f} "
            f"LR={sched.get_last_lr()[0]:.2e}",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}. Best val class {best_acc:.4f}", flush=True)
                break
    return best_state, best_metrics


@torch.no_grad()
def extract_hierarchical_features(model, loader, device, amp=True):
    model.eval()
    amp = amp and device.type == "cuda"
    features, class_labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            batch_features = model.extract_features(images)
        features.append(batch_features.float().cpu().numpy())
        class_labels.append(targets["class"].numpy())
    return np.concatenate(features), np.concatenate(class_labels)
