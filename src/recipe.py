"""
Modern training recipe for FFE fish-eye freshness, grounded in Hoang et al. 2025
(arXiv:2510.24814), which reports Swin-Tiny 84.85% and ConvNeXt-Base 84.51% on the
same 64/16/20 split with mild augmentation (no color/contrast manipulation).

Design choices and why:
- Mild aug only (HFlip, Rotation±30, Brightness±0.2). Freshness signal = eye
  color/turbidity; hue/contrast aug (and CLAHE) destroy it — confirmed by our own
  refuted-CLAHE result and Hoang's deliberately mild aug.
- Full fine-tune (no frozen warmup) with low LR — matches the "deep feature
  optimization" regime that reached 84-86%.
- drop_path (stochastic depth) + EMA + label smoothing = light regularization for a
  small (~4390 img) fine-grained dataset.
- timm native head via num_classes (no manual head surgery) → uniform across
  ConvNeXt / Swin / ResNet, avoids the .head-vs-.classifier bug class.
- pre_logits feature extraction enables the deep-feature + classical-ML head that is
  the current FFE SOTA trick.
"""
import copy

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.dataset import FFEDataset, build_class_map

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_recipe_model(tag, num_classes, dropout=0.3, drop_path=0.1):
    """timm model with native classifier head mapping to num_classes."""
    return timm.create_model(
        tag,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=drop_path,
    )


def get_recipe_transforms(image_size, split):
    """Mild, color-safe augmentation. No hue, no CLAHE, no heavy contrast."""
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


def get_recipe_dataloaders(cfg, split_seed, batch_size, num_workers=8):
    """Fixed split (split_seed constant across runs) so 3-seed ensembling is leak-free."""
    samples, class_names = build_class_map(cfg["dataset"]["root"])
    n = len(samples)
    n_train = int(n * cfg["dataset"]["train_split"])
    n_val = int(n * cfg["dataset"]["val_split"])
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(split_seed)
    train_s, val_s, test_s = random_split(samples, [n_train, n_val, n_test], generator=gen)

    img = cfg["dataset"]["image_size"]
    train_ds = FFEDataset(list(train_s), get_recipe_transforms(img, "train"))
    val_ds = FFEDataset(list(val_s), get_recipe_transforms(img, "val"))
    test_ds = FFEDataset(list(test_s), get_recipe_transforms(img, "test"))

    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=num_workers,
                                   pin_memory=True, persistent_workers=True)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), class_names


def _make_ema(model):
    """Best-effort EMA via timm; returns None if unavailable."""
    try:
        from timm.utils import ModelEmaV2
        return ModelEmaV2(model, decay=0.9998)
    except Exception:
        return None


@torch.no_grad()
def _eval_acc(model, loader, device, amp):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total


def train_recipe(model, train_loader, val_loader, device, epochs=120, warmup=5,
                 lr=2e-4, weight_decay=0.05, label_smoothing=0.1, patience=30, amp=True):
    """Full fine-tune with warmup→cosine, EMA, label smoothing. Returns best state_dict."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[warmup])
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = _make_ema(model)

    best_acc, best_state, wait = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
        sched.step()

        eval_model = ema.module if ema is not None else model
        val_acc = _eval_acc(eval_model, val_loader, device, amp)
        print(f"Epoch {epoch:3d} | Val(EMA): {val_acc:.4f} | LR: {sched.get_last_lr()[0]:.2e}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(eval_model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch}. Best val {best_acc:.4f}", flush=True)
                break
    return best_state, best_acc


@torch.no_grad()
def predict_logits(model, loader, device, amp=True, tta=False):
    """Return (probs[N,C], labels[N]). tta=True averages original + horizontal flip."""
    model.eval()
    amp = amp and device.type == "cuda"
    all_p, all_y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            p = model(imgs).softmax(1)
            if tta:
                p = p + model(torch.flip(imgs, dims=[3])).softmax(1)
        all_p.append((p / (2 if tta else 1)).float().cpu().numpy())
        all_y.append(labels.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


@torch.no_grad()
def extract_features(model, loader, device, amp=True):
    """pre_logits pooled features for the deep-feature + classical-ML hybrid head."""
    model.eval()
    amp = amp and device.type == "cuda"
    X, Y = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            feats = model.forward_head(model.forward_features(imgs), pre_logits=True)
        X.append(feats.float().cpu().numpy())
        Y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(Y)
