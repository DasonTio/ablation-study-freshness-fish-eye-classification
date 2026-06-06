import os
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path


def build_optimizer(model, learning_rate, weight_decay):
    """Keep frozen params in the optimizer so later unfreezing can fine-tune them."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _move_images(imgs, device, channels_last=False):
    imgs = imgs.to(device, non_blocking=True)
    if channels_last and imgs.ndim == 4:
        imgs = imgs.contiguous(memory_format=torch.channels_last)
    return imgs


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp_enabled=False, channels_last=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = _move_images(imgs, device, channels_last=channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled=False, channels_last=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Eval ", leave=False):
        imgs = _move_images(imgs, device, channels_last=channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def run_experiment(
    experiment_name,
    model,
    train_loader,
    val_loader,
    cfg,
    device,
    model_name
):
    """Full training run for one ablation experiment. Saves best checkpoint."""
    Path(cfg["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)

    log_path = f"{cfg['paths']['logs']}/{experiment_name}.csv"
    ckpt_path = f"{cfg['paths']['checkpoints']}/{experiment_name}_best.pth"

    optimizer = build_optimizer(
        model,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    amp_enabled = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    channels_last = bool(cfg["training"].get("channels_last", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    best_val_acc = 0.0
    patience_counter = 0
    patience = cfg["training"]["patience"]
    freeze_epochs = cfg["models"].get(model_name, {}).get("freeze_epochs", 5)
    unfreeze_lr = cfg["training"].get("unfreeze_lr", 1e-5)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            if epoch == freeze_epochs + 1:
                from src.models import unfreeze_all
                unfreeze_all(model)
                print(f"Epoch {epoch}: Unfreezing all layers, reducing LR to {unfreeze_lr:g}")
                for g in optimizer.param_groups:
                    g["lr"] = unfreeze_lr

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                scaler=scaler, amp_enabled=amp_enabled, channels_last=channels_last
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device,
                amp_enabled=amp_enabled, channels_last=channels_last
            )
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])
            f.flush()

            print(f"Epoch {epoch:3d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}. Best val acc: {best_val_acc:.4f}")
                    break

    print(f"\nBest val acc for {experiment_name}: {best_val_acc:.4f}")
    print(f"Checkpoint saved: {ckpt_path}")
    return best_val_acc, ckpt_path
