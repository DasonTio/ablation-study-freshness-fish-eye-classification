"""
Recipe study to decisively beat the FFE benchmark (ResNet50 78.82%) and match 2025
SOTA (Hoang et al. 85.99%, arXiv:2510.24814).

For each backbone x 3 training seeds (FIXED 64/16/20 split → leak-free ensembling):
  - full fine-tune with mild aug + drop_path + EMA + label smoothing (src/recipe.py)
  - evaluate: plain, TTA (hflip), and deep-feature → ExtraTrees/LGBM hybrid head
  - 3-seed soft-vote ensemble (+TTA) on the shared test set
Outputs CSVs + comparison table (with benchmark + SOTA rows) + BEST checkpoints, then
writes results/ALL_DONE for the local watcher.
"""
import sys
sys.path.insert(0, ".")

import copy
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import load_config
from src.evaluate import compute_metrics
from src.recipe import (
    build_recipe_model,
    extract_features,
    get_recipe_dataloaders,
    predict_logits,
    train_recipe,
)
from src.seed import set_seed

SPLIT_SEED = 42          # fixed → identical test set across all runs (valid ensemble)
TRAIN_SEEDS = [42, 123, 2024]

# Tags verified on-server before launch (list_pretrained). Adjust if invalid.
BACKBONES = [
    {"key": "convnext_tiny",  "tag": "convnext_tiny.fb_in22k_ft_in1k"},
    {"key": "swin_tiny",      "tag": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"},
    {"key": "convnext_small", "tag": "convnext_small.fb_in22k_ft_in1k"},
]

BENCHMARK_ROWS = [
    {"Model": "ResNet50 [Prasetyo 2022, benchmark]", "Accuracy (%)": "78.82"},
    {"Model": "ResNet50 [Hoang 2025]",               "Accuracy (%)": "80.07"},
    {"Model": "EfficientNet-B0 [Hoang 2025]",        "Accuracy (%)": "81.32"},
    {"Model": "ConvNeXt-Base [Hoang 2025]",          "Accuracy (%)": "84.51"},
    {"Model": "Swin-Tiny [Hoang 2025]",              "Accuracy (%)": "84.85"},
    {"Model": "Swin-T + ExtraTrees + LGBM [Hoang 2025, SOTA]", "Accuracy (%)": "85.99"},
]


def hybrid_head_acc(model, train_loader, val_loader, test_loader, device):
    """Deep features → ExtraTrees and LightGBM. Returns dict of test accuracies (or {})."""
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.preprocessing import StandardScaler
        Xtr, ytr = extract_features(model, train_loader, device)
        Xva, yva = extract_features(model, val_loader, device)
        Xte, yte = extract_features(model, test_loader, device)
        Xtr = np.concatenate([Xtr, Xva]); ytr = np.concatenate([ytr, yva])
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

        out = {}
        et = ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=0).fit(Xtr, ytr)
        out["hybrid_extratrees"] = compute_metrics(et.predict(Xte), yte)["accuracy"]
        try:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(n_estimators=600, n_jobs=-1, random_state=0, verbose=-1).fit(Xtr, ytr)
            out["hybrid_lgbm"] = compute_metrics(lgbm.predict(Xte), yte)["accuracy"]
        except Exception:
            print("[hybrid] lightgbm unavailable, ExtraTrees only", flush=True)
        return out
    except Exception:
        print(f"[hybrid] failed:\n{traceback.format_exc()}", flush=True)
        return {}


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = cfg["training"]["batch_size"]
    print(f"Device: {device} | {len(BACKBONES)} backbones x {len(TRAIN_SEEDS)} seeds", flush=True)

    Path("results").mkdir(exist_ok=True)
    ckpt_dir = Path(cfg["paths"]["checkpoints"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path("results/recipe_results.csv")

    rows = []
    ensemble_probs = {}   # backbone_key -> list of per-seed test prob arrays (TTA)
    test_labels = None

    for bb in BACKBONES:
        ensemble_probs[bb["key"]] = []
        best_val_for_bb, best_state_for_bb = -1.0, None
        for seed in TRAIN_SEEDS:
            tag_name = f"{bb['key']}_seed{seed}"
            print(f"\n{'='*64}\n{tag_name}  ({bb['tag']})\n{'='*64}", flush=True)
            try:
                set_seed(seed)
                train_loader, val_loader, test_loader, class_names = get_recipe_dataloaders(
                    cfg, split_seed=SPLIT_SEED, batch_size=bs
                )
                model = build_recipe_model(bb["tag"], cfg["dataset"]["num_classes"])
                best_state, best_val = train_recipe(model, train_loader, val_loader, device)
                model.load_state_dict(best_state)

                p_plain, y = predict_logits(model, test_loader, device, tta=False)
                p_tta, _ = predict_logits(model, test_loader, device, tta=True)
                if test_labels is None:
                    test_labels = y
                ensemble_probs[bb["key"]].append(p_tta)

                m_plain = compute_metrics(p_plain.argmax(1), y)
                m_tta = compute_metrics(p_tta.argmax(1), y)
                hyb = hybrid_head_acc(model, train_loader, val_loader, test_loader, device)

                row = {
                    "backbone": bb["key"], "seed": seed, "best_val": round(best_val * 100, 2),
                    "acc": m_plain["accuracy"], "acc_tta": m_tta["accuracy"],
                    "precision": m_tta["precision"], "recall": m_tta["recall"], "f1": m_tta["f1"],
                    **{k: v for k, v in hyb.items()},
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(raw_path, index=False)  # incremental
                print(f"RESULT {tag_name}: acc={m_plain['accuracy']} tta={m_tta['accuracy']} {hyb}", flush=True)

                if best_val > best_val_for_bb:
                    best_val_for_bb, best_state_for_bb = best_val, copy.deepcopy(best_state)
            except Exception:
                print(f"[run failed] {tag_name}:\n{traceback.format_exc()}", flush=True)

        if best_state_for_bb is not None:
            torch.save(best_state_for_bb, ckpt_dir / f"recipe_{bb['key']}_BEST.pth")

    if not rows:
        print("No successful runs.", flush=True)
        return

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)

    # Per-backbone mean±std (TTA) + 3-seed ensemble
    summary, table = [], []
    for bb in BACKBONES:
        g = df[df["backbone"] == bb["key"]]
        if g.empty:
            continue
        ens_acc = None
        plist = ensemble_probs[bb["key"]]
        if len(plist) >= 2 and test_labels is not None:
            ens_pred = np.mean(plist, axis=0).argmax(1)
            ens_acc = compute_metrics(ens_pred, test_labels)["accuracy"]
        summary.append({
            "backbone": bb["key"], "n": len(g),
            "acc_mean": round(g["acc"].mean(), 2), "acc_std": round(g["acc"].std(ddof=1), 2) if len(g) > 1 else 0.0,
            "acc_tta_mean": round(g["acc_tta"].mean(), 2),
            "hybrid_et_max": round(g.get("hybrid_extratrees", pd.Series([np.nan])).max(), 2),
            "ensemble_tta_acc": ens_acc,
        })
        table.append({"Model": f"{bb['key']} + TTA (ours)", "Accuracy (%)": f"{g['acc_tta'].mean():.2f} ± {g['acc_tta'].std(ddof=1):.2f}" if len(g) > 1 else f"{g['acc_tta'].mean():.2f}"})
        if ens_acc is not None:
            table.append({"Model": f"{bb['key']} 3-seed ensemble + TTA (ours)", "Accuracy (%)": f"{ens_acc:.2f}"})

    # Cross-backbone ensemble (all seeds, all backbones)
    allp = [p for plist in ensemble_probs.values() for p in plist]
    if len(allp) >= 2 and test_labels is not None:
        x_acc = compute_metrics(np.mean(allp, axis=0).argmax(1), test_labels)["accuracy"]
        table.append({"Model": "ALL-backbone ensemble + TTA (ours)", "Accuracy (%)": f"{x_acc:.2f}"})

    pd.DataFrame(summary).to_csv("results/recipe_summary.csv", index=False)
    full_table = pd.concat([pd.DataFrame(BENCHMARK_ROWS), pd.DataFrame(table)], ignore_index=True)
    full_table.to_csv("results/recipe_comparison_table.csv", index=False)

    print("\n=== SUMMARY ===\n" + pd.DataFrame(summary).to_string(index=False), flush=True)
    print("\n=== COMPARISON TABLE ===\n" + full_table.to_string(index=False), flush=True)

    Path("results/ALL_DONE").touch()
    print("\n[ALL_DONE] Recipe study complete.", flush=True)


if __name__ == "__main__":
    main()
