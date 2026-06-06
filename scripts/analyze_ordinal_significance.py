"""
Statistical strengthening for the hierarchical-ordinal paper (local, no GPU runs needed).

Fixes the unfair severe-error claim: the flat baseline's freshness was read from its
UNTRAINED CORAL head (freshness_weight=0). The only freshness signal a flat model can
provide is derived from its 24-class prediction. This script:

  1. Runs the saved flat Swin-Tiny BEST checkpoint on the identical fixed test split.
  2. Derives freshness from the 24-class argmax for BOTH models (apples-to-apples).
  3. Reports the hierarchical model's dedicated CORAL head as the added capability.
  4. McNemar (paired, same test set) for 24-class correctness and for severe errors.
  5. Quadratic Weighted Kappa, severe-error rate, MAE, and bootstrap 95% CIs.

Outputs results/stats_strengthening.json and a console summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from scipy import stats

sys.path.insert(0, ".")
import yaml
from src.hierarchical import (
    FRESHNESS_TO_RANK,
    HierarchicalClassifier,
    get_hierarchical_dataloaders,
    parse_ffe_class_name,
)


def load_config(path="configs/config.yaml"):
    return yaml.safe_load(Path(path).read_text())

FLAT_CKPT = "results/checkpoints/flat_swin_tiny_flat_BEST.pth"
HIER_CSV = "results/figures/hierarchical_ordinal/hierarchical_ordinal_test_predictions.csv"
BACKBONE = "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
RNG = np.random.default_rng(42)


def class_idx_to_freshness_rank(class_names):
    return np.array([FRESHNESS_TO_RANK[parse_ffe_class_name(c).freshness] for c in class_names])


@torch.no_grad()
def flat_predictions(device):
    cfg = load_config()
    _, _, test_loader, meta = get_hierarchical_dataloaders(cfg, split_seed=42, batch_size=64, num_workers=2)
    model = HierarchicalClassifier(BACKBONE, meta.num_classes, meta.num_species, pretrained=False).to(device)
    model.load_state_dict(torch.load(FLAT_CKPT, map_location=device, weights_only=True))
    model.eval()
    cls2fresh = class_idx_to_freshness_rank(meta.class_names)
    paths, pred_cls, true_cls, true_fresh = [], [], [], []
    offset = 0
    samples = test_loader.dataset.samples
    for images, targets in test_loader:
        logits = model(images.to(device))["class"]
        p = logits.argmax(1).cpu().numpy()
        pred_cls.append(p)
        true_cls.append(targets["class"].numpy())
        true_fresh.append(targets["freshness"].numpy())
        for i in range(len(p)):
            paths.append(samples[offset + i].path)
        offset += len(p)
    pred_cls = np.concatenate(pred_cls)
    return pd.DataFrame({
        "path": paths,
        "true_class_idx": np.concatenate(true_cls),
        "pred_class_idx": pred_cls,
        "true_freshness": np.concatenate(true_fresh),
        "flat_pred_fresh_classderived": cls2fresh[pred_cls],
    })


def load_hier(meta_class_to_rank_csv):
    df = pd.read_csv(HIER_CSV)
    df["true_freshness_rank"] = df["true_freshness"].map(FRESHNESS_TO_RANK)
    df["hier_coral_rank"] = df["pred_freshness"].map(FRESHNESS_TO_RANK)
    df["hier_pred_fresh_classderived"] = df["pred_class"].apply(
        lambda c: FRESHNESS_TO_RANK[parse_ffe_class_name(c).freshness])
    df["hier_class_correct"] = (df["pred_class"] == df["true_class"]).astype(int)
    return df[["path", "true_freshness_rank", "hier_coral_rank",
               "hier_pred_fresh_classderived", "hier_class_correct"]]


def mcnemar(a_correct, b_correct):
    a = np.asarray(a_correct).astype(bool)
    b = np.asarray(b_correct).astype(bool)
    b_only = int(np.sum(a & ~b))   # A right, B wrong
    c_only = int(np.sum(~a & b))   # A wrong, B right
    n = b_only + c_only
    if n == 0:
        return {"b": b_only, "c": c_only, "p_exact": 1.0}
    p = float(stats.binomtest(min(b_only, c_only), n, 0.5).pvalue)
    return {"b": b_only, "c": c_only, "p_exact": p}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device={device}\nRunning flat checkpoint inference on fixed test split ...", flush=True)
    flat = flat_predictions(device)
    hier = load_hier(None)
    df = flat.merge(hier, on="path", how="inner")
    assert len(df) == len(flat) == len(hier), f"join mismatch {len(df)},{len(flat)},{len(hier)}"
    n = len(df)
    tf = df["true_freshness"].to_numpy()
    assert np.array_equal(tf, df["true_freshness_rank"].to_numpy()), "freshness label mismatch"

    flat_fresh = df["flat_pred_fresh_classderived"].to_numpy()
    hier_fresh_cd = df["hier_pred_fresh_classderived"].to_numpy()
    hier_coral = df["hier_coral_rank"].to_numpy()
    flat_cls_correct = (df["pred_class_idx"].to_numpy() == df["true_class_idx"].to_numpy()).astype(int)
    hier_cls_correct = df["hier_class_correct"].to_numpy()

    def severe(pred):
        return int(np.sum(np.abs(pred - tf) >= 2))
    def severe_rate(pred):
        return float(np.mean(np.abs(pred - tf) >= 2) * 100)
    def mae(pred):
        return float(np.mean(np.abs(pred - tf)))
    def qwk(pred):
        return float(cohen_kappa_score(tf, pred, weights="quadratic"))

    res = {
        "n_test": n,
        "flat_24class_acc": round(flat_cls_correct.mean() * 100, 2),
        "hier_24class_acc": round(hier_cls_correct.mean() * 100, 2),
        "mcnemar_24class_flat_vs_hier": mcnemar(flat_cls_correct, hier_cls_correct),
        "freshness_class_derived": {
            "flat_severe": severe(flat_fresh), "hier_severe": severe(hier_fresh_cd),
            "flat_severe_rate_pct": round(severe_rate(flat_fresh), 3),
            "hier_severe_rate_pct": round(severe_rate(hier_fresh_cd), 3),
            "flat_mae": round(mae(flat_fresh), 4), "hier_mae": round(mae(hier_fresh_cd), 4),
            "flat_qwk": round(qwk(flat_fresh), 4), "hier_qwk": round(qwk(hier_fresh_cd), 4),
            "mcnemar_severe": mcnemar(np.abs(flat_fresh - tf) < 2, np.abs(hier_fresh_cd - tf) < 2),
        },
        "freshness_hier_coral_head": {
            "severe": severe(hier_coral), "severe_rate_pct": round(severe_rate(hier_coral), 3),
            "mae": round(mae(hier_coral), 4), "qwk": round(qwk(hier_coral), 4),
        },
        "severe_reduction_factor_flatCD_vs_hierCORAL": (
            round(severe(flat_fresh) / max(1, severe(hier_coral)), 1)),
    }

    # Paired bootstrap CIs: resample (prediction, label) pairs together.
    def boot_pair(metric_fn, pred, k=2000, dec=4):
        idx = np.arange(n); out = []
        for _ in range(k):
            s = RNG.choice(idx, n, replace=True)
            out.append(metric_fn(pred[s], tf[s]))
        return [round(float(np.percentile(out, 2.5)), dec), round(float(np.percentile(out, 97.5)), dec)]

    accv = hier_cls_correct.astype(float)
    accs = [float(accv[RNG.choice(np.arange(n), n, replace=True)].mean() * 100) for _ in range(2000)]
    res["hier_24class_acc_ci95"] = [round(np.percentile(accs, 2.5), 2), round(np.percentile(accs, 97.5), 2)]
    res["freshness_hier_coral_head"]["severe_rate_ci95"] = boot_pair(
        lambda p, t: float(np.mean(np.abs(p - t) >= 2) * 100), hier_coral, dec=3)
    res["hier_coral_qwk_ci95"] = boot_pair(
        lambda p, t: float(cohen_kappa_score(t, p, weights="quadratic")), hier_coral)
    res["hier_coral_mae_ci95"] = boot_pair(
        lambda p, t: float(np.mean(np.abs(p - t))), hier_coral)

    # Per-seed 24-class accuracy paired t-test (class head trained in both; fair)
    flat_seed = [86.33, 88.50, 87.70]
    hier_seed = [89.41, 88.50, 87.47]
    t, p = stats.ttest_rel(hier_seed, flat_seed)
    res["accuracy_paired_ttest_3seed"] = {"t": round(float(t), 3), "p": round(float(p), 4),
                                          "mean_diff_pp": round(float(np.mean(hier_seed) - np.mean(flat_seed)), 2)}

    Path("results/stats_strengthening.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nSaved -> results/stats_strengthening.json")


if __name__ == "__main__":
    main()
