#!/usr/bin/env python3
"""Statistical readout for the second-order FFE pooling study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


PRIMARY_ARMS = ["B_raw_bilinear", "C_gap_raw_bilinear"]


def paired_arm_stats(df: pd.DataFrame, baseline_arm: str, candidate_arm: str) -> dict:
    base = df[df["arm"] == baseline_arm][["seed", "test_class_acc"]].rename(columns={"test_class_acc": "base"})
    cand = df[df["arm"] == candidate_arm][["seed", "test_class_acc"]].rename(columns={"test_class_acc": "candidate"})
    paired = base.merge(cand, on="seed", how="inner").sort_values("seed")
    if paired.empty:
        return {"baseline_arm": baseline_arm, "candidate_arm": candidate_arm, "n": 0, "error": "no_matched_seeds", "verdict": "incomplete"}
    deltas = paired["candidate"].to_numpy(dtype=float) - paired["base"].to_numpy(dtype=float)
    mean_delta = float(deltas.mean())
    if len(deltas) < 2:
        t_stat, p_value = math.nan, math.nan
    elif float(np.std(deltas, ddof=1)) == 0.0:
        t_stat = math.inf if mean_delta > 0 else -math.inf if mean_delta < 0 else 0.0
        p_value = 0.0 if mean_delta != 0 else 1.0
    else:
        t_stat, p_value = stats.ttest_rel(paired["candidate"], paired["base"])
        t_stat, p_value = float(t_stat), float(p_value)

    rng = np.random.default_rng(0)
    boot = [float(rng.choice(deltas, len(deltas), replace=True).mean()) for _ in range(5000)]
    if mean_delta >= 1.0 and (not math.isnan(p_value)) and p_value < 0.05:
        verdict = "primary_candidate"
    elif mean_delta > 0:
        verdict = "secondary_positive"
    else:
        verdict = "kill_no_gain"
    return {
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "n": int(len(paired)),
        "seeds": [int(s) for s in paired["seed"].tolist()],
        "baseline_mean": round(float(paired["base"].mean()), 2),
        "candidate_mean": round(float(paired["candidate"].mean()), 2),
        "mean_delta_pp": round(mean_delta, 2),
        "delta_ci95": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
        "paired_t": None if math.isnan(t_stat) else round(float(t_stat), 4),
        "p_value": None if math.isnan(p_value) else round(float(p_value), 6),
        "verdict": verdict,
    }


def mcnemar_exact(a_correct, b_correct) -> dict:
    a = np.asarray(a_correct).astype(bool)
    b = np.asarray(b_correct).astype(bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p_exact": round(p, 6)}


def safe_qwk(true, pred) -> float | None:
    value = float(cohen_kappa_score(true, pred, weights="quadratic"))
    if math.isnan(value):
        return None
    return round(value, 4)


def prediction_stats(pred_df: pd.DataFrame, baseline_arm: str, candidate_arm: str) -> dict:
    rows = []
    for seed in sorted(set(pred_df["seed"])):
        base = pred_df[(pred_df["arm"] == baseline_arm) & (pred_df["seed"] == seed)]
        cand = pred_df[(pred_df["arm"] == candidate_arm) & (pred_df["seed"] == seed)]
        if base.empty or cand.empty:
            continue
        joined = base.merge(cand, on="path", suffixes=("_base", "_candidate"))
        if joined.empty:
            continue
        rows.append({
            "seed": int(seed),
            "n": int(len(joined)),
            "mcnemar_class": mcnemar_exact(joined["class_correct_base"], joined["class_correct_candidate"]),
            "mcnemar_nonsevere_classderived": mcnemar_exact(
                1 - joined["severe_classderived_base"],
                1 - joined["severe_classderived_candidate"],
            ),
            "baseline_qwk_classderived": safe_qwk(
                joined["true_freshness_rank_base"],
                joined["pred_freshness_classderived_base"],
            ),
            "candidate_qwk_classderived": safe_qwk(
                joined["true_freshness_rank_candidate"],
                joined["pred_freshness_classderived_candidate"],
            ),
        })
    return {"per_seed": rows}


def summarize(df: pd.DataFrame) -> list[dict]:
    out = []
    for arm, group in df.groupby("arm", sort=False):
        out.append({
            "arm": arm,
            "n": int(len(group)),
            "class_acc_mean": round(float(group["test_class_acc"].mean()), 2),
            "class_acc_std": round(float(group["test_class_acc"].std(ddof=1)), 2) if len(group) > 1 else 0.0,
            "freshness_acc_classderived_mean": round(float(group["test_freshness_acc_classderived"].mean()), 2),
            "freshness_mae_classderived_mean": round(float(group["test_freshness_mae_classderived"].mean()), 4),
            "severe_classderived_mean": round(float(group["test_freshness_severe_classderived"].mean()), 2),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze second-order pooling significance.")
    parser.add_argument("--results-dir", default="results/secondorder")
    parser.add_argument("--baseline-arm", default="A_gap")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "secondorder_results.csv")
    out = {
        "summary": summarize(df),
        "paired_seed_tests": {},
        "pre_registered_success_rule": {
            "primary": "mean(candidate)-mean(A_gap) >= +1.0pp and paired t-test p < 0.05",
            "secondary": "positive mean delta without primary significance",
            "kill": "mean delta <= 0",
        },
    }
    for arm in [a for a in df["arm"].unique() if a != args.baseline_arm]:
        out["paired_seed_tests"][arm] = paired_arm_stats(df, args.baseline_arm, arm)

    pred_path = results_dir / "secondorder_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        out["paired_prediction_tests"] = {
            arm: prediction_stats(pred_df, args.baseline_arm, arm)
            for arm in [a for a in pred_df["arm"].unique() if a != args.baseline_arm]
        }

    primary = [
        out["paired_seed_tests"][arm]
        for arm in PRIMARY_ARMS
        if arm in out["paired_seed_tests"]
    ]
    primary = sorted(primary, key=lambda x: x["mean_delta_pp"], reverse=True)
    out["paper_verdict"] = primary[0] if primary else None
    out_path = results_dir / "secondorder_significance.json"
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(out, indent=2, allow_nan=False))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
