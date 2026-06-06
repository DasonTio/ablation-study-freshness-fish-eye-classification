#!/usr/bin/env python3
"""Direct global ocular signal analysis for the FFE manuscript.

This script quantifies first-order luminance/color distribution statistics from
the FFE images and tests whether they track the ordered freshness labels. It is
designed as a direct evidence bridge for the paper's global-first-order thesis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.global_signal import (
    build_global_feature_table,
    evaluate_feature_set,
    evaluate_feature_set_runs,
    summarize_species_stratified_correlations,
)

RESULT_DIR = ROOT / "results" / "global_signal"
FIG_DIR = ROOT / "results" / "figures" / "jutisi"
SEEDS = [42, 123, 2024, 7, 2025]
FRESHNESS_ORDER = ["Not Fresh", "Fresh", "Highly Fresh"]


def feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix) and pd.api.types.is_numeric_dtype(df[col])]


def add_species_one_hot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    one_hot = pd.get_dummies(df["species"], prefix="species", dtype=float)
    out = pd.concat([df.reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1)
    return out, list(one_hot.columns)


def pretty_feature_name(name: str) -> str:
    return (
        name.replace("raw_center_", "")
        .replace("clahe_center_", "CLAHE ")
        .replace("_", " ")
        .replace("lab l", "Lab L")
        .replace("hsv s", "HSV S")
        .replace("hsv v", "HSV V")
        .replace("gray", "Gray")
    )


def make_distribution_figure(df: pd.DataFrame, top_features: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.reshape(-1)
    for ax, feature in zip(axes, top_features[:4]):
        data = [df[df["freshness"] == freshness][feature].to_numpy() for freshness in FRESHNESS_ORDER]
        ax.boxplot(data, tick_labels=["Not\nFresh", "Fresh", "Highly\nFresh"], patch_artist=True,
                   boxprops={"facecolor": "#d9eaf7", "edgecolor": "black"},
                   medianprops={"color": "#c0392b", "linewidth": 1.5},
                   whiskerprops={"color": "black"},
                   capprops={"color": "black"})
        ax.set_title(pretty_feature_name(feature), fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Global Ocular Statistics Across Freshness Levels", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_classifier_figure(summary: pd.DataFrame, out_path: Path) -> None:
    order = [
        "species_only",
        "raw_full_global",
        "raw_center_global",
        "clahe_center_global",
        "species_plus_raw_center",
    ]
    labels = [
        "Species\nonly",
        "Raw full\nstats",
        "Raw center\nstats",
        "CLAHE center\nstats",
        "Species + raw\ncenter stats",
    ]
    rows = summary.set_index("feature_set").loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(range(len(order)), rows["accuracy_mean"] * 100.0,
                  yerr=rows["accuracy_std"] * 100.0, capsize=4,
                  color=["#888888", "#1f3b73", "#1f3b73", "#c0392b", "#2e7d32"],
                  edgecolor="black", linewidth=0.7)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Freshness accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    feature_path = RESULT_DIR / "global_signal_features.csv"
    if feature_path.exists() and not args.force:
        features = pd.read_csv(feature_path)
    else:
        features = build_global_feature_table(
            ROOT / args.dataset,
            central_fraction=args.central_fraction,
            max_images=args.max_images,
        )
        features.to_csv(feature_path, index=False)

    raw_center = feature_columns(features, "raw_center_")
    raw_full = feature_columns(features, "raw_full_")
    clahe_center = feature_columns(features, "clahe_center_")

    corr_raw = summarize_species_stratified_correlations(features, raw_center)
    corr_raw["feature_family"] = "raw_center"
    corr_clahe = summarize_species_stratified_correlations(features, clahe_center)
    corr_clahe["feature_family"] = "clahe_center"
    corr_full = summarize_species_stratified_correlations(features, raw_full)
    corr_full["feature_family"] = "raw_full"
    corr = pd.concat([corr_raw, corr_clahe, corr_full], ignore_index=True)
    corr.to_csv(RESULT_DIR / "global_signal_correlation_summary.csv", index=False)

    model_df, species_cols = add_species_one_hot(features)
    feature_sets = {
        "species_only": species_cols,
        "raw_full_global": raw_full,
        "raw_center_global": raw_center,
        "clahe_center_global": clahe_center,
        "species_plus_raw_center": species_cols + raw_center,
        "species_plus_clahe_center": species_cols + clahe_center,
    }
    classifier_rows = []
    per_seed_rows = []
    for name, cols in feature_sets.items():
        result = evaluate_feature_set(model_df, cols, seeds=SEEDS)
        result["feature_set"] = name
        classifier_rows.append(result)
        runs = evaluate_feature_set_runs(model_df, cols, seeds=SEEDS)
        runs["feature_set"] = name
        per_seed_rows.append(runs)
    classifier_summary = pd.DataFrame(classifier_rows)
    classifier_summary = classifier_summary[[
        "feature_set",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "qwk_mean",
        "mae_mean",
        "severe_mean",
    ]]
    classifier_summary.to_csv(RESULT_DIR / "global_signal_classifier_summary.csv", index=False)
    classifier_runs = pd.concat(per_seed_rows, ignore_index=True)
    classifier_runs.to_csv(RESULT_DIR / "global_signal_classifier_runs.csv", index=False)

    comparisons = []
    for baseline, candidate in [
        ("clahe_center_global", "raw_center_global"),
        ("species_plus_clahe_center", "species_plus_raw_center"),
        ("species_only", "raw_center_global"),
    ]:
        base = classifier_runs[classifier_runs["feature_set"] == baseline].sort_values("seed")
        cand = classifier_runs[classifier_runs["feature_set"] == candidate].sort_values("seed")
        merged = pd.merge(
            base[["seed", "accuracy", "qwk", "mae", "severe"]],
            cand[["seed", "accuracy", "qwk", "mae", "severe"]],
            on="seed",
            suffixes=("_baseline", "_candidate"),
        )
        delta = merged["accuracy_candidate"] - merged["accuracy_baseline"]
        t_stat, p_value = stats.ttest_rel(merged["accuracy_candidate"], merged["accuracy_baseline"])
        comparisons.append({
            "baseline": baseline,
            "candidate": candidate,
            "n": int(len(merged)),
            "accuracy_delta_mean": float(delta.mean()),
            "accuracy_delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "paired_t": float(t_stat),
            "paired_p": float(p_value),
            "qwk_delta_mean": float((merged["qwk_candidate"] - merged["qwk_baseline"]).mean()),
            "mae_delta_mean": float((merged["mae_candidate"] - merged["mae_baseline"]).mean()),
            "severe_delta_mean": float((merged["severe_candidate"] - merged["severe_baseline"]).mean()),
        })
    comparison_summary = pd.DataFrame(comparisons)
    comparison_summary.to_csv(RESULT_DIR / "global_signal_classifier_comparisons.csv", index=False)

    top_raw = corr_raw.sort_values(["mean_abs_rho", "consistent_species"], ascending=[False, False]).head(8)
    make_distribution_figure(
        features,
        top_raw["feature"].head(4).tolist(),
        FIG_DIR / "figure2_global_signal_distributions.png",
    )
    make_classifier_figure(
        classifier_summary,
        FIG_DIR / "figure3_global_signal_classifier.png",
    )

    summary = {
        "n_images": int(len(features)),
        "central_fraction": args.central_fraction,
        "top_raw_center_features": top_raw[[
            "feature",
            "global_rho",
            "global_p",
            "mean_abs_rho",
            "consistent_species",
            "species_n",
        ]].to_dict(orient="records"),
        "classifier_summary": classifier_summary.to_dict(orient="records"),
        "classifier_comparisons": comparison_summary.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - start, 3),
    }
    (RESULT_DIR / "global_signal_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/FFE")
    parser.add_argument("--central-fraction", type=float, default=0.65)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
