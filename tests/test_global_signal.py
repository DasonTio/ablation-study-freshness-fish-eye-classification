import numpy as np
import pandas as pd

from src.global_signal import (
    extract_first_order_features,
    evaluate_feature_set,
    evaluate_feature_set_runs,
    parse_ffe_label,
    select_evenly_spaced,
    summarize_species_stratified_correlations,
)


def test_parse_ffe_label_handles_extra_spaces_and_ordered_rank():
    species, freshness, rank = parse_ffe_label("Nibea Albiflora -  Highly Fresh")

    assert species == "Nibea Albiflora"
    assert freshness == "Highly Fresh"
    assert rank == 2


def test_extract_first_order_features_capture_global_brightness():
    dark = np.full((16, 16, 3), 20, dtype=np.uint8)
    bright = np.full((16, 16, 3), 220, dtype=np.uint8)

    dark_features = extract_first_order_features(dark, prefix="raw")
    bright_features = extract_first_order_features(bright, prefix="raw")

    assert bright_features["raw_gray_mean"] > dark_features["raw_gray_mean"]
    assert bright_features["raw_lab_l_mean"] > dark_features["raw_lab_l_mean"]
    assert np.isfinite(list(dark_features.values())).all()
    assert np.isfinite(list(bright_features.values())).all()


def test_summarize_species_stratified_correlations_counts_consistent_species():
    rows = []
    for species in ["A", "B"]:
        for rank in [0, 1, 2]:
            for rep in range(5):
                rows.append({
                    "species": species,
                    "freshness_rank": rank,
                    "signal": rank * 10 + rep,
                    "noise": rep,
                })
    df = pd.DataFrame(rows)

    summary = summarize_species_stratified_correlations(df, ["signal", "noise"])
    signal = summary.set_index("feature").loc["signal"]

    assert signal["species_n"] == 2
    assert signal["consistent_species"] == 2
    assert signal["mean_abs_rho"] > 0.9


def test_evaluate_feature_set_returns_accuracy_and_ordinal_metrics():
    rows = []
    for rank in [0, 1, 2]:
        for i in range(40):
            rows.append({
                "class_name": f"Species {i % 4} - Rank {rank}",
                "freshness_rank": rank,
                "freshness": str(rank),
                "species": f"Species {i % 4}",
                "signal": rank + i * 0.001,
            })
    df = pd.DataFrame(rows)

    result = evaluate_feature_set(df, ["signal"], seeds=[1, 2])

    assert result["n_seeds"] == 2
    assert result["accuracy_mean"] > 0.9
    assert result["qwk_mean"] > 0.9
    assert result["mae_mean"] < 0.2

    runs = evaluate_feature_set_runs(df, ["signal"], seeds=[1, 2])
    assert len(runs) == 2
    assert {"seed", "accuracy", "qwk", "mae", "severe"}.issubset(runs.columns)


def test_select_evenly_spaced_covers_collection_when_limited():
    selected = select_evenly_spaced(list(range(100)), 5)

    assert selected[0] == 0
    assert selected[-1] == 99
    assert len(selected) == 5
