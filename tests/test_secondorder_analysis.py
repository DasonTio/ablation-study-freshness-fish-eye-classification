import pandas as pd

from scripts.analyze_secondorder_significance import paired_arm_stats


def test_paired_arm_stats_uses_matched_seeds_and_pre_registered_verdict():
    rows = pd.DataFrame([
        {"arm": "A_gap", "seed": 1, "test_class_acc": 88.0},
        {"arm": "A_gap", "seed": 2, "test_class_acc": 89.0},
        {"arm": "A_gap", "seed": 3, "test_class_acc": 87.0},
        {"arm": "B_raw_bilinear", "seed": 1, "test_class_acc": 89.5},
        {"arm": "B_raw_bilinear", "seed": 2, "test_class_acc": 90.5},
        {"arm": "B_raw_bilinear", "seed": 3, "test_class_acc": 88.5},
    ])

    stats = paired_arm_stats(rows, "A_gap", "B_raw_bilinear")

    assert stats["n"] == 3
    assert stats["mean_delta_pp"] == 1.5
    assert stats["verdict"] == "primary_candidate"
