import numpy as np

from scripts.run_secondorder_study import compute_secondorder_metrics


def test_flat_arm_freshness_metrics_are_class_derived_not_head_derived():
    class_to_freshness = np.array([0, 1, 2, 0])
    metrics = compute_secondorder_metrics(
        class_preds=np.array([0, 1, 2]),
        class_labels=np.array([0, 2, 1]),
        true_freshness=np.array([0, 2, 1]),
        class_to_freshness=class_to_freshness,
        head_freshness_preds=np.array([2, 2, 2]),
    )

    assert metrics["class_acc"] == 33.33
    assert metrics["freshness_acc_classderived"] == 33.33
    assert metrics["freshness_acc_head"] == 33.33
    assert metrics["freshness_severe_classderived"] == 0
    assert metrics["freshness_severe_head"] == 1
