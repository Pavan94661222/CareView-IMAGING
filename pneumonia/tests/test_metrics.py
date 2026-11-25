import numpy as np

from pneumonia.metrics import find_best_threshold


def test_accuracy_threshold_tie_prefers_specificity():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.3, 0.1, 0.9, 0.7])

    threshold, metrics = find_best_threshold(
        y_true, y_prob, metric="accuracy", grid=np.array([0.2, 0.8])
    )

    assert threshold == 0.8
    assert metrics["accuracy"] == 0.75
    assert metrics["specificity"] == 1.0
