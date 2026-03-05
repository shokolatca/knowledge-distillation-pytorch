"""Metrics for language classification."""

from __future__ import annotations

from typing import Dict

import numpy as np


def accuracy_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    if targets.size == 0:
        return 0.0
    return float((predictions == targets).sum() / targets.size)


def macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    f1_scores = []
    for class_id in range(num_classes):
        pred_pos = predictions == class_id
        true_pos = targets == class_id

        tp = np.logical_and(pred_pos, true_pos).sum()
        fp = np.logical_and(pred_pos, np.logical_not(true_pos)).sum()
        fn = np.logical_and(np.logical_not(pred_pos), true_pos).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(predictions, targets),
        "macro_f1": macro_f1(predictions, targets, num_classes=num_classes),
    }
