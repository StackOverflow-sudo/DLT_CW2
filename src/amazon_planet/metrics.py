from __future__ import annotations

import numpy as np
import pandas as pd


def fbeta_score_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 2.0,
    eps: float = 1e-9,
) -> float:
    """Compute mean sample-wise F-beta score for multi-label predictions."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.logical_and(y_true, y_pred).sum(axis=1)
    fp = np.logical_and(~y_true, y_pred).sum(axis=1)
    fn = np.logical_and(y_true, ~y_pred).sum(axis=1)

    beta2 = beta**2
    scores = ((1 + beta2) * tp) / ((1 + beta2) * tp + beta2 * fn + fp + eps)
    return float(scores.mean())


def optimise_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.01)

    best_threshold = 0.2
    best_score = -1.0

    for threshold in thresholds:
        score = fbeta_score_multilabel(y_true, probabilities >= threshold)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Compute label-wise precision, recall, F1, F2, and support."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    rows = []
    for index, label in enumerate(labels):
        true_col = y_true[:, index]
        pred_col = y_pred[:, index]

        tp = int(np.logical_and(true_col, pred_col).sum())
        fp = int(np.logical_and(~true_col, pred_col).sum())
        fn = int(np.logical_and(true_col, ~pred_col).sum())
        support = int(true_col.sum())

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        f2 = (5 * precision * recall) / (4 * precision + recall + eps)

        rows.append(
            {
                "label": label,
                "support": support,
                "predicted": int(pred_col.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
            }
        )

    return pd.DataFrame(rows).sort_values("f2", ascending=False)
