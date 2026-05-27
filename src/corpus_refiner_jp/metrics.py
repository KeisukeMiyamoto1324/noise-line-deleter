from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    return {
        "precision_keep": float(precision),
        "recall_keep": float(recall),
        "f1_keep": float(f1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
