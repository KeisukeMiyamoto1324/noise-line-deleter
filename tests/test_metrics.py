from __future__ import annotations

import pytest

from corpus_refiner_jp.metrics import compute_metrics


def test_compute_metrics_reports_keep_positive_metrics() -> None:
    metrics = compute_metrics([1, 0, 1, 0], [1, 0, 0, 0])

    assert metrics["precision_keep"] == 1.0
    assert metrics["recall_keep"] == 0.5
    assert metrics["f1_keep"] == pytest.approx(2 / 3)
    assert set(metrics) == {"precision_keep", "recall_keep", "f1_keep", "accuracy", "macro_f1"}
