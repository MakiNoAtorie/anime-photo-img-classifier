"""Evaluation metrics, breakdowns and reports (M3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pvd.evaluate import coverage_accuracy, evaluate_arrays, render_report


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [f"data/test/img{i}.png" for i in range(n)],
            "format": ["PNG" if i % 2 == 0 else "JPEG" for i in range(n)],
            "has_alpha": [i % 3 == 0 for i in range(n)],
            "width": [400 + 600 * (i % 4) for i in range(n)],
            "height": [300 + 400 * (i % 4) for i in range(n)],
            "source_dir": ["a" if i < n // 2 else "b" for i in range(n)],
        }
    )


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    result = evaluate_arrays(y, probs, ["photo", "drawing"], _frame(4), ["anime", None, "anime", None])
    assert result.accuracy == 1.0
    assert result.balanced_accuracy == 1.0
    assert result.auc == 1.0
    assert result.confusion == [[2, 0], [0, 2]]
    assert result.per_tag["anime"]["n"] == 2
    assert result.per_tag["anime"]["accuracy"] == 1.0


def test_confusion_and_errors() -> None:
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.6, 0.4], [0.2, 0.8]])
    result = evaluate_arrays(y, probs, ["photo", "drawing"], _frame(4))
    assert result.confusion == [[1, 1], [1, 1]]
    assert len(result.worst_errors) == 2
    # Highest-confidence mistakes first.
    assert result.worst_errors[0]["confidence"] >= result.worst_errors[1]["confidence"]


def test_coverage_accuracy_threshold() -> None:
    # 20 correct at 0.99 and 20 correct at 0.6: a high threshold covers few.
    probs = np.array([[0.99, 0.01]] * 20 + [[0.6, 0.4]] * 20)
    y = np.zeros(40, dtype=int)
    y_pred = probs.argmax(axis=1)
    table, recommended = coverage_accuracy(probs, y_pred, y, target=0.98)
    assert table[0][0] == 0.5
    assert table[0][1] == 1.0  # everything covered at 0.5
    assert recommended == 0.5


def test_bias_breakdowns_present() -> None:
    y = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    result = evaluate_arrays(y, probs, ["photo", "drawing"], _frame(4))
    assert set(result.bias) == {"format", "has_alpha", "resolution", "source_dir"}
    assert "PNG" in result.bias["format"]


def test_render_report_sections() -> None:
    y = np.array([0, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = evaluate_arrays(y, probs, ["photo", "drawing"], _frame(2), [None, None])
    text = render_report("probe", result)
    for section in ("# Evaluation report", "## Per-class", "## Confusion matrix",
                    "## Coverage vs accuracy", "unsure_threshold"):
        assert section in text
