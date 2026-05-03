import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.retry_missing_answers import (  # noqa: E402
    DetectionStats,
    build_subset_dataset,
    detect_after_merge,
    detect_missing_predictions,
    merge_results,
)  # noqa: E402


def make_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["q1", "q2", "q3"],
            "predicted": [None, "B", "  f "],
            "is_correct": [None, True, None],
            "points_earned": [None, 5.0, None],
        }
    )


def make_retry_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["q1", "q3"],
            "predicted": ["C", "DECLINED"],
            "is_correct": [False, False],
            "points_earned": [1.25, 0.0],
            "retry_run_id": ["20250101_model", "20250101_model"],
            "retry_timestamp": ["20250101", "20250101"],
            "retry_attempt": [1, 1],
            "retry_source": ["same_model", "same_model"],
        }
    )


def make_dataset_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["q1", "q2", "q3", "q4"],
            "points": [4.0, 3.0, 5.0, 6.0],
        }
    )


def test_detect_missing_predictions_counts_invalid():
    df = make_results_df()
    stats = detect_missing_predictions(
        df, valid_predictions=["A", "B", "C", "D", "E", "DECLINED"]
    )
    assert isinstance(stats, DetectionStats)
    assert stats.total_rows == 3
    assert stats.missing_rows == 2
    assert stats.invalid_counts == {"F": 1}
    assert stats.question_ids == ["q1", "q3"]


def test_build_subset_dataset_filters_and_sorts():
    dataset = make_dataset_df()
    subset = build_subset_dataset(dataset, "id", ["q3", "q1"])
    assert list(subset["id"]) == ["q1", "q3"]


def test_merge_results_fills_missing_predictions():
    results = make_results_df()
    retry = make_retry_df()
    merged, unresolved = merge_results(
        results,
        retry,
        question_column="id",
        missing_ids=["q1", "q3"],
    )
    assert not unresolved
    q1 = merged.iloc[0]
    assert q1["predicted"] == "C"
    assert q1["predicted_original"] is None
    assert q1["predicted_filled"] == "C"
    assert bool(q1["filled_by_retry"]) is True
    assert q1["filled_run_id"] == "20250101_model"
    assert q1["retry_source"] == "same_model"

    # Existing prediction stays untouched
    q2 = merged.iloc[1]
    assert q2["predicted"] == "B"
    assert pd.isna(q2["predicted_filled"])
    assert bool(q2["filled_by_retry"]) is False


def test_detect_after_merge_reports_remaining_missing():
    results = make_results_df()
    retry = make_retry_df()
    # Remove retry for q3 to simulate unresolved
    retry = retry[retry["id"] == "q1"]
    merged, unresolved = merge_results(
        results, retry, question_column="id", missing_ids=["q1", "q3"]
    )
    assert "q3" in unresolved
    remaining = detect_after_merge(
        merged, valid_predictions=["A", "B", "C", "D", "E", "DECLINED"]
    )
    assert remaining == 1
