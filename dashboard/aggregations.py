"""Aggregation utilities for dashboard analytics and comparisons."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence

from . import schemas
from .utils import grade_group_sort_key

LETTER_SET = {"A", "B", "C", "D", "E"}
CORRECTNESS_LABELS = {
    "all": "All filtered rows",
    "incorrect": "Incorrect",
    "correct": "Correct",
    "unknown": "Unknown/Skipped",
}
CORRECTNESS_ORDER = ["all", "incorrect", "correct", "unknown"]


def _empty_bucket() -> Dict[str, object]:
    return {
        "count": 0,
        "correct_count": 0,
        "multimodal_true": 0,
        "points": [],
        "points_earned": [],
        "latency": [],
        "tokens": [],
        "reasoning_tokens": [],
        "cost": [],
        "group_counter": Counter(),
        "year_counter": Counter(),
        "language_counter": Counter(),
        "reasoning_counter": Counter(),
    }


def _extend_bucket(bucket: Dict[str, object], row: schemas.RowRecord) -> None:
    bucket["count"] += 1
    if row.is_correct is True:
        bucket["correct_count"] += 1
    if row.multimodal:
        bucket["multimodal_true"] += 1

    if row.points is not None:
        bucket["points"].append(float(row.points))
    if row.points_earned is not None:
        bucket["points_earned"].append(float(row.points_earned))
    if row.latency_ms is not None:
        bucket["latency"].append(float(row.latency_ms))
    if row.total_tokens is not None:
        bucket["tokens"].append(float(row.total_tokens))
    if row.reasoning_tokens is not None:
        bucket["reasoning_tokens"].append(float(row.reasoning_tokens))
    if row.cost_usd is not None:
        bucket["cost"].append(float(row.cost_usd))

    group_value = str(row.group) if row.group not in (None, "") else "Unknown"
    bucket["group_counter"][group_value] += 1

    year_value = str(row.year) if row.year not in (None, "") else "Unknown"
    bucket["year_counter"][year_value] += 1

    if row.language:
        bucket["language_counter"][str(row.language)] += 1
    else:
        bucket["language_counter"]["Unknown"] += 1

    if row.reasoning_mode:
        bucket["reasoning_counter"][str(row.reasoning_mode)] += 1
    else:
        bucket["reasoning_counter"]["Unknown"] += 1


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _numeric_summary(values: Sequence[float]) -> schemas.NumericSummary:
    if not values:
        return schemas.NumericSummary()
    ordered = sorted(float(v) for v in values)
    count = len(ordered)
    mean_value = sum(ordered) / count if count else None
    median_value = statistics.median(ordered) if count else None
    min_value = ordered[0]
    max_value = ordered[-1]
    if count > 1:
        try:
            quantiles = statistics.quantiles(ordered, n=4, method="inclusive")
            p25, p75 = quantiles[0], quantiles[2]
        except (statistics.StatisticsError, IndexError):
            p25 = ordered[0]
            p75 = ordered[-1]
        stddev = statistics.pstdev(ordered)
    else:
        p25 = ordered[0]
        p75 = ordered[0]
        stddev = 0.0
    return schemas.NumericSummary(
        count=count,
        mean=_round(mean_value),
        median=_round(median_value),
        min=_round(min_value),
        max=_round(max_value),
        p25=_round(p25),
        p75=_round(p75),
        stddev=_round(stddev),
    )


def _distribution_from_counter(
    counter: Counter,
    total: int,
    *,
    sort_key=None,
    limit: int | None = None,
) -> List[schemas.DistributionBucket]:
    if total <= 0 or not counter:
        return []
    items = list(counter.items())
    if sort_key is not None:
        items.sort(key=sort_key)
    else:
        items.sort(key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        items = items[:limit]
    result: List[schemas.DistributionBucket] = []
    for key, count in items:
        label = str(key)
        if not label or label.lower() == "none":
            label = "Unknown"
        percentage = count / total if total else 0.0
        result.append(
            schemas.DistributionBucket(
                key=str(key),
                label=label,
                count=int(count),
                percentage=_round(percentage),
            )
        )
    return result


def _build_subset_metrics(
    rows: Sequence[schemas.RowRecord],
) -> List[schemas.SubsetMetrics]:
    buckets: Dict[str, Dict[str, object]] = {
        key: _empty_bucket() for key in CORRECTNESS_LABELS.keys()
    }

    for row in rows:
        _extend_bucket(buckets["all"], row)
        if row.is_correct is True:
            target_key = "correct"
        elif row.is_correct is False:
            target_key = "incorrect"
        else:
            target_key = "unknown"
        _extend_bucket(buckets[target_key], row)

    total_count = buckets["all"]["count"]
    metrics: List[schemas.SubsetMetrics] = []

    for key in CORRECTNESS_ORDER:
        bucket = buckets[key]
        count = bucket["count"]
        share = (count / total_count) if total_count else 0.0
        accuracy = (bucket["correct_count"] / count) if count else None
        multimodal_share = (bucket["multimodal_true"] / count) if count else None

        points_summary = _numeric_summary(bucket["points"])
        points_earned_summary = _numeric_summary(bucket["points_earned"])
        latency_summary = _numeric_summary(bucket["latency"])
        tokens_summary = _numeric_summary(bucket["tokens"])
        reasoning_tokens_summary = _numeric_summary(bucket["reasoning_tokens"])
        cost_summary = _numeric_summary(bucket["cost"])

        points_hist = (
            build_histogram(bucket["points"])
            if bucket["points"]
            else schemas.Histogram()
        )
        points_earned_hist = (
            build_histogram(bucket["points_earned"])
            if bucket["points_earned"]
            else schemas.Histogram()
        )

        metrics.append(
            schemas.SubsetMetrics(
                key=key,
                label=CORRECTNESS_LABELS[key],
                count=int(count),
                share=_round(share),
                accuracy=_round(accuracy) if accuracy is not None else None,
                multimodal_share=_round(multimodal_share)
                if multimodal_share is not None
                else None,
                points_summary=points_summary,
                points_earned_summary=points_earned_summary,
                latency_summary=latency_summary,
                tokens_summary=tokens_summary,
                reasoning_tokens_summary=reasoning_tokens_summary,
                cost_summary=cost_summary,
                grade_distribution=_distribution_from_counter(
                    bucket["group_counter"],
                    count,
                    sort_key=lambda item: grade_group_sort_key(item[0]),
                ),
                year_distribution=_distribution_from_counter(
                    bucket["year_counter"],
                    count,
                ),
                language_distribution=_distribution_from_counter(
                    bucket["language_counter"],
                    count,
                    limit=8,
                ),
                reasoning_mode_distribution=_distribution_from_counter(
                    bucket["reasoning_counter"],
                    count,
                    limit=8,
                ),
                points_hist=points_hist,
                points_earned_hist=points_earned_hist,
            )
        )

    return metrics


def compute_aggregates(rows: Sequence[schemas.RowRecord]) -> schemas.AggregatesResponse:
    breakdown_group: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "correct": 0,
            "points_total": 0.0,
            "points_earned": 0.0,
        }
    )
    breakdown_year: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "correct": 0,
            "points_total": 0.0,
            "points_earned": 0.0,
        }
    )

    def record_breakdown(
        target: Dict[str, Dict[str, float]], key: str | None, row: schemas.RowRecord
    ) -> None:
        if not key:
            return
        entry = target[key]
        entry["count"] += 1
        if row.is_correct:
            entry["correct"] += 1
        if row.points is not None:
            entry["points_total"] += float(row.points)
        if row.points_earned is not None:
            entry["points_earned"] += float(row.points_earned)

    for row in rows:
        record_breakdown(breakdown_group, row.group, row)
        record_breakdown(breakdown_year, row.year, row)

    def finalize_breakdown(
        mapping: Dict[str, Dict[str, float]],
    ) -> Dict[str, schemas.BreakdownEntry]:
        result: Dict[str, schemas.BreakdownEntry] = {}
        for key in sorted(mapping.keys(), key=grade_group_sort_key):
            entry = mapping[key]
            count = int(entry["count"])
            accuracy = (entry["correct"] / count) if count else 0.0
            total_points = entry["points_total"]
            pwa = (entry["points_earned"] / total_points) if total_points else 0.0
            result[key] = schemas.BreakdownEntry(
                count=count,
                accuracy=round(accuracy, 4),
                points_weighted_accuracy=round(pwa, 4),
            )
        return result

    confusion_matrix: Dict[str, Dict[str, int]] = {
        letter: {p: 0 for p in LETTER_SET} for letter in LETTER_SET
    }
    predicted_counts: Counter[str] = Counter()
    latency_values: List[float] = []
    token_values: List[float] = []
    reasoning_values: List[float] = []
    warning_counts: Counter[str] = Counter()

    for row in rows:
        answer = row.answer.upper() if row.answer else None
        predicted = row.predicted.upper() if row.predicted else None
        if predicted:
            predicted_counts[predicted] += 1
        if answer in LETTER_SET and predicted in LETTER_SET:
            confusion_matrix[answer][predicted] += 1
        if row.latency_ms is not None:
            latency_values.append(float(row.latency_ms))
        if row.total_tokens is not None:
            token_values.append(float(row.total_tokens))
        if row.reasoning_tokens is not None:
            reasoning_values.append(float(row.reasoning_tokens))
        if row.warnings:
            warning_counts.update(w for w in row.warnings if w)

    latency_hist = build_histogram(latency_values)
    tokens_hist = build_histogram(token_values)
    reasoning_hist = build_histogram(reasoning_values)

    subset_metrics = _build_subset_metrics(rows)

    warning_toplist = [
        schemas.WarningBreakdown(warning_type=w, count=c)
        for w, c in warning_counts.most_common(10)
    ]
    return schemas.AggregatesResponse(
        subset_metrics=subset_metrics,
        breakdown_by_group=finalize_breakdown(breakdown_group),
        breakdown_by_year=finalize_breakdown(breakdown_year),
        confusion_matrix=confusion_matrix,
        latency_hist=latency_hist,
        tokens_hist=tokens_hist,
        reasoning_tokens_hist=reasoning_hist,
        predicted_counts=dict(predicted_counts),
        warning_toplist=warning_toplist,
    )


def build_histogram(values: Sequence[float]) -> schemas.Histogram:
    if not values:
        return schemas.Histogram()
    values = sorted(float(v) for v in values)
    min_val = values[0]
    max_val = values[-1]
    if math.isclose(min_val, max_val):
        return schemas.Histogram(
            bins=[min_val, max_val], counts=[len(values)], min=min_val, max=max_val
        )
    bucket_count = min(20, max(5, int(math.sqrt(len(values)))))
    step = (max_val - min_val) / bucket_count or 1.0
    bins = [min_val + i * step for i in range(bucket_count + 1)]
    counts = [0 for _ in range(bucket_count)]
    for value in values:
        idx = min(int((value - min_val) / step), bucket_count - 1)
        counts[idx] += 1
    return schemas.Histogram(bins=bins, counts=counts, min=min_val, max=max_val)


def _confusion_for_rows(
    rows: Mapping[str, schemas.RowRecord] | Sequence[schemas.RowRecord],
) -> Dict[str, Dict[str, int]]:
    matrix = {letter: {p: 0 for p in LETTER_SET} for letter in LETTER_SET}
    if isinstance(rows, Mapping):
        iterable = rows.values()
    else:
        iterable = rows
    for row in iterable:
        answer = row.answer.upper() if row.answer else None
        predicted = row.predicted.upper() if row.predicted else None
        if answer in LETTER_SET and predicted in LETTER_SET:
            matrix[answer][predicted] += 1
    return matrix


def compute_compare(
    run_a: schemas.RunSummary,
    run_b: schemas.RunSummary,
    metrics_a: schemas.RunMetrics,
    metrics_b: schemas.RunMetrics,
    rows_a: Mapping[str, schemas.RowRecord],
    rows_b: Mapping[str, schemas.RowRecord],
) -> schemas.CompareResponse:
    metric_keys = [
        "answered_count",
        "skipped_count",
        "failed_count",
        "accuracy",
        "points_weighted_accuracy",
        "mean_latency_ms",
        "median_latency_ms",
        "mean_total_tokens",
        "total_cost_usd_known",
    ]

    metrics: List[schemas.CompareMetricDelta] = []
    for key in metric_keys:
        value_a = getattr(metrics_a, key, None)
        value_b = getattr(metrics_b, key, None)
        delta = None
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            delta = value_b - value_a
        metrics.append(
            schemas.CompareMetricDelta(
                metric=key,
                run_a=float(value_a) if isinstance(value_a, (int, float)) else None,
                run_b=float(value_b) if isinstance(value_b, (int, float)) else None,
                delta=float(delta) if delta is not None else None,
            )
        )

    breakdown_deltas: Dict[str, List[schemas.CompareMetricDelta]] = {}
    mapping_pairs = {
        "group": (metrics_a.breakdown_by_group, metrics_b.breakdown_by_group),
        "year": (metrics_a.breakdown_by_year, metrics_b.breakdown_by_year),
    }
    for name, (map_a, map_b) in mapping_pairs.items():
        sort_key = grade_group_sort_key
        keys = sorted(set(map_a.keys()) | set(map_b.keys()), key=sort_key)
        series: List[schemas.CompareMetricDelta] = []
        for key in keys:
            entry_a = map_a.get(key)
            entry_b = map_b.get(key)
            val_a = entry_a.accuracy if entry_a else None
            val_b = entry_b.accuracy if entry_b else None
            delta = None
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
            series.append(
                schemas.CompareMetricDelta(
                    metric=key,
                    run_a=val_a,
                    run_b=val_b,
                    delta=delta,
                )
            )
        breakdown_deltas[name] = series

    common_ids = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    row_deltas: List[schemas.CompareRowDelta] = []
    for row_id in common_ids:
        row_a = rows_a[row_id]
        row_b = rows_b[row_id]
        delta_points = None
        if row_a.points_earned is not None and row_b.points_earned is not None:
            delta_points = row_b.points_earned - row_a.points_earned
        delta_latency = None
        if row_a.latency_ms is not None and row_b.latency_ms is not None:
            delta_latency = row_b.latency_ms - row_a.latency_ms
        delta_tokens = None
        if row_a.total_tokens is not None and row_b.total_tokens is not None:
            delta_tokens = row_b.total_tokens - row_a.total_tokens
        row_deltas.append(
            schemas.CompareRowDelta(
                id=row_id,
                run_a_correct=row_a.is_correct,
                run_b_correct=row_b.is_correct,
                run_a_predicted=row_a.predicted,
                run_b_predicted=row_b.predicted,
                run_a_points=row_a.points_earned,
                run_b_points=row_b.points_earned,
                delta_points=delta_points,
                delta_latency_ms=delta_latency,
                delta_total_tokens=delta_tokens,
            )
        )

    return schemas.CompareResponse(
        run_a=run_a,
        run_b=run_b,
        metrics=metrics,
        breakdown_deltas=breakdown_deltas,
        row_deltas=row_deltas,
        confusion_matrices={
            "run_a": _confusion_for_rows(rows_a),
            "run_b": _confusion_for_rows(rows_b),
        },
    )


__all__ = ["compute_aggregates", "compute_compare", "build_histogram"]
