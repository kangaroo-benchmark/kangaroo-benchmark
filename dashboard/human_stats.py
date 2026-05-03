"""Summary statistics utilities for LLM vs human comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from . import schemas
from .human_compare import compute_run_human_comparison
from .human_baseline import HumanBaselineIndex, HumanGradeStats
from .data_index import RunIndex

ComparatorType = Literal["average", "best"]
WeightMode = Literal["micro", "macro"]

_DEFAULT_THRESHOLDS = (0.05, 0.10)  # percentage-point thresholds expressed as fractions
_TOP_LIMIT = 5
_TOLERANCE = 1e-9


def compute_run_summary(
    run_id: str,
    index: RunIndex,
    humans: HumanBaselineIndex,
    *,
    comparator: ComparatorType = "average",
    weight_mode: WeightMode = "micro",
    late_year_strategy: str = "best",
    top_limit: int = _TOP_LIMIT,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
) -> schemas.HumanComparisonSummary:
    response = compute_run_human_comparison(
        run_id,
        index,
        humans,
        late_year_strategy=late_year_strategy,
    )
    entries = [(run_id, entry) for entry in response.entries]
    return _summarize_entries(
        entries,
        run_ids=[run_id],
        comparator=comparator,
        weight_mode=weight_mode,
        top_limit=top_limit,
        thresholds=thresholds,
    )


def compute_cohort_summary(
    run_ids: Sequence[str],
    index: RunIndex,
    humans: HumanBaselineIndex,
    *,
    comparator: ComparatorType = "average",
    weight_mode: WeightMode = "micro",
    late_year_strategy: str = "best",
    top_limit: int = _TOP_LIMIT,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
) -> schemas.HumanComparisonSummary:
    entries: List[Tuple[str, schemas.HumanRunGradeComparison]] = []
    for run_id in run_ids:
        response = compute_run_human_comparison(
            run_id,
            index,
            humans,
            late_year_strategy=late_year_strategy,
        )
        entries.extend((run_id, entry) for entry in response.entries)
    return _summarize_entries(
        entries,
        run_ids=list(run_ids),
        comparator=comparator,
        weight_mode=weight_mode,
        top_limit=top_limit,
        thresholds=thresholds,
    )


def compute_human_baseline_summary(
    humans: HumanBaselineIndex,
    *,
    comparator: ComparatorType = "average",
) -> schemas.HumanBaselineSummary:
    best_year: Optional[schemas.HumanComparisonBestYear] = None
    best_grade: Optional[schemas.HumanComparisonBestGrade] = None

    year_totals: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"score_sum": 0.0, "weight": 0.0}
    )
    grade_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"score_sum": 0.0, "weight": 0.0, "label": ""}
    )

    for year in humans.available_years():
        try:
            human_year = humans.get_year(year)
        except KeyError:
            continue
        for grade_id in human_year.grade_ids():
            grade = human_year.grade(grade_id)
            score = _select_human_score(grade, comparator)
            if score is None or grade.max_points <= 0:
                continue
            score_pct = score / grade.max_points
            weight = float(grade.total_count) if grade.total_count else 1.0
            year_totals[year]["score_sum"] += score_pct * weight
            year_totals[year]["weight"] += weight

            g_key = f"{grade_id}:{grade.label}"
            grade_totals[g_key]["score_sum"] += score_pct * weight
            grade_totals[g_key]["weight"] += weight
            grade_totals[g_key]["label"] = grade.label

    best_year = _resolve_best_year_from_humans(year_totals)
    best_grade = _resolve_best_grade_from_humans(grade_totals)

    return schemas.HumanBaselineSummary(
        comparator=comparator,
        best_year=best_year,
        best_grade=best_grade,
        notes=[],
    )


def _summarize_entries(
    entries: Sequence[Tuple[str, schemas.HumanRunGradeComparison]],
    *,
    run_ids: Sequence[str],
    comparator: ComparatorType,
    weight_mode: WeightMode,
    top_limit: int,
    thresholds: Sequence[float],
) -> schemas.HumanComparisonSummary:
    run_ids_sorted = sorted(set(run_ids))

    if not entries:
        return schemas.HumanComparisonSummary(
            comparator=comparator,
            weight_mode=weight_mode,
            run_ids=run_ids_sorted,
        )

    run_cell_counts: Counter[str] = Counter(run_id for run_id, _ in entries)
    if not run_ids_sorted:
        run_ids_sorted = sorted(run_cell_counts.keys())

    run_weight = 1.0 / len(run_cell_counts) if run_cell_counts else 1.0

    threshold_stats: Dict[float, Dict[str, int]] = {
        thr: {"llm": 0, "human": 0} for thr in thresholds
    }

    total_cells = 0
    llm_win_count = 0
    human_win_count = 0
    tie_count = 0

    percentile_sum = 0.0
    percentile_weight = 0.0
    z_sum = 0.0
    z_weight = 0.0
    gap_pct_sum = 0.0
    gap_weight = 0.0

    year_stats: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {
            "percentile_sum": 0.0,
            "percentile_weight": 0.0,
            "gap_sum": 0.0,
            "gap_weight": 0.0,
            "score_sum": 0.0,
            "score_weight": 0.0,
            "llm_pct_sum": 0.0,
            "llm_pct_weight": 0.0,
        }
    )
    grade_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "percentile_sum": 0.0,
            "percentile_weight": 0.0,
            "gap_sum": 0.0,
            "gap_weight": 0.0,
            "score_sum": 0.0,
            "score_weight": 0.0,
            "llm_pct_sum": 0.0,
            "llm_pct_weight": 0.0,
            "label": "",
        }
    )

    llm_positive_cells: List[schemas.HumanComparisonTopCell] = []
    human_positive_cells: List[schemas.HumanComparisonTopCell] = []

    for run_id, entry in entries:
        human_score = _select_human_score(entry, comparator)
        llm_score_pct = _score_pct(entry.llm_total, entry.llm_max)
        human_score_pct = _score_pct(human_score, entry.llm_max)

        gap_points = None
        gap_pct = None

        if human_score is not None and entry.llm_total is not None:
            gap_points = entry.llm_total - human_score
        if llm_score_pct is not None and human_score_pct is not None:
            gap_pct = llm_score_pct - human_score_pct

        weight = _weight_for_entry(
            run_id, entry, weight_mode, run_cell_counts, run_weight
        )

        percentile = entry.human_percentile
        if percentile is not None:
            percentile_sum += percentile * weight
            percentile_weight += weight
            year_stats[entry.year]["percentile_sum"] += percentile * weight
            year_stats[entry.year]["percentile_weight"] += weight
            grade_stats[entry.grade_id]["percentile_sum"] += percentile * weight
            grade_stats[entry.grade_id]["percentile_weight"] += weight

        if entry.z_score is not None:
            z_sum += entry.z_score * weight
            z_weight += weight

        if gap_pct is not None:
            gap_pct_sum += gap_pct * weight
            gap_weight += weight
            year_stats[entry.year]["gap_sum"] += gap_pct * weight
            year_stats[entry.year]["gap_weight"] += weight
            grade_stats[entry.grade_id]["gap_sum"] += gap_pct * weight
            grade_stats[entry.grade_id]["gap_weight"] += weight

        if human_score_pct is not None:
            year_stats[entry.year]["score_sum"] += human_score_pct * weight
            year_stats[entry.year]["score_weight"] += weight
            grade_stats[entry.grade_id]["score_sum"] += human_score_pct * weight
            grade_stats[entry.grade_id]["score_weight"] += weight

        if llm_score_pct is not None:
            year_stats[entry.year]["llm_pct_sum"] += llm_score_pct * weight
            year_stats[entry.year]["llm_pct_weight"] += weight
            grade_stats[entry.grade_id]["llm_pct_sum"] += llm_score_pct * weight
            grade_stats[entry.grade_id]["llm_pct_weight"] += weight

        grade_stats[entry.grade_id]["label"] = entry.grade_label

        cell_summary = schemas.HumanComparisonTopCell(
            year=entry.year,
            grade_id=entry.grade_id,
            grade_label=entry.grade_label,
            members=list(entry.members),
            llm_score_pct=llm_score_pct,
            human_score_pct=human_score_pct,
            gap_score_pct=gap_pct,
            gap_points=gap_points,
            human_percentile=entry.human_percentile,
            z_score=entry.z_score,
            llm_total=entry.llm_total,
            human_score=human_score,
        )

        if gap_points is not None:
            if gap_points > _TOLERANCE:
                llm_win_count += 1
                llm_positive_cells.append(cell_summary)
            elif gap_points < -_TOLERANCE:
                human_win_count += 1
                human_positive_cells.append(cell_summary)
            else:
                tie_count += 1

        if gap_pct is not None:
            for thr in thresholds:
                if gap_pct >= thr:
                    threshold_stats[thr]["llm"] += 1
                elif gap_pct <= -thr:
                    threshold_stats[thr]["human"] += 1

        total_cells += 1

    llm_positive_cells.sort(key=lambda cell: cell.gap_score_pct or 0.0, reverse=True)
    human_positive_cells.sort(key=lambda cell: cell.gap_score_pct or 0.0)

    top_llm = llm_positive_cells[:top_limit]
    top_human = human_positive_cells[:top_limit]

    avg_percentile = (
        (percentile_sum / percentile_weight) if percentile_weight > 0 else None
    )
    avg_z = (z_sum / z_weight) if z_weight > 0 else None
    avg_gap_pct = (gap_pct_sum / gap_weight) if gap_weight > 0 else None

    best_year = _resolve_best_year(year_stats)
    best_grade = _resolve_best_grade(grade_stats)

    threshold_breakdown = [
        schemas.HumanComparisonThresholdBreakdown(
            threshold_pp=thr * 100.0,
            llm_win_count=counts["llm"],
            human_win_count=counts["human"],
        )
        for thr, counts in threshold_stats.items()
    ]

    return schemas.HumanComparisonSummary(
        run_ids=run_ids_sorted,
        comparator=comparator,
        weight_mode=weight_mode,
        total_cells=total_cells,
        llm_win_count=llm_win_count,
        human_win_count=human_win_count,
        tie_count=tie_count,
        avg_percentile=avg_percentile,
        avg_z_score=avg_z,
        avg_gap_pct=avg_gap_pct,
        top_llm_wins=top_llm,
        top_human_wins=top_human,
        best_year=best_year,
        best_grade=best_grade,
        threshold_breakdown=threshold_breakdown,
    )


def _select_human_score(
    entry: schemas.HumanRunGradeComparison | HumanGradeStats,
    comparator: ComparatorType,
) -> Optional[float]:
    if comparator == "best":
        value = getattr(entry, "human_best", None)
        if value is None:
            value = getattr(entry, "best_score_estimate", None)
        return value
    # For averages prefer mean_estimate, fallback to avg_score_reported if available.
    mean_value = getattr(entry, "human_mean", None)
    if mean_value is None:
        mean_value = getattr(entry, "mean_estimate", None)
    if mean_value is None:
        mean_value = getattr(entry, "avg_score_reported", None)
    return mean_value


def _score_pct(score: Optional[float], max_points: Optional[float]) -> Optional[float]:
    if score is None or max_points is None or max_points <= 0:
        return None
    return score / max_points


def _weight_for_entry(
    run_id: str,
    entry: schemas.HumanRunGradeComparison,
    weight_mode: WeightMode,
    run_cell_counts: Counter[str],
    run_weight: float,
) -> float:
    if weight_mode == "macro":
        cell_count = run_cell_counts.get(run_id, 1)
        if cell_count <= 0:
            cell_count = 1
        return run_weight / cell_count
    # micro weighting by maximum available points (plus fallback)
    if entry.llm_max and entry.llm_max > 0:
        return entry.llm_max
    if entry.llm_points_available and entry.llm_points_available > 0:
        return entry.llm_points_available
    return 1.0


def _resolve_best_year(
    stats: Dict[int, Dict[str, float]],
) -> Optional[schemas.HumanComparisonBestYear]:
    best_entry: Optional[Tuple[int, float]] = None
    for year, data in stats.items():
        weight = data["percentile_weight"]
        gap_weight = data["gap_weight"]
        percentile_avg = (data["percentile_sum"] / weight) if weight > 0 else None
        gap_avg = (data["gap_sum"] / gap_weight) if gap_weight > 0 else None
        metric = (
            percentile_avg
            if percentile_avg is not None
            else (gap_avg if gap_avg is not None else -float("inf"))
        )
        if percentile_avg is None and gap_avg is None:
            continue
        if best_entry is None or metric > best_entry[1]:
            best_entry = (year, metric)

    if best_entry is None:
        return None

    year = best_entry[0]
    data = stats[year]
    weight = data["percentile_weight"]
    gap_weight = data["gap_weight"]
    score_weight = data["score_weight"]

    return schemas.HumanComparisonBestYear(
        year=year,
        avg_percentile=(data["percentile_sum"] / weight) if weight > 0 else None,
        avg_gap_pct=(data["gap_sum"] / gap_weight) if gap_weight > 0 else None,
        avg_score_pct=(data["score_sum"] / score_weight) if score_weight > 0 else None,
    )


def _resolve_best_grade(
    stats: Dict[str, Dict[str, float]],
) -> Optional[schemas.HumanComparisonBestGrade]:
    best_entry: Optional[Tuple[str, float]] = None
    for grade_id, data in stats.items():
        weight = data["percentile_weight"]
        gap_weight = data["gap_weight"]
        score_weight = data["score_weight"]
        percentile_avg = (data["percentile_sum"] / weight) if weight > 0 else None
        gap_avg = (data["gap_sum"] / gap_weight) if gap_weight > 0 else None
        metric = (
            percentile_avg
            if percentile_avg is not None
            else (gap_avg if gap_avg is not None else -float("inf"))
        )
        if percentile_avg is None and gap_avg is None:
            continue
        if best_entry is None or metric > best_entry[1]:
            best_entry = (grade_id, metric)

    if best_entry is None:
        return None

    grade_id = best_entry[0]
    data = stats[grade_id]
    label = data.get("label", grade_id)
    weight = data["percentile_weight"]
    gap_weight = data["gap_weight"]
    score_weight = data["score_weight"]

    return schemas.HumanComparisonBestGrade(
        grade_id=grade_id,
        grade_label=label,
        avg_percentile=(data["percentile_sum"] / weight) if weight > 0 else None,
        avg_gap_pct=(data["gap_sum"] / gap_weight) if gap_weight > 0 else None,
        avg_score_pct=(data["score_sum"] / score_weight) if score_weight > 0 else None,
    )


def _resolve_best_year_from_humans(
    year_totals: Dict[int, Dict[str, float]],
) -> Optional[schemas.HumanComparisonBestYear]:
    best: Optional[Tuple[int, float]] = None
    for year, data in year_totals.items():
        if data["weight"] <= 0:
            continue
        avg_score = data["score_sum"] / data["weight"]
        if best is None or avg_score > best[1]:
            best = (year, avg_score)
    if best is None:
        return None
    year, avg = best
    return schemas.HumanComparisonBestYear(
        year=year,
        avg_score_pct=avg,
    )


def _resolve_best_grade_from_humans(
    grade_totals: Dict[str, Dict[str, float]],
) -> Optional[schemas.HumanComparisonBestGrade]:
    best: Optional[Tuple[str, float]] = None
    for key, data in grade_totals.items():
        if data["weight"] <= 0:
            continue
        avg_score = data["score_sum"] / data["weight"]
        if best is None or avg_score > best[1]:
            best = (key, avg_score)
    if best is None:
        return None
    key, avg = best
    label = grade_totals[key].get("label", key.split(":", 1)[-1])
    grade_id = key.split(":", 1)[0]
    return schemas.HumanComparisonBestGrade(
        grade_id=grade_id,
        grade_label=label,
        avg_score_pct=avg,
    )


__all__ = [
    "compute_run_summary",
    "compute_cohort_summary",
    "compute_human_baseline_summary",
]
