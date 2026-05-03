"""Comparison helpers between LLM runs and human baselines."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from score_utils import start_points_for_members

from . import schemas
from .data_index import RunIndex, RunNotFoundError
from .human_baseline import HumanBaselineIndex, HumanGradeStats, HumanYear


@dataclass
class _Accumulator:
    total: float = 0.0
    maximum: float = 0.0
    rows: int = 0


def compute_run_human_comparison(
    run_id: str,
    index: RunIndex,
    humans: HumanBaselineIndex,
    *,
    late_year_strategy: str = "best",  # "best" or "average"
) -> schemas.HumanRunComparisonResponse:
    notes: List[str] = []
    try:
        index.get_run(run_id)
    except RunNotFoundError as exc:
        raise exc

    accumulators: Dict[Tuple[int, str], _Accumulator] = {}
    human_year_cache: Dict[int, HumanYear] = {}

    for row in index.iter_results(run_id):
        year = _parse_year(row.year)
        if year is None:
            continue
        if year not in human_year_cache:
            try:
                human_year_cache[year] = humans.get_year(year)
            except KeyError:
                notes.append(f"No human baseline available for year {year}")
                human_year_cache[year] = None  # type: ignore
        human_year = human_year_cache.get(year)
        if not human_year:
            continue

        grade_id = _resolve_grade_id(human_year, row.group)
        if grade_id is None:
            # Try range-based aggregation for 2007+ years
            if year >= 2007 and _is_range_group(row.group):
                grade_id = _resolve_range_group(
                    human_year, row.group, late_year_strategy
                )
                if grade_id is None:
                    notes.append(
                        f"Row {row.id} skipped: group '{row.group}' not mapped to human grade for {year}"
                    )
                    continue
            else:
                notes.append(
                    f"Row {row.id} skipped: group '{row.group}' not mapped to human grade for {year}"
                )
                continue

        points = _safe_float(row.points)
        if points is None or points <= 0:
            notes.append(f"Row {row.id} skipped: missing point value")
            continue
        earned = _safe_float(row.points_earned)
        if earned is None:
            if row.is_correct is True:
                earned = points
            else:
                earned = 0.0

        key = (year, grade_id)
        acc = accumulators.setdefault(key, _Accumulator())
        acc.total += earned
        acc.maximum += points
        acc.rows += 1

    entries: List[schemas.HumanRunGradeComparison] = []
    for (year, grade_id), acc in sorted(
        accumulators.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        human_year = human_year_cache.get(year)
        if not human_year:
            continue
        # Handle synthetic grade IDs that represent aggregated groups
        if grade_id.startswith("_range_aggregated_"):
            grade_stats = _get_aggregated_grade_stats(
                human_year, grade_id, late_year_strategy
            )
            if grade_stats is None:
                notes.append("Failed to aggregate grade stats for range group")
                continue
        else:
            grade_stats = human_year.grade(grade_id)
        start_bonus = start_points_for_members(grade_stats.members)
        llm_points_awarded = acc.total
        llm_points_available = acc.maximum
        llm_total = llm_points_awarded + start_bonus
        observed_max = llm_points_available + start_bonus
        baseline_max = grade_stats.max_points
        llm_max = baseline_max if baseline_max > 0 else observed_max
        llm_score_pct = llm_total / llm_max if llm_max > 0 else None
        human_percentile = grade_stats.percentile(llm_total)
        z_score = grade_stats.z_score(llm_total)

        if baseline_max > 0 and abs(observed_max - baseline_max) > 1e-6:
            notes.append(
                f"Run {run_id} year {year} grade {grade_id}: dataset-derived max {observed_max:.2f} "
                f"differs from human baseline {baseline_max:.2f}"
            )

        bin_comparison = _build_bin_comparison(human_year, grade_stats, llm_total)

        member_overrides: Dict[str, schemas.HumanMemberComparison] = {}
        member_grade_ids = _member_grade_ids_for_stats(
            human_year, grade_stats, grade_id
        )
        for member_grade_id in member_grade_ids:
            try:
                member_grade = human_year.grade(member_grade_id)
            except KeyError:
                if member_grade_id != grade_id:
                    continue
                member_grade = grade_stats
            member_overrides[member_grade_id] = _build_member_override(
                member_grade, llm_total
            )

        entries.append(
            schemas.HumanRunGradeComparison(
                year=year,
                grade_id=grade_id,
                grade_label=grade_stats.label,
                members=list(grade_stats.members),
                max_points=grade_stats.max_points,
                llm_total=llm_total,
                llm_max=llm_max,
                llm_score_pct=llm_score_pct,
                llm_start_points=start_bonus,
                llm_points_awarded=llm_points_awarded,
                llm_points_available=llm_points_available,
                human_percentile=human_percentile,
                z_score=z_score,
                human_mean=grade_stats.mean_estimate,
                human_std=grade_stats.stddev_estimate,
                human_best=grade_stats.best_score_estimate,
                bin_comparison=bin_comparison,
                notes=[],
                member_overrides=member_overrides,
            )
        )

    entries.sort(key=lambda entry: (entry.year, entry.grade_id))
    return schemas.HumanRunComparisonResponse(
        run_id=run_id, entries=entries, notes=sorted(set(notes))
    )


def compute_cohort_human_comparison(
    run_ids: Sequence[str],
    index: RunIndex,
    humans: HumanBaselineIndex,
) -> schemas.HumanCohortComparisonResponse:
    run_ids = list(run_ids)
    run_responses: Dict[str, schemas.HumanRunComparisonResponse] = {}
    cohort_notes: List[str] = []

    for run_id in run_ids:
        try:
            response = compute_run_human_comparison(run_id, index, humans)
            run_responses[run_id] = response
            if response.notes:
                cohort_notes.extend([f"{run_id}: {note}" for note in response.notes])
        except RunNotFoundError as exc:
            cohort_notes.append(str(exc))

    per_group: Dict[
        Tuple[int, str], List[Tuple[str, schemas.HumanRunGradeComparison]]
    ] = defaultdict(list)
    for run_id, response in run_responses.items():
        for entry in response.entries:
            per_group[(entry.year, entry.grade_id)].append((run_id, entry))

    micro_entries: List[schemas.HumanAggregateEntry] = []
    macro_entries: List[schemas.HumanAggregateEntry] = []

    for (year, grade_id), samples in sorted(
        per_group.items(), key=lambda item: item[0]
    ):
        try:
            human_year = humans.get_year(year)
            grade_stats = human_year.grade(grade_id)
        except KeyError:
            continue
        micro_entries.append(
            _aggregate_group(
                samples,
                human_year,
                grade_stats,
                len(run_responses),
                weight_mode="micro",
            )
        )
        macro_entries.append(
            _aggregate_group(
                samples,
                human_year,
                grade_stats,
                len(run_responses),
                weight_mode="macro",
            )
        )

    micro_stats = schemas.HumanAggregateStats(
        cohort_type="micro", entries=micro_entries
    )
    macro_stats = schemas.HumanAggregateStats(
        cohort_type="macro", entries=macro_entries
    )

    return schemas.HumanCohortComparisonResponse(
        run_ids=list(run_responses.keys()),
        micro=micro_stats,
        macro=macro_stats,
        notes=sorted(set(cohort_notes)),
    )


def _build_bin_comparison(
    human_year: HumanYear,
    grade_stats: HumanGradeStats,
    score: float,
) -> List[schemas.HumanBinComparison]:
    human_shares = grade_stats.share_by_bin()
    raw_shares = [0.0 for _ in human_shares]
    smoothed = [0.0 for _ in human_shares]

    bin_index = grade_stats.bin_index_for_score(score)
    if bin_index is not None:
        raw_shares[bin_index] = 1.0

    smoothed_distribution = _smoothed_bin_weights(grade_stats, score)
    if smoothed_distribution:
        smoothed = smoothed_distribution
    elif bin_index is not None:
        smoothed[bin_index] = 1.0

    result: List[schemas.HumanBinComparison] = []
    for idx, bin_obj in enumerate(human_year.bins):
        human_share = human_shares[idx] if idx < len(human_shares) else 0.0
        llm_share = raw_shares[idx] if idx < len(raw_shares) else 0.0
        llm_share_smoothed = smoothed[idx] if idx < len(smoothed) else llm_share
        result.append(
            schemas.HumanBinComparison(
                bin_id=bin_obj.id,
                human_share=human_share,
                llm_share=llm_share,
                delta=llm_share - human_share,
                llm_share_smoothed=llm_share_smoothed,
                delta_smoothed=llm_share_smoothed - human_share,
            )
        )
    return result


def _smoothed_bin_weights(grade_stats: HumanGradeStats, score: float) -> List[float]:
    midpoints = [rng.midpoint for rng in grade_stats.bin_ranges]
    if not midpoints:
        return []
    distances = [abs(score - mid) for mid in midpoints]
    indexed = sorted(enumerate(distances), key=lambda item: item[1])
    shares = [0.0 for _ in midpoints]
    if not indexed:
        return shares
    primary_idx, primary_dist = indexed[0]
    if primary_dist == 0 or len(indexed) == 1:
        shares[primary_idx] = 1.0
        return shares
    secondary_idx, secondary_dist = indexed[1]
    if secondary_dist == 0:
        shares[primary_idx] = shares[secondary_idx] = 0.5
        return shares
    inv_primary = 1.0 / primary_dist
    inv_secondary = 1.0 / secondary_dist
    total = inv_primary + inv_secondary
    if total == 0:
        shares[primary_idx] = 1.0
        return shares
    shares[primary_idx] = inv_primary / total
    shares[secondary_idx] = inv_secondary / total
    return shares


def _aggregate_group(
    samples: Sequence[Tuple[str, schemas.HumanRunGradeComparison]],
    human_year: HumanYear,
    grade_stats: HumanGradeStats,
    run_count: int,
    weight_mode: str,
) -> schemas.HumanAggregateEntry:
    sample_count = len(samples)
    if sample_count == 0:
        return schemas.HumanAggregateEntry(
            year=human_year.year,
            grade_id=grade_stats.id,
            grade_label=grade_stats.label,
            members=list(grade_stats.members),
            run_count=run_count,
            sample_count=0,
            bin_comparison=[],
            notes=[],
        )

    percentiles: List[float] = []
    percentiles_weighted: List[Tuple[float, float]] = []
    score_pcts: List[float] = []
    score_pcts_weighted: List[Tuple[float, float]] = []
    human_mean_pcts: List[float] = []
    human_mean_pcts_weighted: List[Tuple[float, float]] = []
    human_best_pcts: List[float] = []
    human_best_pcts_weighted: List[Tuple[float, float]] = []
    z_scores: List[float] = []
    z_scores_weighted: List[Tuple[float, float]] = []
    best_candidate: Optional[Tuple[str, float]] = None
    worst_candidate: Optional[Tuple[str, float]] = None
    notes: List[str] = []

    raw_share_matrix: List[List[float]] = []
    smooth_share_matrix: List[List[float]] = []
    member_samples: Dict[str, List[Tuple[str, schemas.HumanMemberComparison]]] = (
        defaultdict(list)
    )

    for run_id, entry in samples:
        weight = entry.llm_max if entry.llm_max > 0 else 1.0
        if entry.human_percentile is not None:
            percentiles.append(entry.human_percentile)
            percentiles_weighted.append((entry.human_percentile, weight))
            if best_candidate is None or entry.human_percentile > best_candidate[1]:
                best_candidate = (run_id, entry.human_percentile)
            if worst_candidate is None or entry.human_percentile < worst_candidate[1]:
                worst_candidate = (run_id, entry.human_percentile)
        if entry.llm_score_pct is not None:
            score_pcts.append(entry.llm_score_pct)
            score_pcts_weighted.append((entry.llm_score_pct, weight))
        if entry.human_mean is not None and entry.llm_max:
            human_mean_pct = (
                entry.human_mean / entry.llm_max if entry.llm_max > 0 else None
            )
            if human_mean_pct is not None:
                human_mean_pcts.append(human_mean_pct)
                human_mean_pcts_weighted.append((human_mean_pct, weight))
        if entry.human_best is not None and entry.llm_max:
            human_best_pct = (
                entry.human_best / entry.llm_max if entry.llm_max > 0 else None
            )
            if human_best_pct is not None:
                human_best_pcts.append(human_best_pct)
                human_best_pcts_weighted.append((human_best_pct, weight))
        if entry.z_score is not None:
            z_scores.append(entry.z_score)
            z_scores_weighted.append((entry.z_score, weight))
        raw_share_matrix.append(
            [bin_entry.llm_share for bin_entry in entry.bin_comparison]
        )
        smooth_share_matrix.append(
            [bin_entry.llm_share_smoothed for bin_entry in entry.bin_comparison]
        )
        if entry.notes:
            notes.extend(entry.notes)
        if entry.member_overrides:
            for member_id, override in entry.member_overrides.items():
                member_samples[member_id].append((run_id, override))

    weights = [entry.llm_max if entry.llm_max > 0 else 1.0 for _, entry in samples]
    use_weights = weight_mode == "micro" and any(weight > 0 for weight in weights)

    avg_score_pct = _average(score_pcts_weighted if use_weights else score_pcts)
    avg_percentile = _average(percentiles_weighted if use_weights else percentiles)
    avg_human_mean_pct = _average(
        human_mean_pcts_weighted if use_weights else human_mean_pcts
    )
    avg_human_best_pct = _average(
        human_best_pcts_weighted if use_weights else human_best_pcts
    )
    avg_z = _average(z_scores_weighted if use_weights else z_scores)

    median_percentile, p25_percentile, p75_percentile = _percentile_stats(percentiles)
    min_percentile = min(percentiles) if percentiles else None
    max_percentile = max(percentiles) if percentiles else None

    avg_raw = _average_matrix(raw_share_matrix, weights if use_weights else None)
    avg_smooth = _average_matrix(smooth_share_matrix, weights if use_weights else None)

    human_shares = grade_stats.share_by_bin()
    bin_entries: List[schemas.HumanBinComparison] = []
    for idx, bin_obj in enumerate(human_year.bins):
        human_share = human_shares[idx] if idx < len(human_shares) else 0.0
        llm_share = avg_raw[idx] if idx < len(avg_raw) else 0.0
        llm_share_smoothed = avg_smooth[idx] if idx < len(avg_smooth) else llm_share
        bin_entries.append(
            schemas.HumanBinComparison(
                bin_id=bin_obj.id,
                human_share=human_share,
                llm_share=llm_share,
                delta=llm_share - human_share,
                llm_share_smoothed=llm_share_smoothed,
                delta_smoothed=llm_share_smoothed - human_share,
            )
        )

    return schemas.HumanAggregateEntry(
        year=human_year.year,
        grade_id=grade_stats.id,
        grade_label=grade_stats.label,
        members=list(grade_stats.members),
        run_count=run_count,
        sample_count=sample_count,
        avg_llm_score_pct=avg_score_pct,
        avg_human_mean_pct=avg_human_mean_pct,
        avg_human_best_pct=avg_human_best_pct,
        avg_human_percentile=avg_percentile,
        median_percentile=median_percentile,
        p25_percentile=p25_percentile,
        p75_percentile=p75_percentile,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        avg_z_score=avg_z,
        best_run_id=best_candidate[0] if best_candidate else None,
        worst_run_id=worst_candidate[0] if worst_candidate else None,
        bin_comparison=bin_entries,
        notes=sorted(set(notes)),
        member_overrides=_aggregate_member_overrides(
            human_year,
            grade_stats,
            member_samples,
        ),
    )


def _average(
    values: Iterable[float] | Iterable[Tuple[float, float]],
) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    if isinstance(values[0], tuple):  # type: ignore[index]
        weighted = values  # type: ignore[assignment]
        total_weight = sum(weight for _, weight in weighted)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in weighted) / total_weight
    return sum(values) / len(values)  # type: ignore[return-value]


def _percentile_stats(
    values: Sequence[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not values:
        return (None, None, None)
    median = statistics.median(values)
    if len(values) < 2:
        return (median, None, None)
    try:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    except statistics.StatisticsError:
        q1 = q3 = None
    return (median, q1, q3)


def _average_matrix(
    matrix: Sequence[Sequence[float]], weights: Optional[Sequence[float]]
) -> List[float]:
    if not matrix:
        return []
    length = len(matrix[0])
    result: List[float] = []
    for idx in range(length):
        column = [row[idx] for row in matrix if idx < len(row)]
        if weights is not None and len(weights) == len(matrix) and sum(weights) > 0:
            result.append(
                sum(value * weight for value, weight in zip(column, weights))
                / sum(weights)
            )
        elif column:
            result.append(sum(column) / len(column))
        else:
            result.append(0.0)
    return result


def _member_grade_ids_for_stats(
    human_year: HumanYear,
    grade_stats: HumanGradeStats,
    grade_id: str,
) -> List[str]:
    member_grade_ids: List[str] = []
    for member in grade_stats.members:
        for candidate_id in human_year.grade_ids():
            candidate = human_year.grade(candidate_id)
            if len(candidate.members) == 1 and candidate.members[0] == member:
                member_grade_ids.append(candidate_id)
                break
    if not member_grade_ids:
        member_grade_ids.append(grade_id)
    return member_grade_ids


def _build_member_override(
    grade: HumanGradeStats, score: float
) -> schemas.HumanMemberComparison:
    return schemas.HumanMemberComparison(
        grade_id=grade.id,
        grade_label=grade.label,
        members=list(grade.members),
        max_points=grade.max_points,
        total_count=grade.total_count,
        human_mean=grade.mean_estimate,
        human_std=grade.stddev_estimate,
        human_best=grade.best_score_estimate,
        human_percentile=grade.percentile(score),
        z_score=grade.z_score(score),
    )


def _aggregate_member_overrides(
    human_year: HumanYear,
    grade_stats: HumanGradeStats,
    member_samples: Dict[str, List[Tuple[str, schemas.HumanMemberComparison]]],
) -> Dict[str, schemas.HumanMemberComparison]:
    member_overrides: Dict[str, schemas.HumanMemberComparison] = {}
    member_grade_ids = _member_grade_ids_for_stats(
        human_year, grade_stats, grade_stats.id
    )
    for member_grade_id in member_grade_ids:
        try:
            member_grade = human_year.grade(member_grade_id)
        except KeyError:
            if member_grade_id != grade_stats.id:
                continue
            member_grade = grade_stats

        samples = member_samples.get(member_grade_id, [])
        percentiles = [
            ov.human_percentile for _, ov in samples if ov.human_percentile is not None
        ]
        z_scores = [ov.z_score for _, ov in samples if ov.z_score is not None]

        member_overrides[member_grade_id] = schemas.HumanMemberComparison(
            grade_id=member_grade_id,
            grade_label=member_grade.label,
            members=list(member_grade.members),
            max_points=member_grade.max_points,
            total_count=member_grade.total_count,
            human_mean=member_grade.mean_estimate,
            human_std=member_grade.stddev_estimate,
            human_best=member_grade.best_score_estimate,
            human_percentile=_average(percentiles),
            z_score=_average(z_scores),
        )

    if not member_overrides:
        member_overrides[grade_stats.id] = schemas.HumanMemberComparison(
            grade_id=grade_stats.id,
            grade_label=grade_stats.label,
            members=list(grade_stats.members),
            max_points=grade_stats.max_points,
            total_count=grade_stats.total_count,
            human_mean=grade_stats.mean_estimate,
            human_std=grade_stats.stddev_estimate,
            human_best=grade_stats.best_score_estimate,
            human_percentile=None,
            z_score=None,
        )

    return member_overrides


def _parse_year(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_grade_id(
    human_year: HumanYear,
    group_value: Optional[str],
) -> Optional[str]:
    group_str = (str(group_value).strip() if group_value else "").replace("\u2013", "-")
    if group_str:
        for grade_id in human_year.grade_ids():
            grade = human_year.grade(grade_id)
            normalized_label = grade.label.replace("/", "-") if grade.label else ""
            if group_str == grade_id:
                return grade_id
            if normalized_label and group_str == normalized_label:
                return grade_id
            if grade.label and group_str == grade.label:
                return grade_id
            members_text = "-".join(str(m) for m in grade.members)
            if group_str == members_text:
                return grade_id
        # attempt relaxed comparison removing whitespace and separators
        normalized = group_str.replace("/", "-")
        for grade_id in human_year.grade_ids():
            grade = human_year.grade(grade_id)
            if normalized == grade_id.replace("/", "-"):
                return grade_id
            if grade.label and normalized == grade.label.replace("/", "-"):
                return grade_id
    # fallback: use single numeric membership
    try:
        value_int = int(group_str)
    except (TypeError, ValueError):
        return None
    for grade_id in human_year.grade_ids():
        grade = human_year.grade(grade_id)
        if value_int in grade.members:
            return grade_id
    return None


def _is_range_group(group_value: Optional[str]) -> bool:
    """Check if group represents a range like '11-13' or '11/13'."""
    if not group_value:
        return False
    group_str = str(group_value).strip().replace("/", "-").replace("\u2013", "-")
    try:
        parts = group_str.split("-")
        if len(parts) == 2:
            int(parts[0])
            int(parts[1])
            return True
    except (ValueError, IndexError):
        pass
    return False


def _resolve_range_group(
    human_year: HumanYear,
    group_value: str,
    strategy: str,
) -> Optional[str]:
    """Resolve a range group like '11-13' by aggregating constituent grades."""
    group_str = str(group_value).strip().replace("/", "-")
    parts = group_str.split("-")
    if len(parts) != 2:
        return None

    try:
        start_grade = int(parts[0])
        end_grade = int(parts[1])
    except ValueError:
        return None

    # Find individual grades in this range
    constituent_grades = []
    for grade_id in human_year.grade_ids():
        grade = human_year.grade(grade_id)
        if len(grade.members) == 1 and start_grade <= grade.members[0] <= end_grade:
            constituent_grades.append(grade_id)

    if not constituent_grades:
        return None

    # Create synthetic grade ID for this aggregation
    synthetic_id = f"_range_aggregated_{group_str}_{strategy}"
    return synthetic_id


def _get_aggregated_grade_stats(
    human_year: HumanYear,
    synthetic_grade_id: str,
    strategy: str,
) -> Optional[HumanGradeStats]:
    """Get aggregated grade stats for a synthetic range-based grade ID."""
    # Parse the synthetic ID to extract the range
    parts = synthetic_grade_id.split("_")
    if len(parts) < 5:
        return None

    range_str = parts[
        3
    ]  # Extract the range part (e.g., "3-4" from "_range_aggregated_3-4_best")
    parts2 = range_str.split("-")
    if len(parts2) != 2:
        return None

    try:
        start_grade = int(parts2[0])
        end_grade = int(parts2[1])
    except ValueError:
        return None

    # Find constituent grades
    constituent_grades = []
    for grade_id in human_year.grade_ids():
        grade = human_year.grade(grade_id)
        if len(grade.members) == 1 and start_grade <= grade.members[0] <= end_grade:
            constituent_grades.append(grade_id)

    if not constituent_grades:
        return None

    # Aggregate the grades based on strategy
    if strategy == "best":
        # Use grade with best human performance (highest mean)
        best_grade_id = None
        best_mean = -1

        for grade_id in constituent_grades:
            grade = human_year.grade(grade_id)
            if grade.mean_estimate and grade.mean_estimate > best_mean:
                best_mean = grade.mean_estimate
                best_grade_id = grade_id

        if best_grade_id:
            base_grade = human_year.grade(best_grade_id)
            # Modify label to reflect aggregation
            aggregated_members = tuple(sorted(range(start_grade, end_grade + 1)))
            return HumanGradeStats(
                id=synthetic_grade_id,
                label=f"{range_str} (best: {base_grade.label})",
                members=aggregated_members,
                max_points=base_grade.max_points,
                counts=base_grade.counts.copy(),
                total_count=base_grade.total_count,
                avg_score_reported=base_grade.avg_score_reported,
                bin_ranges=base_grade.bin_ranges.copy(),
                probabilities=base_grade.probabilities.copy(),
                cdf_points=base_grade.cdf_points.copy(),
                mean_estimate=base_grade.mean_estimate,
                stddev_estimate=base_grade.stddev_estimate,
                best_score_estimate=base_grade.best_score_estimate,
            )

    elif strategy == "average":
        # Aggregate all constituent grades
        aggregated_grade = human_year.aggregate_grades(
            constituent_grades,
            synthetic_grade_id,
            f"{range_str} (avg)",
        )
        if aggregated_grade:
            aggregated_grade.id = synthetic_grade_id
            aggregated_grade.label = f"{range_str} (avg of {len(constituent_grades)})"
            return aggregated_grade

    return None
