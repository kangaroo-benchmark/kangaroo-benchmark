"""Human baseline loader and distribution utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator

from . import schemas
from .utils import load_json


VALID_INCLUSIVE = {"left", "right", "both"}


class RangeSpec(BaseModel):
    """Numeric range definition for a histogram bin."""

    min: float
    max: float
    inclusive: str = Field(pattern="^(left|right|both)$")

    @model_validator(mode="after")
    def _validate_bounds(cls, values):  # type: ignore[override]
        if values.min > values.max:
            raise ValueError("range min cannot exceed max")
        return values


class BinSpec(BaseModel):
    id: str
    label_pdf: str
    range_default: RangeSpec
    ranges_by_grade: Dict[str, RangeSpec] = Field(default_factory=dict)


class GradeSpec(BaseModel):
    id: str
    label: str
    members: List[int]
    max_points: float
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _validate_members(cls, values):  # type: ignore[override]
        if not values.members:
            raise ValueError("grade.members must not be empty")
        return values


class RawHumanBaseline(BaseModel):
    schema_version: str
    year: int
    locale: str
    grades: List[GradeSpec]
    bins: List[BinSpec]
    counts_by_grade: Dict[str, List[int]]
    totals_by_grade: Dict[str, int]
    avg_score_by_grade: Dict[str, float]
    ui_groups: Dict[str, List[int]] = Field(default_factory=dict)
    source: Optional[Dict[str, object]] = None

    model_config = {
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def _validate_schema(cls, values):  # type: ignore[override]
        if values.schema_version != "2.0":
            raise ValueError("Unsupported schema_version; expected '2.0'")
        if not values.bins:
            raise ValueError("At least one bin is required")
        grade_ids = {grade.id for grade in values.grades}
        for grade in values.grades:
            if grade.id not in values.counts_by_grade:
                raise ValueError(f"Missing counts for grade '{grade.id}'")
            counts = values.counts_by_grade[grade.id]
            if len(counts) != len(values.bins):
                raise ValueError(
                    f"Counts for grade '{grade.id}' must match bin count ({len(values.bins)})"
                )
            if grade.id not in values.totals_by_grade:
                raise ValueError(f"Missing totals for grade '{grade.id}'")
            if sum(counts) != values.totals_by_grade[grade.id]:
                raise ValueError(
                    f"Counts for grade '{grade.id}' must sum to totals_by_grade"
                )
        extra_ids = set(values.counts_by_grade.keys()) - grade_ids
        if extra_ids:
            raise ValueError(
                "counts_by_grade contains grades not defined in 'grades': "
                + ", ".join(sorted(extra_ids))
            )
        return values


@dataclass(frozen=True)
class HumanBinRange:
    min: float
    max: float
    inclusive: str

    def contains(self, score: float) -> bool:
        if self.inclusive not in VALID_INCLUSIVE:
            raise ValueError(f"Unknown inclusive mode: {self.inclusive}")
        lo = self.min
        hi = self.max
        if self.inclusive == "both":
            return lo <= score <= hi
        if self.inclusive == "left":
            return lo <= score < hi
        # inclusive == "right"
        return lo < score <= hi

    @property
    def width(self) -> float:
        return max(self.max - self.min, 0.0)

    @property
    def midpoint(self) -> float:
        if self.width == 0:
            return self.max
        return (self.min + self.max) / 2.0


@dataclass(frozen=True)
class HumanBin:
    id: str
    label: str
    default_range: HumanBinRange
    ranges_by_grade: Dict[str, HumanBinRange] = field(default_factory=dict)

    def range_for_grade(self, grade_id: str) -> HumanBinRange:
        return self.ranges_by_grade.get(grade_id, self.default_range)


@dataclass
class HumanGradeStats:
    id: str
    label: str
    members: Tuple[int, ...]
    max_points: float
    counts: List[int]
    total_count: int
    avg_score_reported: Optional[float]
    bin_ranges: List[HumanBinRange]
    probabilities: List[float]
    cdf_points: List[Tuple[float, float]]
    mean_estimate: Optional[float]
    stddev_estimate: Optional[float]
    best_score_estimate: Optional[float]

    def percentile(self, score: float) -> Optional[float]:
        """Approximate percentile for ``score`` on the grade's own scale."""
        if self.total_count <= 0:
            return None
        clamped = max(0.0, min(score, self.max_points))
        cumulative = 0.0
        for rng, count in self._ascending_bins():
            lower, upper = rng.min, rng.max
            if count <= 0:
                continue
            if clamped > upper:
                cumulative += count
                continue
            if rng.width == 0:
                fraction = 1.0 if clamped >= upper else 0.0
            else:
                fraction = (clamped - lower) / rng.width
            fraction = max(0.0, min(1.0, fraction))
            percentile = (cumulative + count * fraction) / self.total_count
            return max(0.0, min(1.0, percentile))
        percentile = cumulative / self.total_count
        return max(0.0, min(1.0, percentile))

    def z_score(self, score: float) -> Optional[float]:
        if self.mean_estimate is None or self.stddev_estimate is None:
            return None
        if self.stddev_estimate < 1e-9:
            return None
        return (score - self.mean_estimate) / self.stddev_estimate

    def bin_index_for_score(self, score: float) -> Optional[int]:
        clamped = max(0.0, min(score, self.max_points))
        for idx, rng in enumerate(self.bin_ranges):
            if rng.contains(clamped):
                return idx
        # fallback: return closest bin
        distances = [self._distance_to_range(rng, clamped) for rng in self.bin_ranges]
        return distances.index(min(distances)) if distances else None

    def share_by_bin(self) -> List[float]:
        if self.total_count <= 0:
            return [0.0 for _ in self.counts]
        return [count / self.total_count for count in self.counts]

    def _ascending_bins(self) -> List[Tuple[HumanBinRange, int]]:
        return list(reversed(list(zip(self.bin_ranges, self.counts))))

    @staticmethod
    def _distance_to_range(rng: HumanBinRange, score: float) -> float:
        if rng.contains(score):
            return 0.0
        if score < rng.min:
            return rng.min - score
        return score - rng.max


@dataclass
class HumanYear:
    year: int
    locale: str
    bins: List[HumanBin]
    grades: Dict[str, HumanGradeStats]
    grade_order: List[str]
    ui_groups: Dict[str, List[int]]
    source: Optional[Dict[str, object]] = None

    def grade(self, grade_id: str) -> HumanGradeStats:
        return self.grades[grade_id]

    def list_grades(self) -> List[HumanGradeStats]:
        return [self.grades[g] for g in self.grade_order]

    def grade_ids(self) -> List[str]:
        return list(self.grade_order)

    def find_grade_ids_for_members(self, members: Iterable[int]) -> List[str]:
        target = set(members)
        selected: List[str] = []
        covered: set[int] = set()
        for grade_id in self.grade_order:
            grade = self.grades[grade_id]
            grade_members = set(grade.members)
            if grade_members.issubset(target):
                selected.append(grade_id)
                covered.update(grade_members)
        if covered != target:
            return []
        return selected

    def aggregate_grades(
        self, grade_ids: Sequence[str], aggregate_id: str, label: str
    ) -> Optional[HumanGradeStats]:
        if not grade_ids:
            return None
        reference = self.grades[grade_ids[0]]
        total_counts = [0 for _ in reference.counts]
        total_total = 0
        members: set[int] = set()
        for gid in grade_ids:
            grade = self.grades[gid]
            total_counts = [a + b for a, b in zip(total_counts, grade.counts)]
            total_total += grade.total_count
            members.update(grade.members)
        avg_reported = None
        if total_total > 0:
            weighted_sum = 0.0
            for gid in grade_ids:
                grade = self.grades[gid]
                if grade.avg_score_reported is not None:
                    weighted_sum += grade.avg_score_reported * grade.total_count
            if weighted_sum:
                avg_reported = weighted_sum / total_total
        stats = _build_grade_stats(
            aggregate_id,
            label,
            tuple(sorted(members)),
            reference.max_points,
            reference.bin_ranges,
            total_counts,
            total_total,
            avg_reported,
        )
        return stats


class HumanBaselineIndex:
    """Loads and exposes human baseline distributions."""

    def __init__(
        self,
        directory: Path | str = Path("human_results"),
        *,
        strict: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.strict = strict
        self._years: Dict[int, HumanYear] = {}
        self._errors: List[Tuple[Path, Exception]] = []
        self.reload()

    def reload(self) -> None:
        self._years.clear()
        self._errors.clear()
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("human_baseline_*.json")):
            try:
                raw = load_json(path)
                baseline = RawHumanBaseline.model_validate(raw)
            except (ValidationError, ValueError) as exc:
                if self.strict:
                    raise ValueError(f"Failed to load {path}: {exc}") from exc
                self._errors.append((path, exc))
                print(f"[human_baseline] warning: skipped {path}: {exc}")
                continue
            human_year = _build_human_year(baseline)
            self._years[human_year.year] = human_year

    def available_years(self) -> List[int]:
        return sorted(self._years.keys())

    def has_data(self) -> bool:
        return bool(self._years)

    def get_year(self, year: int) -> HumanYear:
        if year not in self._years:
            raise KeyError(f"No human baseline for year {year}")
        return self._years[year]

    @property
    def errors(self) -> List[Tuple[Path, Exception]]:
        return list(self._errors)

    def percentile(self, year: int, grade_id: str, score: float) -> Optional[float]:
        try:
            return self.get_year(year).grade(grade_id).percentile(score)
        except KeyError:
            return None

    def z_score(self, year: int, grade_id: str, score: float) -> Optional[float]:
        try:
            return self.get_year(year).grade(grade_id).z_score(score)
        except KeyError:
            return None

    def cdf_points(self, year: int, grade_id: str) -> List[Tuple[float, float]]:
        return list(self.get_year(year).grade(grade_id).cdf_points)

    def year_summary(self, year: int) -> schemas.HumanYearSummary:
        return _human_year_to_schema(self.get_year(year))

    def year_list(self) -> List[schemas.HumanYearListEntry]:
        entries: List[schemas.HumanYearListEntry] = []
        for year in self.available_years():
            human_year = self.get_year(year)
            entries.append(
                schemas.HumanYearListEntry(
                    year=human_year.year,
                    locale=human_year.locale,
                    grades=[
                        _grade_to_schema(grade) for grade in human_year.list_grades()
                    ],
                    ui_groups=human_year.ui_groups,
                )
            )
        return entries

    def percentile_response(
        self, year: int, grade_id: str, score: float
    ) -> Optional[schemas.HumanPercentileResponse]:
        try:
            grade = self.get_year(year).grade(grade_id)
        except KeyError:
            return None
        percentile = grade.percentile(score)
        z = grade.z_score(score)
        return schemas.HumanPercentileResponse(
            year=year,
            grade_id=grade_id,
            score=score,
            max_points=grade.max_points,
            percentile=percentile,
            z_score=z,
            mean_estimate=grade.mean_estimate,
            stddev_estimate=grade.stddev_estimate,
            notes=[],
        )

    def cdf_response(
        self, year: int, grade_id: str
    ) -> Optional[schemas.HumanCDFResponse]:
        try:
            grade = self.get_year(year).grade(grade_id)
        except KeyError:
            return None
        points = [
            schemas.HumanCDFPoint(score=score, percentile=percentile)
            for score, percentile in grade.cdf_points
        ]
        return schemas.HumanCDFResponse(
            year=year,
            grade_id=grade_id,
            max_points=grade.max_points,
            mean_estimate=grade.mean_estimate,
            stddev_estimate=grade.stddev_estimate,
            points=points,
        )


def _build_human_year(raw: RawHumanBaseline) -> HumanYear:
    bins = [
        HumanBin(
            id=bin_spec.id,
            label=bin_spec.label_pdf,
            default_range=_range_to_dataclass(bin_spec.range_default),
            ranges_by_grade={
                gid: _range_to_dataclass(rng)
                for gid, rng in bin_spec.ranges_by_grade.items()
            },
        )
        for bin_spec in raw.bins
    ]

    grades: Dict[str, HumanGradeStats] = {}
    grade_order: List[str] = []
    for grade_spec in raw.grades:
        grade_order.append(grade_spec.id)
        counts = raw.counts_by_grade[grade_spec.id]
        total = raw.totals_by_grade[grade_spec.id]
        avg_reported = raw.avg_score_by_grade.get(grade_spec.id)
        bin_ranges = [bin.range_for_grade(grade_spec.id) for bin in bins]
        stats = _build_grade_stats(
            grade_spec.id,
            grade_spec.label,
            tuple(sorted(grade_spec.members)),
            float(grade_spec.max_points),
            bin_ranges,
            counts,
            total,
            avg_reported,
        )
        grades[grade_spec.id] = stats

    return HumanYear(
        year=raw.year,
        locale=raw.locale,
        bins=bins,
        grades=grades,
        grade_order=grade_order,
        ui_groups=raw.ui_groups,
        source=raw.source,
    )


def _build_grade_stats(
    grade_id: str,
    label: str,
    members: Tuple[int, ...],
    max_points: float,
    bin_ranges: Sequence[HumanBinRange],
    counts: Sequence[int],
    total: int,
    avg_reported: Optional[float],
) -> HumanGradeStats:
    counts_list = [int(c) for c in counts]
    total_count = int(total)
    probabilities: List[float]
    if total_count > 0:
        probabilities = [c / total_count for c in counts_list]
    else:
        probabilities = [0.0 for _ in counts_list]

    descending_bins = list(zip(bin_ranges, counts_list))
    ascending_bins = list(reversed(descending_bins))
    cdf_points: List[Tuple[float, float]] = []
    if total_count > 0:
        cumulative = 0
        for rng, count in ascending_bins:
            cumulative += count
            cdf_points.append((rng.max, cumulative / total_count))
        cdf_points.sort(key=lambda item: item[0])

    mean_estimate: Optional[float] = None
    stddev_estimate: Optional[float] = None
    if total_count > 0:
        midpoints = [rng.midpoint for rng in bin_ranges]
        mean = sum(p * m for p, m in zip(probabilities, midpoints))
        variance = sum(p * (m - mean) ** 2 for p, m in zip(probabilities, midpoints))
        mean_estimate = mean
        stddev_estimate = math.sqrt(max(variance, 0.0))

    best_score_estimate: Optional[float] = None
    if total_count > 0:
        for rng, count in descending_bins:
            if count > 0:
                best_score_estimate = rng.max
                break

    return HumanGradeStats(
        id=grade_id,
        label=label,
        members=members,
        max_points=float(max_points),
        counts=counts_list,
        total_count=total_count,
        avg_score_reported=avg_reported,
        bin_ranges=list(bin_ranges),
        probabilities=probabilities,
        cdf_points=cdf_points,
        mean_estimate=mean_estimate,
        stddev_estimate=stddev_estimate,
        best_score_estimate=best_score_estimate,
    )


def _range_to_dataclass(rng: RangeSpec) -> HumanBinRange:
    return HumanBinRange(
        min=float(rng.min), max=float(rng.max), inclusive=str(rng.inclusive)
    )


def _range_to_schema(range_obj: HumanBinRange) -> schemas.HumanBinRange:
    return schemas.HumanBinRange(
        min=range_obj.min, max=range_obj.max, inclusive=range_obj.inclusive
    )


def _bin_to_schema(bin_obj: HumanBin) -> schemas.HumanBin:
    return schemas.HumanBin(
        id=bin_obj.id,
        label=bin_obj.label,
        range_default=_range_to_schema(bin_obj.default_range),
        ranges_by_grade={
            gid: _range_to_schema(rng) for gid, rng in bin_obj.ranges_by_grade.items()
        },
    )


def _grade_to_schema(grade: HumanGradeStats) -> schemas.HumanGradeSummary:
    return schemas.HumanGradeSummary(
        id=grade.id,
        label=grade.label,
        members=list(grade.members),
        max_points=grade.max_points,
        total_count=grade.total_count,
        avg_score_reported=grade.avg_score_reported,
        mean_estimate=grade.mean_estimate,
        stddev_estimate=grade.stddev_estimate,
        best_estimate=grade.best_score_estimate,
    )


def _human_year_to_schema(human_year: HumanYear) -> schemas.HumanYearSummary:
    return schemas.HumanYearSummary(
        year=human_year.year,
        locale=human_year.locale,
        grades=[_grade_to_schema(grade) for grade in human_year.list_grades()],
        bins=[_bin_to_schema(bin_obj) for bin_obj in human_year.bins],
        counts_by_grade={
            gid: list(human_year.grades[gid].counts) for gid in human_year.grade_ids()
        },
        totals_by_grade={
            gid: human_year.grades[gid].total_count for gid in human_year.grade_ids()
        },
        avg_score_by_grade={
            gid: human_year.grades[gid].avg_score_reported
            for gid in human_year.grade_ids()
        },
        ui_groups=human_year.ui_groups,
        source=human_year.source,
    )
