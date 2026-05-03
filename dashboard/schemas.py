"""Pydantic schemas shared across the dashboard backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


class BreakdownEntry(BaseModel):
    count: int = 0
    accuracy: float = 0.0
    points_weighted_accuracy: float = 0.0


class RunMetrics(BaseModel):
    answered_count: int = 0
    declined_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    accuracy: float = 0.0
    points_weighted_accuracy: float = 0.0
    total_points_earned: float = 0.0
    mean_latency_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    # Average tokens
    mean_total_tokens: Optional[float] = None
    mean_completion_tokens: Optional[float] = None
    mean_reasoning_tokens: Optional[float] = None
    total_reasoning_tokens: Optional[int] = None
    reasoning_tokens_known_count: Optional[int] = None
    total_cost_usd_known: Optional[float] = None
    unknown_usage_count: Optional[int] = None
    warning_row_count: int = 0
    warning_counts: Dict[str, int] = Field(default_factory=dict)
    breakdown_by_group: Dict[str, BreakdownEntry] = Field(default_factory=dict)
    breakdown_by_year: Dict[str, BreakdownEntry] = Field(default_factory=dict)

    @computed_field(return_type=bool)
    def has_warnings(self) -> bool:
        return bool(self.warning_row_count)


class RunConfigArgs(BaseModel):
    dataset: Optional[str] = None
    model: Optional[str] = None
    reasoning: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    limit: Optional[int] = None
    seed: Optional[int] = None
    concurrency: Optional[int] = None
    output_dir: Optional[str] = None
    fail_fast: Optional[bool] = None


class RunConfigModel(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    supports_vision: Optional[bool] = None
    supports_json_response_format: Optional[bool] = None


class RunConfig(BaseModel):
    timestamp_utc: Optional[str] = None
    args: RunConfigArgs = Field(default_factory=RunConfigArgs)
    model: RunConfigModel = Field(default_factory=RunConfigModel)


class RowRecord(BaseModel):
    id: str
    year: Optional[str]
    group: Optional[str]
    problem_number: Optional[str | int]
    language: Optional[str]
    multimodal: Optional[bool]
    points: Optional[float]
    answer: Optional[str]
    predicted: Optional[str]
    is_correct: Optional[bool]
    points_earned: Optional[float]
    reasoning_mode: Optional[str]
    latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    # Optional usage details captured from some providers (e.g., OpenRouter)
    reasoning_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    audio_prompt_tokens: Optional[int] = None
    cost_usd: Optional[float]
    rationale: Optional[str]
    raw_text_response: Optional[str]
    generation_id: Optional[str]
    error: Optional[str]
    warnings: Optional[List[str]] = None

    # Allow forward compatibility if additional fields are written by the runner
    model_config = {
        "extra": "ignore",
    }


class RunPaths(BaseModel):
    results_json: Optional[str] = None
    results_jsonl: Optional[str] = None
    results_parquet: Optional[str] = None
    metrics_json: str
    config_json: str
    raw_responses_jsonl: Optional[str] = None
    failures_jsonl: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    timestamp: Optional[str]
    model_id: Optional[str]
    model_label: Optional[str]
    dataset_name: Optional[str]
    metrics: RunMetrics
    reasoning_mode: Optional[str] = None
    has_failures: bool = False
    results_source: Optional[str] = None


class RunDetail(RunSummary):
    config: RunConfig
    paths: RunPaths


@dataclass
class RunOverviewFilterParams:
    model: List[str] = field(default_factory=list)
    dataset: List[str] = field(default_factory=list)
    reasoning_mode: List[str] = field(default_factory=list)
    results_source: List[str] = field(default_factory=list)
    has_failures: Optional[bool] = None
    q: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = "timestamp"
    sort_dir: str = "desc"

    def model_dump(self) -> Dict[str, object]:
        """Dictionary representation for templating convenience."""
        return {
            "model": list(self.model),
            "dataset": list(self.dataset),
            "reasoning_mode": list(self.reasoning_mode),
            "results_source": list(self.results_source),
            "has_failures": self.has_failures,
            "q": self.q,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "sort_by": self.sort_by,
            "sort_dir": self.sort_dir,
        }


class FacetOption(BaseModel):
    value: str
    label: str
    total: int = 0
    active: int = 0


class RunOverviewFacets(BaseModel):
    models: List[FacetOption] = Field(default_factory=list)
    datasets: List[FacetOption] = Field(default_factory=list)
    reasoning_modes: List[FacetOption] = Field(default_factory=list)
    results_sources: List[FacetOption] = Field(default_factory=list)
    has_failures: List[FacetOption] = Field(default_factory=list)


class RunListResponse(BaseModel):
    runs: List[RunSummary] = Field(default_factory=list)
    total: int = 0
    total_all: int = 0
    facets: RunOverviewFacets = Field(default_factory=RunOverviewFacets)


class WarningBreakdown(BaseModel):
    warning_type: str
    count: int


class FailureEntry(BaseModel):
    timestamp: Optional[str] = None
    status_code: Optional[int] = None
    message: Optional[str] = None
    id: Optional[str] = None


class DatasetChoice(BaseModel):
    label: Optional[str] = None
    text: Optional[str] = None
    image: Optional[str] = None


class HumanPerformanceMetrics(BaseModel):
    p_correct: Optional[float] = None
    sample_size: Optional[int] = None


class DatasetRow(BaseModel):
    id: str
    problem_statement: Optional[str] = None
    options: List[DatasetChoice] = Field(default_factory=list)
    question_image: Optional[str] = None
    associated_images: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    year: Optional[str] = None
    group: Optional[str] = None
    points: Optional[float] = None
    human_performance: Optional[HumanPerformanceMetrics] = None
    tags: List[str] = Field(default_factory=list)


class RowDetailResponse(BaseModel):
    row: RowRecord
    dataset: Optional[DatasetRow] = None


class ResultFilterParams(BaseModel):
    group: List[str] = Field(default_factory=list)
    year: List[str] = Field(default_factory=list)
    language: List[str] = Field(default_factory=list)
    multimodal: Optional[bool] = None
    correctness: Optional[str] = Field(default=None, pattern="^(true|false|unknown)$")
    predicted: List[str] = Field(default_factory=list)
    reasoning_mode: List[str] = Field(default_factory=list)
    points_min: Optional[float] = None
    points_max: Optional[float] = None
    latency_min: Optional[float] = None
    latency_max: Optional[float] = None
    tokens_min: Optional[int] = None
    tokens_max: Optional[int] = None
    reasoning_tokens_min: Optional[int] = None
    reasoning_tokens_max: Optional[int] = None
    cost_min: Optional[float] = None
    cost_max: Optional[float] = None
    warnings_present: Optional[bool] = None
    warning_types: List[str] = Field(default_factory=list)
    predicted_letter: List[str] = Field(default_factory=list)
    sort_by: str = Field(default="id")
    sort_dir: str = Field(default="asc")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=200)

    model_config = {
        "validate_assignment": True,
        "extra": "ignore",
    }

    @computed_field  # type: ignore[misc]
    def cache_key(self) -> tuple:
        return (
            tuple(sorted(self.group)),
            tuple(sorted(self.year)),
            tuple(sorted(self.language)),
            self.multimodal,
            self.correctness,
            tuple(sorted(self.predicted or self.predicted_letter)),
            tuple(sorted(self.reasoning_mode)),
            self.points_min,
            self.points_max,
            self.latency_min,
            self.latency_max,
            self.tokens_min,
            self.tokens_max,
            self.reasoning_tokens_min,
            self.reasoning_tokens_max,
            self.cost_min,
            self.cost_max,
            self.warnings_present,
            tuple(sorted(self.warning_types)),
        )

    def normalized_predicted(self) -> List[str]:
        if self.predicted:
            return [p.upper() for p in self.predicted]
        return [p.upper() for p in self.predicted_letter]


class ResultsPage(BaseModel):
    items: List[RowRecord]
    total: int
    page: int
    page_size: int


class Histogram(BaseModel):
    bins: List[float] = Field(default_factory=list)
    counts: List[int] = Field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None


class NumericSummary(BaseModel):
    count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    stddev: Optional[float] = None


class DistributionBucket(BaseModel):
    key: str
    label: str
    count: int
    percentage: float


class HumanBinRange(BaseModel):
    min: float
    max: float
    inclusive: str


class HumanBin(BaseModel):
    id: str
    label: str
    range_default: HumanBinRange
    ranges_by_grade: Dict[str, HumanBinRange] = Field(default_factory=dict)


class HumanGradeSummary(BaseModel):
    id: str
    label: str
    members: List[int] = Field(default_factory=list)
    max_points: float
    total_count: int
    avg_score_reported: Optional[float] = None
    mean_estimate: Optional[float] = None
    stddev_estimate: Optional[float] = None
    best_estimate: Optional[float] = None


class HumanYearSummary(BaseModel):
    year: int
    locale: str
    grades: List[HumanGradeSummary] = Field(default_factory=list)
    bins: List[HumanBin] = Field(default_factory=list)
    counts_by_grade: Dict[str, List[int]] = Field(default_factory=dict)
    totals_by_grade: Dict[str, int] = Field(default_factory=dict)
    avg_score_by_grade: Dict[str, Optional[float]] = Field(default_factory=dict)
    ui_groups: Dict[str, List[int]] = Field(default_factory=dict)
    source: Optional[Dict[str, object]] = None


class HumanPercentileResponse(BaseModel):
    year: int
    grade_id: str
    score: float
    max_points: float
    percentile: Optional[float] = None
    z_score: Optional[float] = None
    mean_estimate: Optional[float] = None
    stddev_estimate: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class HumanYearListEntry(BaseModel):
    year: int
    locale: str
    grades: List[HumanGradeSummary] = Field(default_factory=list)
    ui_groups: Dict[str, List[int]] = Field(default_factory=dict)


class HumanCDFPoint(BaseModel):
    score: float
    percentile: float


class HumanCDFResponse(BaseModel):
    year: int
    grade_id: str
    max_points: float
    mean_estimate: Optional[float] = None
    stddev_estimate: Optional[float] = None
    points: List[HumanCDFPoint] = Field(default_factory=list)


class HumanBinComparison(BaseModel):
    bin_id: str
    human_share: float
    llm_share: float
    delta: float
    llm_share_smoothed: float
    delta_smoothed: float


class HumanMemberComparison(BaseModel):
    grade_id: str
    grade_label: str
    members: List[int] = Field(default_factory=list)
    max_points: Optional[float] = None
    total_count: Optional[int] = None
    human_mean: Optional[float] = None
    human_std: Optional[float] = None
    human_best: Optional[float] = None
    human_percentile: Optional[float] = None
    z_score: Optional[float] = None


class HumanRunGradeComparison(BaseModel):
    year: int
    grade_id: str
    grade_label: str
    members: List[int] = Field(default_factory=list)
    max_points: float
    llm_total: float
    llm_max: float
    llm_score_pct: Optional[float] = None
    llm_start_points: Optional[float] = None
    llm_points_awarded: Optional[float] = None
    llm_points_available: Optional[float] = None
    human_percentile: Optional[float] = None
    z_score: Optional[float] = None
    human_mean: Optional[float] = None
    human_std: Optional[float] = None
    human_best: Optional[float] = None
    bin_comparison: List[HumanBinComparison] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    member_overrides: Dict[str, HumanMemberComparison] = Field(default_factory=dict)


class HumanRunComparisonResponse(BaseModel):
    run_id: str
    entries: List[HumanRunGradeComparison] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class HumanAggregateEntry(BaseModel):
    year: int
    grade_id: str
    grade_label: str
    members: List[int] = Field(default_factory=list)
    run_count: int
    sample_count: int
    avg_llm_score_pct: Optional[float] = None
    avg_human_mean_pct: Optional[float] = None
    avg_human_best_pct: Optional[float] = None
    avg_human_percentile: Optional[float] = None
    median_percentile: Optional[float] = None
    p25_percentile: Optional[float] = None
    p75_percentile: Optional[float] = None
    min_percentile: Optional[float] = None
    max_percentile: Optional[float] = None
    avg_z_score: Optional[float] = None
    best_run_id: Optional[str] = None
    worst_run_id: Optional[str] = None
    bin_comparison: List[HumanBinComparison] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    member_overrides: Dict[str, HumanMemberComparison] = Field(default_factory=dict)


class HumanAggregateStats(BaseModel):
    cohort_type: Literal["micro", "macro"]
    entries: List[HumanAggregateEntry] = Field(default_factory=list)


class HumanCohortComparisonResponse(BaseModel):
    run_ids: List[str] = Field(default_factory=list)
    micro: HumanAggregateStats
    macro: HumanAggregateStats
    notes: List[str] = Field(default_factory=list)


class HumanComparisonTopCell(BaseModel):
    year: int
    grade_id: str
    grade_label: str
    members: List[int] = Field(default_factory=list)
    llm_score_pct: Optional[float] = None
    human_score_pct: Optional[float] = None
    gap_score_pct: Optional[float] = None
    gap_points: Optional[float] = None
    human_percentile: Optional[float] = None
    z_score: Optional[float] = None
    llm_total: Optional[float] = None
    human_score: Optional[float] = None


class HumanComparisonBestYear(BaseModel):
    year: int
    avg_percentile: Optional[float] = None
    avg_gap_pct: Optional[float] = None
    avg_score_pct: Optional[float] = None


class HumanComparisonBestGrade(BaseModel):
    grade_id: str
    grade_label: str
    avg_percentile: Optional[float] = None
    avg_gap_pct: Optional[float] = None
    avg_score_pct: Optional[float] = None


class HumanComparisonThresholdBreakdown(BaseModel):
    threshold_pp: float
    llm_win_count: int = 0
    human_win_count: int = 0


class HumanComparisonSummary(BaseModel):
    run_ids: List[str] = Field(default_factory=list)
    comparator: Literal["average", "best"] = "average"
    weight_mode: Literal["micro", "macro"] = "micro"
    total_cells: int = 0
    llm_win_count: int = 0
    human_win_count: int = 0
    tie_count: int = 0
    avg_percentile: Optional[float] = None
    avg_z_score: Optional[float] = None
    avg_gap_pct: Optional[float] = None
    top_llm_wins: List[HumanComparisonTopCell] = Field(default_factory=list)
    top_human_wins: List[HumanComparisonTopCell] = Field(default_factory=list)
    best_year: Optional[HumanComparisonBestYear] = None
    best_grade: Optional[HumanComparisonBestGrade] = None
    threshold_breakdown: List[HumanComparisonThresholdBreakdown] = Field(
        default_factory=list
    )


class HumanComparisonSummaryResponse(BaseModel):
    summary: HumanComparisonSummary


class HumanBaselineSummary(BaseModel):
    comparator: Literal["average", "best"] = "average"
    best_year: Optional[HumanComparisonBestYear] = None
    best_grade: Optional[HumanComparisonBestGrade] = None
    notes: List[str] = Field(default_factory=list)


class HumanCohortSummaryRequest(BaseModel):
    run_ids: List[str] = Field(default_factory=list)
    comparator: Literal["average", "best"] = "average"
    weight_mode: Literal["micro", "macro"] = "micro"
    late_year_strategy: str = "best"
    top_limit: int = Field(default=5, ge=1, le=50)
    thresholds: List[float] = Field(default_factory=list)


class SubsetMetrics(BaseModel):
    key: str
    label: str
    count: int = 0
    share: float = 0.0
    accuracy: Optional[float] = None
    multimodal_share: Optional[float] = None
    points_summary: NumericSummary = Field(default_factory=NumericSummary)
    points_earned_summary: NumericSummary = Field(default_factory=NumericSummary)
    latency_summary: NumericSummary = Field(default_factory=NumericSummary)
    tokens_summary: NumericSummary = Field(default_factory=NumericSummary)
    reasoning_tokens_summary: NumericSummary = Field(default_factory=NumericSummary)
    cost_summary: NumericSummary = Field(default_factory=NumericSummary)
    grade_distribution: List[DistributionBucket] = Field(default_factory=list)
    year_distribution: List[DistributionBucket] = Field(default_factory=list)
    language_distribution: List[DistributionBucket] = Field(default_factory=list)
    reasoning_mode_distribution: List[DistributionBucket] = Field(default_factory=list)
    points_hist: Histogram = Field(default_factory=Histogram)
    points_earned_hist: Histogram = Field(default_factory=Histogram)


class AggregatesResponse(BaseModel):
    subset_metrics: List[SubsetMetrics] = Field(default_factory=list)
    breakdown_by_group: Dict[str, BreakdownEntry] = Field(default_factory=dict)
    breakdown_by_year: Dict[str, BreakdownEntry] = Field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    latency_hist: Histogram = Field(default_factory=Histogram)
    tokens_hist: Histogram = Field(default_factory=Histogram)
    reasoning_tokens_hist: Histogram = Field(default_factory=Histogram)
    predicted_counts: Dict[str, int] = Field(default_factory=dict)
    warning_toplist: List[WarningBreakdown] = Field(default_factory=list)


class FilterFacets(BaseModel):
    groups: List[str] = Field(default_factory=list)
    years: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    reasoning_modes: List[str] = Field(default_factory=list)
    predicted_letters: List[str] = Field(default_factory=list)
    warning_types: List[str] = Field(default_factory=list)


class CompareMetricDelta(BaseModel):
    metric: str
    run_a: Optional[float] = None
    run_b: Optional[float] = None
    delta: Optional[float] = None


class CompareRowDelta(BaseModel):
    id: str
    run_a_correct: Optional[bool]
    run_b_correct: Optional[bool]
    run_a_predicted: Optional[str]
    run_b_predicted: Optional[str]
    run_a_points: Optional[float]
    run_b_points: Optional[float]
    delta_points: Optional[float]
    delta_latency_ms: Optional[float]
    delta_total_tokens: Optional[float]


class CompareResponse(BaseModel):
    run_a: RunSummary
    run_b: RunSummary
    metrics: List[CompareMetricDelta]
    breakdown_deltas: Dict[str, List[CompareMetricDelta]]
    row_deltas: List[CompareRowDelta]
    confusion_matrices: Dict[str, Dict[str, Dict[str, int]]] = Field(
        default_factory=dict
    )


class ExportResponse(BaseModel):
    filename: str
    content_type: str
    content: bytes
