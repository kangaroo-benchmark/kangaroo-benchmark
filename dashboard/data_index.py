"""Data indexing and results loading utilities for the dashboard."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Optional, Set, Tuple

import cachetools
import pyarrow.parquet as pq

from score_utils import coerce_points, start_points_for_group

from . import aggregations, schemas
from .utils import (
    grade_group_sort_key,
    load_json,
    load_jsonl,
    load_models_registry,
    parse_run_timestamp,
    slug_to_label,
)


class RunNotFoundError(Exception):
    pass


class RowNotFoundError(Exception):
    pass


@dataclass
class RunFilePaths:
    base_dir: Path
    metrics: Path
    config: Path
    results_json: Optional[Path]
    results_jsonl: Optional[Path]
    results_parquet: Optional[Path]
    raw_responses: Optional[Path]
    failures: Optional[Path]


@dataclass
class RunRecord:
    run_id: str
    paths: RunFilePaths
    metrics: schemas.RunMetrics
    config: schemas.RunConfig
    model_id: Optional[str]
    model_label: Optional[str]
    dataset_name: Optional[str]
    reasoning_mode: Optional[str]
    has_failures: bool
    results_source: Optional[str]
    timestamp: Optional[str]
    timestamp_dt: Optional[datetime]

    def to_summary(self) -> schemas.RunSummary:
        return schemas.RunSummary(
            run_id=self.run_id,
            timestamp=self.timestamp,
            model_id=self.model_id,
            model_label=self.model_label,
            dataset_name=self.dataset_name,
            metrics=self.metrics,
            reasoning_mode=self.reasoning_mode,
            has_failures=self.has_failures,
            results_source=self.results_source,
        )

    def to_detail(self) -> schemas.RunDetail:
        return schemas.RunDetail(
            **self.to_summary().model_dump(),
            config=self.config,
            paths=schemas.RunPaths(
                results_json=str(self.paths.results_json)
                if self.paths.results_json
                else None,
                results_jsonl=str(self.paths.results_jsonl)
                if self.paths.results_jsonl
                else None,
                results_parquet=str(self.paths.results_parquet)
                if self.paths.results_parquet
                else None,
                metrics_json=str(self.paths.metrics),
                config_json=str(self.paths.config),
                raw_responses_jsonl=str(self.paths.raw_responses)
                if self.paths.raw_responses
                else None,
                failures_jsonl=str(self.paths.failures)
                if self.paths.failures
                else None,
            ),
        )


class RunIndex:
    """Indexes run folders and exposes helpers for loading data."""

    def __init__(self, runs_dir: Path | str, models_path: Path | str) -> None:
        self.runs_dir = Path(runs_dir)
        self.models_path = Path(models_path)
        self._runs: Dict[str, RunRecord] = {}
        self._id_to_runs: DefaultDict[str, List[str]] = defaultdict(list)
        self._aggregate_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=64)
        self._models_registry: Dict[str, Dict[str, object]] = {}
        self.reload()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def reload(self) -> None:
        self._runs.clear()
        self._id_to_runs.clear()
        self._aggregate_cache.clear()
        self._models_registry = load_models_registry(self.models_path)
        if not self.runs_dir.exists():
            return

        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            record = self._load_run_record(run_id, run_dir)
            if record:
                self._runs[run_id] = record

        self._build_inverted_index()

    def _load_run_record(self, run_id: str, run_dir: Path) -> Optional[RunRecord]:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            return None

        metrics_raw = load_json(metrics_path)
        config_raw = load_json(config_path)
        metrics = schemas.RunMetrics.model_validate(metrics_raw)
        config = schemas.RunConfig.model_validate(config_raw)

        dataset_name: Optional[str] = None
        if config.args.dataset:
            dataset_name = Path(config.args.dataset).name

        model_id = config.model.id or (config.args.model if config.args else None)
        model_label = (
            slug_to_label(model_id, self._models_registry) if model_id else None
        )
        if not model_label and config.model.label:
            model_label = config.model.label
        if model_id and not config.model.id:
            config.model.id = model_id

        files = RunFilePaths(
            base_dir=run_dir,
            metrics=metrics_path,
            config=config_path,
            results_json=self._optional_file(run_dir, "results.json"),
            results_jsonl=self._optional_file(run_dir, "results.jsonl"),
            results_parquet=self._optional_file(run_dir, "results.parquet"),
            raw_responses=self._optional_file(run_dir, "raw_responses.jsonl"),
            failures=self._optional_file(run_dir, "failures.jsonl"),
        )

        results_source = None
        # Prefer parquet (usually the complete results), then jsonl, then json
        if files.results_parquet is not None:
            results_source = "parquet"
        elif files.results_jsonl is not None:
            results_source = "jsonl"
        elif files.results_json is not None:
            results_source = "json"

        has_failures = (
            files.failures is not None
            and files.failures.exists()
            and files.failures.stat().st_size > 0
        )
        ts = parse_run_timestamp(run_id)

        # Create a temporary record to pass to the calculation method
        temp_record = RunRecord(
            run_id=run_id,
            paths=files,
            metrics=metrics,
            config=config,
            model_id=model_id,
            model_label=model_label,
            dataset_name=dataset_name,
            reasoning_mode=config.args.reasoning if config.args else None,
            has_failures=has_failures,
            results_source=results_source,
            timestamp=ts.strftime("%Y-%m-%d %H:%M:%S") if ts else None,
            timestamp_dt=ts,
        )

        # Calculate total_points_earned for legacy runs if not present or still default value
        # If metrics.total_points_earned is 0 and other metrics exist, check if we need to calculate
        if metrics.total_points_earned == 0.0 and metrics.answered_count > 0:
            total_points = self._calculate_total_points_earned_direct(files)
            temp_record.metrics.total_points_earned = total_points

        # Backfill mean_completion_tokens for legacy runs if missing but results have completion_tokens
        if (
            getattr(metrics, "mean_completion_tokens", None) is None
            and metrics.answered_count > 0
        ):
            backfilled_mean = self._calculate_mean_completion_tokens_direct(files)
            if backfilled_mean is not None:
                temp_record.metrics.mean_completion_tokens = backfilled_mean

        return temp_record

    def _optional_file(self, directory: Path, name: str) -> Optional[Path]:
        candidate = directory / name
        return candidate if candidate.exists() else None

    def _calculate_mean_completion_tokens_direct(
        self, paths: RunFilePaths
    ) -> Optional[float]:
        """Compute mean of completion_tokens across answered rows directly from results files.

        Returns None if not available.
        """
        results_path = (
            paths.results_jsonl or paths.results_json or paths.results_parquet
        )
        if not results_path:
            return None
        total = 0
        count = 0
        try:
            if paths.results_jsonl:
                for obj in load_jsonl(paths.results_jsonl):
                    ct = obj.get("completion_tokens")
                    pred = obj.get("predicted")
                    err = obj.get("error")
                    if ct is not None and pred is not None and not err:
                        try:
                            total += int(ct)
                            count += 1
                        except Exception:
                            continue
            elif paths.results_json:
                data = load_json(paths.results_json)
                for obj in data:
                    ct = obj.get("completion_tokens")
                    pred = obj.get("predicted")
                    err = obj.get("error")
                    if ct is not None and pred is not None and not err:
                        try:
                            total += int(ct)
                            count += 1
                        except Exception:
                            continue
            elif paths.results_parquet:
                parquet_file = pq.ParquetFile(paths.results_parquet)
                for batch in parquet_file.iter_batches():
                    for obj in batch.to_pylist():
                        ct = obj.get("completion_tokens")
                        pred = obj.get("predicted")
                        err = obj.get("error")
                        if ct is not None and pred is not None and not err:
                            try:
                                total += int(ct)
                                count += 1
                            except Exception:
                                continue
        except FileNotFoundError:
            return None
        if count == 0:
            return None
        return float(total) / float(count)

    def _build_inverted_index(self) -> None:
        for run_id in self._runs:
            try:
                for row in self.iter_results(run_id):
                    self._id_to_runs[row.id].append(run_id)
            except FileNotFoundError:
                continue

    def total_runs(self) -> int:
        return len(self._runs)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def list_runs(
        self, filters: Optional[schemas.RunOverviewFilterParams] = None
    ) -> List[schemas.RunSummary]:
        records = list(self._runs.values())
        if filters:
            records = [
                record
                for record in records
                if self._run_matches_overview_filters(record, filters)
            ]

        key_lookup = {
            "timestamp": lambda r: r.timestamp_dt or datetime.min,
            "accuracy": lambda r: r.metrics.accuracy or 0.0,
            "answered": lambda r: r.metrics.answered_count,
            "failed": lambda r: r.metrics.failed_count,
        }
        sort_key = key_lookup.get(
            filters.sort_by if filters else "timestamp", key_lookup["timestamp"]
        )
        reverse = (filters.sort_dir if filters else "desc").lower() == "desc"

        records.sort(key=sort_key, reverse=reverse)
        return [record.to_summary() for record in records]

    def _run_matches_overview_filters(
        self,
        record: RunRecord,
        filters: schemas.RunOverviewFilterParams,
    ) -> bool:
        if filters.model and (record.model_id not in filters.model):
            return False
        if filters.dataset and (record.dataset_name not in filters.dataset):
            return False
        if filters.reasoning_mode and (
            record.reasoning_mode not in filters.reasoning_mode
        ):
            return False
        if filters.results_source and (
            record.results_source not in filters.results_source
        ):
            return False
        if (
            filters.has_failures is not None
            and record.has_failures != filters.has_failures
        ):
            return False
        if filters.q:
            needle = filters.q.lower()
            haystacks = filter(
                None,
                [
                    record.run_id,
                    record.model_label,
                    record.model_id,
                    record.dataset_name,
                ],
            )
            if not any(needle in value.lower() for value in haystacks):
                return False
        if filters.date_from:
            if not record.timestamp_dt or record.timestamp_dt < filters.date_from:
                return False
        if filters.date_to:
            if not record.timestamp_dt:
                return False
            upper = filters.date_to
            if upper.time() == datetime.min.time():
                upper = upper + timedelta(days=1) - timedelta(microseconds=1)
            if record.timestamp_dt > upper:
                return False
        return True

    def get_run_overview_facets(
        self,
        filters: Optional[schemas.RunOverviewFilterParams] = None,
    ) -> schemas.RunOverviewFacets:
        records_all = list(self._runs.values())
        if not records_all:
            return schemas.RunOverviewFacets()

        records_active = records_all
        if filters:
            records_active = [
                r for r in records_all if self._run_matches_overview_filters(r, filters)
            ]

        def build_options(
            total_counter: Counter[str],
            active_counter: Counter[str],
            label_lookup: Dict[str, str],
        ) -> List[schemas.FacetOption]:
            options: List[schemas.FacetOption] = []
            for value in sorted(set(total_counter.keys()) | set(active_counter.keys())):
                options.append(
                    schemas.FacetOption(
                        value=value,
                        label=label_lookup.get(
                            value, value.title() if value else "Unknown"
                        ),
                        total=total_counter.get(value, 0),
                        active=active_counter.get(value, 0),
                    )
                )
            return options

        def label_for_model(model_id: str) -> str:
            for record in records_all:
                if record.model_id == model_id:
                    return record.model_label or model_id
            return model_id

        model_total = Counter(r.model_id for r in records_all if r.model_id)
        model_active = Counter(r.model_id for r in records_active if r.model_id)
        model_labels = {mid: label_for_model(mid) for mid in model_total}

        dataset_total = Counter(r.dataset_name for r in records_all if r.dataset_name)
        dataset_active = Counter(
            r.dataset_name for r in records_active if r.dataset_name
        )
        dataset_labels = {name: name for name in dataset_total}

        reasoning_total = Counter(
            r.reasoning_mode for r in records_all if r.reasoning_mode
        )
        reasoning_active = Counter(
            r.reasoning_mode for r in records_active if r.reasoning_mode
        )
        reasoning_labels = {mode: mode for mode in reasoning_total}

        source_total = Counter(
            r.results_source for r in records_all if r.results_source
        )
        source_active = Counter(
            r.results_source for r in records_active if r.results_source
        )
        source_labels = {src: src.upper() for src in source_total}

        failure_total = Counter(
            "true" if r.has_failures else "false" for r in records_all
        )
        failure_active = Counter(
            "true" if r.has_failures else "false" for r in records_active
        )
        failure_labels = {"true": "Has failures", "false": "No failures"}

        return schemas.RunOverviewFacets(
            models=build_options(model_total, model_active, model_labels),
            datasets=build_options(dataset_total, dataset_active, dataset_labels),
            reasoning_modes=build_options(
                reasoning_total, reasoning_active, reasoning_labels
            ),
            results_sources=build_options(source_total, source_active, source_labels),
            has_failures=build_options(failure_total, failure_active, failure_labels),
        )

    def _calculate_total_points_earned_direct(self, paths: RunFilePaths) -> float:
        """Calculate the total points earned for a run, including start capital per grade."""

        total_points = 0.0
        start_points_total = 0.0
        seen_grade_keys: Set[Tuple[str, str]] = set()

        def normalize_year(value: object) -> str:
            if value is None or isinstance(value, bool):
                return "None"
            if isinstance(value, int):
                return str(value)
            if isinstance(value, float):
                if math.isnan(value):
                    return "None"
                if value.is_integer():
                    return str(int(value))
                return str(value)
            text = str(value).strip()
            return text or "None"

        def normalize_group(value: object) -> Optional[Tuple[str, object]]:
            if value is None or isinstance(value, bool):
                return None
            if isinstance(value, int):
                return str(value), value
            if isinstance(value, float):
                if math.isnan(value):
                    return None
                if value.is_integer():
                    integer_value = int(value)
                    return str(integer_value), integer_value
                return str(value), value
            text = str(value).strip()
            if not text:
                return None
            try:
                numeric = float(text)
            except ValueError:
                return text, text
            if math.isfinite(numeric) and numeric.is_integer():
                integer_value = int(numeric)
                return str(integer_value), integer_value
            return text, text

        def process_row(row: schemas.RunResultRow) -> None:
            nonlocal total_points, start_points_total

            if row.points_earned is not None:
                try:
                    total_points += float(row.points_earned)
                except (TypeError, ValueError):
                    pass

            points_possible = coerce_points(row.points)
            if points_possible <= 0.0:
                return

            normalized = normalize_group(row.group)
            if normalized is None:
                return
            group_key, group_value = normalized
            year_key = normalize_year(row.year)
            key = (year_key, group_key)
            if key in seen_grade_keys:
                return
            seen_grade_keys.add(key)
            start_bonus = start_points_for_group(group_value)
            if start_bonus > 0.0:
                start_points_total += start_bonus

        # Determine which results file to use
        results_path = (
            paths.results_jsonl or paths.results_json or paths.results_parquet
        )
        if not results_path:
            return 0.0

        try:
            if paths.results_jsonl:
                for obj in load_jsonl(paths.results_jsonl):
                    row = self._row_from_obj(obj)
                    process_row(row)
            elif paths.results_json:
                data = load_json(paths.results_json)
                for obj in data:
                    row = self._row_from_obj(obj)
                    process_row(row)
            elif paths.results_parquet:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(paths.results_parquet)
                for batch in parquet_file.iter_batches():
                    for obj in batch.to_pylist():
                        row = self._row_from_obj(obj)
                        process_row(row)
        except FileNotFoundError:
            # If results file is missing, return 0
            pass
        return total_points + start_points_total

    def get_run(self, run_id: str) -> RunRecord:
        record = self._runs.get(run_id)
        if not record:
            raise RunNotFoundError(run_id)
        return record

    def get_run_detail(self, run_id: str) -> schemas.RunDetail:
        return self.get_run(run_id).to_detail()

    def get_runs_for_row(self, row_id: str) -> List[str]:
        return self._id_to_runs.get(row_id, [])

    # ------------------------------------------------------------------
    # Results loading & filtering
    # ------------------------------------------------------------------
    def iter_results(self, run_id: str) -> Iterator[schemas.RowRecord]:
        record = self.get_run(run_id)
        paths = record.paths
        # Load based on preferred source set in _load_run_record
        if record.paths.results_parquet:
            parquet_file = pq.ParquetFile(record.paths.results_parquet)
            for batch in parquet_file.iter_batches():
                for obj in batch.to_pylist():
                    yield self._row_from_obj(obj)
            return
        if paths.results_jsonl:
            for obj in load_jsonl(paths.results_jsonl):
                yield self._row_from_obj(obj)
            return
        if paths.results_json:
            data = load_json(paths.results_json)
            for obj in data:
                yield self._row_from_obj(obj)
            return
        if paths.results_parquet:
            parquet_file = pq.ParquetFile(paths.results_parquet)
            for batch in parquet_file.iter_batches():
                for obj in batch.to_pylist():
                    yield self._row_from_obj(obj)
            return
        raise FileNotFoundError(f"Run {run_id} has no results file")

    def _row_from_obj(self, obj: Dict[str, object]) -> schemas.RowRecord:
        data = dict(obj)
        warnings = data.get("warnings")
        if isinstance(warnings, str):
            data["warnings"] = [warnings]
        elif warnings is None:
            data["warnings"] = None
        return schemas.RowRecord.model_validate(data)

    def row_matches_filters(
        self, row: schemas.RowRecord, filters: schemas.ResultFilterParams
    ) -> bool:
        if filters.group and (row.group not in filters.group):
            return False
        if filters.year and (row.year not in filters.year):
            return False
        if filters.language and (row.language not in filters.language):
            return False
        if filters.multimodal is not None:
            row_multimodal = (
                bool(row.multimodal) if row.multimodal is not None else False
            )
            if row_multimodal != filters.multimodal:
                return False
        if filters.correctness:
            if filters.correctness == "true" and row.is_correct is not True:
                return False
            if filters.correctness == "false" and row.is_correct is not False:
                return False
            if filters.correctness == "unknown" and row.is_correct is not None:
                return False
        predicted_values = filters.normalized_predicted()
        if predicted_values:
            pred = row.predicted.upper() if row.predicted else None
            if pred not in predicted_values:
                return False
        if filters.reasoning_mode and (
            row.reasoning_mode not in filters.reasoning_mode
        ):
            return False
        if not self._within_bounds(row.points, filters.points_min, filters.points_max):
            return False
        if not self._within_bounds(
            row.latency_ms, filters.latency_min, filters.latency_max
        ):
            return False
        if not self._within_bounds(
            row.total_tokens, filters.tokens_min, filters.tokens_max
        ):
            return False
        if not self._within_bounds(
            row.reasoning_tokens,
            filters.reasoning_tokens_min,
            filters.reasoning_tokens_max,
        ):
            return False
        if not self._within_bounds(row.cost_usd, filters.cost_min, filters.cost_max):
            return False
        if filters.warnings_present is not None:
            has_warnings = bool(row.warnings)
            if filters.warnings_present != has_warnings:
                return False
        if filters.warning_types:
            if not row.warnings:
                return False
            if not any(w in filters.warning_types for w in row.warnings):
                return False
        return True

    def _within_bounds(
        self,
        value: Optional[float | int],
        minimum: Optional[float | int],
        maximum: Optional[float | int],
    ) -> bool:
        if value is None:
            if minimum is None and maximum is None:
                return True
            return False
        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False
        return True

    def load_results_page(
        self, run_id: str, filters: schemas.ResultFilterParams
    ) -> schemas.ResultsPage:
        rows = [
            row
            for row in self.iter_results(run_id)
            if self.row_matches_filters(row, filters)
        ]
        reverse = filters.sort_dir.lower() == "desc"
        key = self._sort_key(filters.sort_by)
        rows.sort(key=key, reverse=reverse)
        total = len(rows)
        start = (filters.page - 1) * filters.page_size
        end = start + filters.page_size
        items = rows[start:end]
        return schemas.ResultsPage(
            items=items, total=total, page=filters.page, page_size=filters.page_size
        )

    def _sort_key(self, field_name: str):
        lookup = {
            "id": lambda r: r.id,
            "year": lambda r: r.year or "",
            "group": lambda r: grade_group_sort_key(r.group),
            "problem_number": lambda r: (
                int(r.problem_number)
                if isinstance(r.problem_number, (int, float))
                or (isinstance(r.problem_number, str) and r.problem_number.isdigit())
                else (r.problem_number or "")
            ),
            "points": lambda r: r.points or 0.0,
            "is_correct": lambda r: 1 if r.is_correct else 0,
            "latency_ms": lambda r: r.latency_ms or 0.0,
            "total_tokens": lambda r: r.total_tokens or 0,
            "reasoning_tokens": lambda r: r.reasoning_tokens or 0,
            "cost_usd": lambda r: r.cost_usd or 0.0,
            "points_earned": lambda r: r.points_earned or 0.0,
        }
        return lookup.get(field_name, lambda r: getattr(r, field_name, None) or "")

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def get_aggregates(
        self, run_id: str, filters: schemas.ResultFilterParams
    ) -> schemas.AggregatesResponse:
        cache_key = (run_id, filters.cache_key)
        cached = self._aggregate_cache.get(cache_key)
        if cached:
            return cached
        rows = [
            row
            for row in self.iter_results(run_id)
            if self.row_matches_filters(row, filters)
        ]
        aggregates = aggregations.compute_aggregates(rows)
        self._aggregate_cache[cache_key] = aggregates
        return aggregates

    # ------------------------------------------------------------------
    # Compare helpers
    # ------------------------------------------------------------------
    def compare_runs(self, run_a: str, run_b: str) -> schemas.CompareResponse:
        record_a = self.get_run(run_a)
        record_b = self.get_run(run_b)

        rows_a = {row.id: row for row in self.iter_results(run_a)}
        rows_b = {row.id: row for row in self.iter_results(run_b)}

        return aggregations.compute_compare(
            record_a.to_summary(),
            record_b.to_summary(),
            record_a.metrics,
            record_b.metrics,
            rows_a,
            rows_b,
        )

    # ------------------------------------------------------------------
    # Facets
    # ------------------------------------------------------------------
    def get_facets(self, run_id: str) -> schemas.FilterFacets:
        self.get_run(run_id)
        groups: set[str] = set()
        years: set[str] = set()
        languages: set[str] = set()
        reasoning_modes: set[str] = set()
        predicted_letters: set[str] = set()
        warning_types: set[str] = set()

        for row in self.iter_results(run_id):
            if row.group:
                groups.add(str(row.group))
            if row.year:
                years.add(str(row.year))
            if row.language:
                languages.add(str(row.language))
            if row.reasoning_mode:
                reasoning_modes.add(str(row.reasoning_mode))
            if row.predicted:
                predicted_letters.add(row.predicted.upper())
            if row.warnings:
                warning_types.update(row.warnings)

        return schemas.FilterFacets(
            groups=sorted(groups, key=grade_group_sort_key),
            years=sorted(years),
            languages=sorted(languages),
            reasoning_modes=sorted(reasoning_modes),
            predicted_letters=sorted(predicted_letters),
            warning_types=sorted(warning_types),
        )

    # ------------------------------------------------------------------
    # Failures
    # ------------------------------------------------------------------
    def load_failures(self, run_id: str) -> List[schemas.FailureEntry]:
        record = self.get_run(run_id)
        path = record.paths.failures
        if not path or not path.exists():
            return []
        entries = []
        for raw in load_jsonl(path):
            entries.append(
                schemas.FailureEntry(
                    timestamp=str(raw.get("timestamp")),
                    status_code=raw.get("status_code"),
                    message=raw.get("message"),
                    id=raw.get("id"),
                )
            )
        return entries


__all__ = ["RunIndex", "RunNotFoundError", "RowNotFoundError"]
