"""FastAPI application wiring for the dashboard."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, Request, Form
from fastapi.responses import HTMLResponse, ORJSONResponse, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import humanize

from . import schemas
from . import analysis
from .data_index import RunIndex, RunNotFoundError
from .dataset_access import DatasetAccessor
from .human_baseline import HumanBaselineIndex
from .human_compare import (
    compute_cohort_human_comparison,
    compute_run_human_comparison,
)
from .human_stats import (
    compute_cohort_summary,
    compute_human_baseline_summary,
    compute_run_summary,
)

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_MODELS_PATH = Path("models.json")
DEFAULT_TEMPLATE_DIR = Path("web/templates")
DEFAULT_STATIC_DIR = Path("web/static")
HUMAN_PERF_PATH = Path("human_performance.parquet")

LIST_FIELDS = {
    "group",
    "year",
    "language",
    "predicted",
    "predicted_letter",
    "reasoning_mode",
    "warning_types",
}
FLOAT_FIELDS = {
    "points_min",
    "points_max",
    "latency_min",
    "latency_max",
    "cost_min",
    "cost_max",
}
INT_FIELDS = {
    "tokens_min",
    "tokens_max",
    "reasoning_tokens_min",
    "reasoning_tokens_max",
    "page",
    "page_size",
}
BOOL_FIELDS = {"multimodal", "warnings_present"}


def create_app(
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    models_path: Path | str = DEFAULT_MODELS_PATH,
    templates_dir: Path | str = DEFAULT_TEMPLATE_DIR,
    static_dir: Path | str = DEFAULT_STATIC_DIR,
    human_results_dir: Path | str = Path("human_results"),
) -> FastAPI:
    app = FastAPI(default_response_class=ORJSONResponse)
    runs_path = Path(runs_dir)
    models_path = Path(models_path)
    templates_path = Path(templates_dir)
    static_path = Path(static_dir)

    index = RunIndex(runs_path, models_path)
    dataset_accessor = DatasetAccessor(human_performance_path=HUMAN_PERF_PATH)
    humans = HumanBaselineIndex(Path(human_results_dir), strict=False)
    templates = Jinja2Templates(directory=str(templates_path))
    templates.env.filters.setdefault("intcomma", humanize.intcomma)

    app.state.index = index
    app.state.dataset_accessor = dataset_accessor
    app.state.templates = templates
    app.state.humans = humans

    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        candidate = static_path / "assets" / "placeholder.svg"
        if candidate.exists():
            return FileResponse(str(candidate), media_type="image/svg+xml")
        return RedirectResponse(url="/static/assets/placeholder.svg")

    def _parse_datetime(value: str) -> Optional[datetime]:
        raw = value.strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    def parse_run_overview_filters(request: Request) -> schemas.RunOverviewFilterParams:
        filters = schemas.RunOverviewFilterParams()
        multi = request.query_params.multi_items()
        for key, value in multi:
            if value == "":
                continue
            if key in {"model", "dataset", "reasoning_mode", "results_source"}:
                getattr(filters, key).append(value)
            elif key == "has_failures":
                lowered = value.lower()
                if lowered == "any":
                    filters.has_failures = None
                elif lowered in {"true", "1", "yes"}:
                    filters.has_failures = True
                elif lowered in {"false", "0", "no"}:
                    filters.has_failures = False
            elif key == "q":
                filters.q = value
            elif key == "date_from":
                parsed = _parse_datetime(value)
                if parsed:
                    filters.date_from = parsed
            elif key == "date_to":
                parsed = _parse_datetime(value)
                if parsed:
                    filters.date_to = parsed
            elif key == "sort_by":
                if value in {"timestamp", "accuracy", "answered", "failed"}:
                    filters.sort_by = value
            elif key == "sort_dir":
                lowered = value.lower()
                if lowered in {"asc", "desc"}:
                    filters.sort_dir = lowered
        return filters

    def _parse_thresholds_param(raw: Optional[str]) -> Optional[List[float]]:
        if raw is None or not raw.strip():
            return None
        parts = [segment.strip() for segment in raw.split(",") if segment.strip()]
        thresholds: List[float] = []
        for segment in parts:
            try:
                value = float(segment)
            except ValueError:
                continue
            if value < 0:
                continue
            thresholds.append(value / 100.0 if value >= 1 else value)
        return thresholds or None

    @app.on_event("startup")
    async def _startup() -> None:
        index.reload()
        humans.reload()

    @app.get("/", response_class=HTMLResponse)
    async def overview(request: Request) -> HTMLResponse:
        filters = parse_run_overview_filters(request)
        runs = index.list_runs(filters)
        facets = index.get_run_overview_facets(filters)
        total_runs = index.total_runs()
        context = {
            "request": request,
            "runs": runs,
            "filters": filters,
            "filters_data": filters.model_dump(),
            "facets": facets,
            "run_count": len(runs),
            "total_runs": total_runs,
        }
        template_name = (
            "partials/overview_section.html"
            if request.headers.get("HX-Request")
            else "overview.html"
        )
        return templates.TemplateResponse(template_name, context)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        try:
            detail = index.get_run_detail(run_id)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return templates.TemplateResponse(
            "run_detail.html",
            {"request": request, "run": detail},
        )

    @app.get("/compare", response_class=HTMLResponse)
    async def compare(
        request: Request, run_a: Optional[str] = None, run_b: Optional[str] = None
    ) -> HTMLResponse:
        runs = index.list_runs()
        context = {"request": request, "runs": runs, "run_a": run_a, "run_b": run_b}
        if run_a and run_b:
            try:
                compare_payload = index.compare_runs(run_a, run_b)
            except RunNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            context["compare"] = compare_payload
        return templates.TemplateResponse("compare.html", context)

    @app.get("/analysis", response_class=HTMLResponse)
    async def analysis_page(request: Request) -> HTMLResponse:
        runs = index.list_runs()
        return templates.TemplateResponse(
            "analysis.html", {"request": request, "runs": runs}
        )

    @app.get("/humans", response_class=HTMLResponse)
    async def humans_page(request: Request) -> HTMLResponse:
        runs = index.list_runs()
        years = humans.year_list()
        bootstrap = {
            "runs": [run.model_dump() for run in runs],
            "years": [entry.model_dump() for entry in years],
        }
        context = {
            "request": request,
            "runs": runs,
            "bootstrap": bootstrap,
            "has_humans": bool(years),
        }
        return templates.TemplateResponse("humans.html", context)

    # ------------------------------------------------------------------
    # API Endpoints
    # ------------------------------------------------------------------

    @app.post("/api/analysis")
    async def api_analysis(request: Request, run_ids: List[str] = Form(...)):
        question_analysis = analysis.analyze_question_difficulty(run_ids, index)

        dataset_path = None
        if run_ids:
            try:
                first_run_detail = index.get_run_detail(run_ids[0])
                dataset_path = first_run_detail.config.args.dataset
            except RunNotFoundError:
                pass

        scatter_data = []
        yearly_agg = defaultdict(lambda: {"human_scores": [], "llm_scores": []})
        tag_agg = defaultdict(
            lambda: {"human_scores": [], "llm_scores": [], "count": 0}
        )

        for record in question_analysis:
            dataset_row = dataset_accessor.get_row(dataset_path, record.question_id)
            human_p_correct = None
            if dataset_row and dataset_row.human_performance:
                human_p_correct = dataset_row.human_performance.p_correct

            scatter_data.append(
                {
                    "question_id": record.question_id,
                    "avg_llm_score": record.avg_llm_score,
                    "human_p_correct": human_p_correct,
                    "llm_disagreement": record.llm_disagreement,
                    "llm_count": record.llm_count,
                }
            )

            if dataset_row and dataset_row.year and human_p_correct is not None:
                year = dataset_row.year
                yearly_agg[year]["human_scores"].append(human_p_correct)
                yearly_agg[year]["llm_scores"].append(record.avg_llm_score)

            if dataset_row and dataset_row.tags:
                for tag in dataset_row.tags:
                    tag_agg[tag]["count"] += 1
                    if human_p_correct is not None:
                        tag_agg[tag]["human_scores"].append(human_p_correct)
                    tag_agg[tag]["llm_scores"].append(record.avg_llm_score)

        bar_data = []
        for year, data in yearly_agg.items():
            avg_human_score = (
                np.mean(data["human_scores"]) if data["human_scores"] else 0
            )
            avg_llm_score = np.mean(data["llm_scores"]) if data["llm_scores"] else 0
            normalized_human_score = (
                avg_human_score / avg_llm_score if avg_llm_score > 0 else 0
            )
            bar_data.append(
                {
                    "year": year,
                    "avg_human_score": avg_human_score,
                    "avg_llm_score": avg_llm_score,
                    "normalized_human_score": normalized_human_score,
                }
            )

        tag_data = []
        for tag, data in tag_agg.items():
            avg_human_score = (
                np.mean(data["human_scores"]) if data["human_scores"] else 0
            )
            avg_llm_score = np.mean(data["llm_scores"]) if data["llm_scores"] else 0
            tag_data.append(
                {
                    "tag": tag,
                    "avg_human_score": avg_human_score,
                    "avg_llm_score": avg_llm_score,
                    "count": data["count"],
                }
            )

        return {
            "scatter": scatter_data,
            "bars": sorted(bar_data, key=lambda x: x["year"]),
            "tags": sorted(tag_data, key=lambda x: x["tag"]),
        }

    @app.get("/api/runs", response_model=schemas.RunListResponse)
    async def api_list_runs(request: Request) -> schemas.RunListResponse:
        filters = parse_run_overview_filters(request)
        runs = index.list_runs(filters)
        facets = index.get_run_overview_facets(filters)
        return schemas.RunListResponse(
            runs=runs,
            total=len(runs),
            total_all=index.total_runs(),
            facets=facets,
        )

    @app.get("/api/humans/years", response_model=List[schemas.HumanYearListEntry])
    async def api_human_years() -> List[schemas.HumanYearListEntry]:
        return humans.year_list()

    @app.get("/api/humans/{year}/summary", response_model=schemas.HumanYearSummary)
    async def api_human_year_summary(year: int) -> schemas.HumanYearSummary:
        try:
            return humans.year_summary(year)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/humans/{year}/cdf", response_model=schemas.HumanCDFResponse)
    async def api_human_cdf(
        year: int, grade: str = Query(..., description="Grade id")
    ) -> schemas.HumanCDFResponse:
        response = humans.cdf_response(year, grade)
        if response is None:
            raise HTTPException(
                status_code=404, detail="Human baseline not found for requested grade"
            )
        return response

    @app.get("/api/humans/percentile", response_model=schemas.HumanPercentileResponse)
    async def api_human_percentile(
        year: int = Query(...),
        grade: str = Query(...),
        score: float = Query(...),
    ) -> schemas.HumanPercentileResponse:
        response = humans.percentile_response(year, grade, score)
        if response is None:
            raise HTTPException(
                status_code=404, detail="Human baseline not found for requested grade"
            )
        return response

    @app.get(
        "/api/humans/compare/run/{run_id}",
        response_model=schemas.HumanRunComparisonResponse,
    )
    async def api_human_compare_run(
        run_id: str, late_year_strategy: str = "best"
    ) -> schemas.HumanRunComparisonResponse:
        try:
            return compute_run_human_comparison(
                run_id, index, humans, late_year_strategy=late_year_strategy
            )
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get(
        "/api/humans/stats/run/{run_id}",
        response_model=schemas.HumanComparisonSummaryResponse,
    )
    async def api_human_stats_run(
        run_id: str,
        comparator: Literal["average", "best"] = Query("average"),
        weight_mode: Literal["micro", "macro"] = Query("micro"),
        late_year_strategy: str = Query("best"),
        top_limit: int = Query(5, ge=1, le=50),
        thresholds: Optional[str] = Query(
            None,
            description="Optional comma-separated thresholds in either fraction (0.05) or percentage (5) form.",
        ),
    ) -> schemas.HumanComparisonSummaryResponse:
        custom_thresholds = _parse_thresholds_param(thresholds)
        extra_kwargs = {}
        if custom_thresholds:
            extra_kwargs["thresholds"] = custom_thresholds
        try:
            summary = compute_run_summary(
                run_id,
                index,
                humans,
                comparator=comparator,
                weight_mode=weight_mode,
                late_year_strategy=late_year_strategy,
                top_limit=top_limit,
                **extra_kwargs,
            )
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return schemas.HumanComparisonSummaryResponse(summary=summary)

    @app.post(
        "/api/humans/compare/aggregate",
        response_model=schemas.HumanCohortComparisonResponse,
    )
    async def api_human_compare_cohort(
        run_ids: List[str] = Body(..., embed=True, description="Run ids to aggregate"),
    ) -> schemas.HumanCohortComparisonResponse:
        if not run_ids:
            raise HTTPException(status_code=400, detail="run_ids must not be empty")
        return compute_cohort_human_comparison(run_ids, index, humans)

    @app.post(
        "/api/humans/stats/cohort",
        response_model=schemas.HumanComparisonSummaryResponse,
    )
    async def api_human_stats_cohort(
        payload: schemas.HumanCohortSummaryRequest,
    ) -> schemas.HumanComparisonSummaryResponse:
        if not payload.run_ids:
            raise HTTPException(status_code=400, detail="run_ids must not be empty")
        custom_thresholds = payload.thresholds or None
        extra_kwargs = {}
        if custom_thresholds:
            extra_kwargs["thresholds"] = custom_thresholds
        try:
            summary = compute_cohort_summary(
                payload.run_ids,
                index,
                humans,
                comparator=payload.comparator,
                weight_mode=payload.weight_mode,
                late_year_strategy=payload.late_year_strategy,
                top_limit=payload.top_limit,
                **extra_kwargs,
            )
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return schemas.HumanComparisonSummaryResponse(summary=summary)

    @app.get(
        "/api/humans/stats/human-baseline",
        response_model=schemas.HumanBaselineSummary,
    )
    async def api_human_baseline_stats(
        comparator: Literal["average", "best"] = Query("average"),
    ) -> schemas.HumanBaselineSummary:
        return compute_human_baseline_summary(humans, comparator=comparator)

    @app.get("/api/runs/{run_id}", response_model=schemas.RunDetail)
    async def api_run_detail(run_id: str) -> schemas.RunDetail:
        try:
            return index.get_run_detail(run_id)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    def parse_filters(request: Request) -> schemas.ResultFilterParams:
        data: Dict[str, object] = {}
        multi = request.query_params.multi_items()
        for key, value in multi:
            if key == "download":
                continue
            if value == "":
                continue
            if key in LIST_FIELDS:
                data.setdefault(key, [])
                assert isinstance(data[key], list)
                data[key].append(value)
            elif key in FLOAT_FIELDS:
                try:
                    data[key] = float(value)
                except ValueError as exc:  # pragma: no cover - FastAPI converts to 422
                    raise HTTPException(
                        status_code=400, detail=f"Invalid float for {key}"
                    ) from exc
            elif key in INT_FIELDS:
                try:
                    data[key] = int(value)
                except ValueError as exc:  # pragma: no cover - FastAPI converts to 422
                    raise HTTPException(
                        status_code=400, detail=f"Invalid integer for {key}"
                    ) from exc
            elif key in BOOL_FIELDS:
                data[key] = value.lower() in {"1", "true", "yes"}
            else:
                data[key] = value
        return schemas.ResultFilterParams(**data)

    def filtered_rows(
        run_id: str, filters: schemas.ResultFilterParams
    ) -> List[schemas.RowRecord]:
        return [
            row
            for row in index.iter_results(run_id)
            if index.row_matches_filters(row, filters)
        ]

    @app.get("/api/runs/{run_id}/results", response_model=schemas.ResultsPage)
    async def api_run_results(
        request: Request, run_id: str, download: Optional[str] = Query(default=None)
    ):
        filters = parse_filters(request)
        try:
            if download:
                rows = filtered_rows(run_id, filters)
                if download == "json":
                    payload = [row.model_dump() for row in rows]
                    content = ORJSONResponse(content=payload)
                    filename = f"{run_id}_results.json"
                    content.headers["Content-Disposition"] = (
                        f"attachment; filename={filename}"
                    )
                    return content
                if download == "csv":
                    buffer = io.StringIO()
                    fieldnames = list(schemas.RowRecord.model_fields.keys())
                    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row.model_dump())
                    csv_bytes = buffer.getvalue().encode("utf-8")
                    headers = {
                        "Content-Disposition": f"attachment; filename={run_id}_results.csv"
                    }
                    return Response(
                        content=csv_bytes, media_type="text/csv", headers=headers
                    )
                raise HTTPException(
                    status_code=400, detail="Unsupported download format"
                )
            page = index.load_results_page(run_id, filters)
            return page
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/aggregates", response_model=schemas.AggregatesResponse)
    async def api_run_aggregates(request: Request, run_id: str):
        filters = parse_filters(request)
        try:
            return index.get_aggregates(run_id, filters)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/facets", response_model=schemas.FilterFacets)
    async def api_run_facets(run_id: str):
        try:
            return index.get_facets(run_id)
        except (RunNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get(
        "/api/runs/{run_id}/row/{row_id}", response_model=schemas.RowDetailResponse
    )
    async def api_row_detail(run_id: str, row_id: str):
        try:
            record = index.get_run(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        row = next((r for r in index.iter_results(run_id) if r.id == row_id), None)
        if not row:
            raise HTTPException(status_code=404, detail="Row not found")
        dataset = dataset_accessor.get_row(record.config.args.dataset, row_id)
        return schemas.RowDetailResponse(row=row, dataset=dataset)

    @app.get("/api/runs/{run_id}/failures", response_model=List[schemas.FailureEntry])
    async def api_run_failures(run_id: str) -> List[schemas.FailureEntry]:
        try:
            return index.load_failures(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/compare", response_model=schemas.CompareResponse)
    async def api_compare(
        run_a: str = Query(..., description="Baseline run id"),
        run_b: str = Query(..., description="Run id to compare"),
        view: str = Query(
            "all", description="Filter row deltas: all|changed|improved|regressed"
        ),
        limit: Optional[int] = Query(None, ge=1, le=2000),
        include_confusion: bool = Query(
            False, description="Include confusion matrices for both runs"
        ),
    ):
        try:
            compare_payload = index.compare_runs(run_a, run_b)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        if view != "all":
            row_deltas = filter_row_deltas(compare_payload.row_deltas, view)
        else:
            row_deltas = compare_payload.row_deltas
        if limit is not None:
            row_deltas = row_deltas[:limit]
        compare_payload.row_deltas = row_deltas
        if not include_confusion:
            compare_payload.confusion_matrices = {}
        return compare_payload

    def filter_row_deltas(
        rows: List[schemas.CompareRowDelta], view: str
    ) -> List[schemas.CompareRowDelta]:
        if view == "changed":
            return [
                r
                for r in rows
                if r.run_a_correct != r.run_b_correct
                or r.run_a_predicted != r.run_b_predicted
            ]
        if view == "improved":
            return [
                r
                for r in rows
                if (r.run_a_correct is False and r.run_b_correct is True)
                or (r.delta_points is not None and r.delta_points > 0)
            ]
        if view == "regressed":
            return [
                r
                for r in rows
                if (r.run_a_correct is True and r.run_b_correct is False)
                or (r.delta_points is not None and r.delta_points < 0)
            ]
        return rows

    @app.post("/api/reload")
    async def api_reload() -> Dict[str, str]:
        index.reload()
        humans.reload()
        return {"status": "reloaded", "run_count": str(len(index.list_runs()))}

    return app


__all__ = ["create_app"]
