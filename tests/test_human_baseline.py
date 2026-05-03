import json
from pathlib import Path
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from dashboard.human_baseline import HumanBaselineIndex
from dashboard.server import create_app
from score_utils import start_points_for_group


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "human_baseline"


def _write_mock_run(path: Path, results: list[dict]) -> None:
    path.mkdir()
    normalized_results: list[dict] = []
    total_points = 0.0
    total_earned = 0.0
    correct = 0
    for idx, item in enumerate(results, start=1):
        base = {
            "id": item.get("id", f"row-{idx}"),
            "year": item.get("year", "2008"),
            "group": item.get("group", "3"),
            "problem_number": idx,
            "language": "de",
            "multimodal": False,
            "answer": "A",
            "predicted": "A" if item.get("is_correct") else "B",
            "is_correct": item.get("is_correct", False),
            "points": float(item.get("points", 0.0)),
            "points_earned": float(item.get("points_earned", 0.0)),
            "reasoning_mode": "default",
            "latency_ms": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
            "cost_usd": 0.0,
            "rationale": "",
            "raw_text_response": "",
            "generation_id": f"gen-{idx}",
            "error": None,
            "warnings": None,
        }
        base.update(item)
        normalized_results.append(base)
        total_points += base["points"]
        total_earned += base["points_earned"]
        if base.get("is_correct"):
            correct += 1

    start_points_total = 0.0
    seen_grade_keys: set[tuple[object, object]] = set()
    for entry in normalized_results:
        key = (entry.get("year"), entry.get("group"))
        if key in seen_grade_keys:
            continue
        seen_grade_keys.add(key)
        bonus = start_points_for_group(entry.get("group"))
        if bonus > 0.0:
            start_points_total += bonus

    total_points_with_start = total_points + start_points_total
    total_earned_with_start = total_earned + start_points_total

    metrics = {
        "answered_count": len(results),
        "skipped_count": 0,
        "failed_count": 0,
        "accuracy": (correct / len(results)) if results else 0.0,
        "points_weighted_accuracy": (
            (total_earned_with_start / total_points_with_start)
            if total_points_with_start
            else 0.0
        ),
        "total_points_earned": total_earned_with_start,
        "breakdown_by_group": {},
        "breakdown_by_year": {},
    }
    config = {
        "timestamp_utc": "2025-09-30T00:00:00Z",
        "args": {"dataset": "dataset_sample_50.parquet", "model": "test-model"},
        "model": {"id": "test-model", "label": "Test Model"},
    }
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "config.json").write_text(json.dumps(config))
    (path / "results.json").write_text(json.dumps(normalized_results))


def test_loader_builds_grade_stats():
    index = HumanBaselineIndex(FIXTURE_DIR)

    assert index.available_years() == [2008]
    year = index.get_year(2008)

    grade3 = year.grade("3")
    assert grade3.total_count == 6
    assert pytest.approx(grade3.mean_estimate, rel=1e-3) == 86.5625
    assert grade3.percentile(0.0) == pytest.approx(0.0, abs=1e-6)
    assert grade3.percentile(120.0) == pytest.approx(1.0, abs=1e-6)
    assert grade3.bin_index_for_score(118.0) == 1

    cdf_points = grade3.cdf_points
    assert cdf_points == sorted(cdf_points, key=lambda item: item[0])
    assert cdf_points[-1][1] == pytest.approx(1.0, abs=1e-6)

    # Aggregating two grades should sum counts and recompute stats.
    combined = year.aggregate_grades(["3", "4"], "3-4", "Grades 3-4")
    assert combined is not None
    assert combined.total_count == 12
    assert combined.counts == [1, 5, 6]


def test_missing_directory_is_safe(tmp_path):
    missing_dir = tmp_path / "nope"
    index = HumanBaselineIndex(missing_dir)
    assert index.available_years() == []
    assert not index.has_data()


def test_invalid_counts_raise(tmp_path):
    data_dir = tmp_path / "human_results"
    data_dir.mkdir(parents=True)
    bad_path = data_dir / "human_baseline_1999.json"
    bad_path.write_text(
        """
        {
          "schema_version": "2.0",
          "year": 1999,
          "locale": "DE",
          "grades": [
            {"id": "3-4", "label": "3/4", "members": [3,4], "max_points": 105.0}
          ],
          "bins": [
            {
              "id": "150",
              "label_pdf": "150,00 (105,00)",
              "range_default": {"min": 150.0, "max": 150.0, "inclusive": "both"},
              "ranges_by_grade": {"3-4": {"min": 105.0, "max": 105.0, "inclusive": "both"}}
            }
          ],
          "counts_by_grade": {"3-4": [1]},
          "totals_by_grade": {"3-4": 2},
          "avg_score_by_grade": {"3-4": 90.0}
        }
        """
    )

    with pytest.raises(ValueError):
        HumanBaselineIndex(data_dir)


def test_human_endpoints(tmp_path):
    humans_dir = tmp_path / "humans"
    humans_dir.mkdir()
    fixture_file = FIXTURE_DIR / "human_baseline_2008.json"
    shutil.copy(fixture_file, humans_dir / fixture_file.name)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    run1 = "20250930_000000-modelA"
    run2 = "20250930_010000-modelB"
    _write_mock_run(
        runs_dir / run1,
        results=[
            {
                "id": "g3-1",
                "year": "2008",
                "group": "3",
                "points": 3.0,
                "points_earned": 3.0,
                "is_correct": True,
            },
            {
                "id": "g3-2",
                "year": "2008",
                "group": "3",
                "points": 4.0,
                "points_earned": 4.0,
                "is_correct": True,
            },
            {
                "id": "g3-3",
                "year": "2008",
                "group": "3",
                "points": 5.0,
                "points_earned": -1.25,
                "is_correct": False,
            },
            {
                "id": "g4-1",
                "year": "2008",
                "group": "4",
                "points": 3.0,
                "points_earned": -0.75,
                "is_correct": False,
            },
            {
                "id": "g4-2",
                "year": "2008",
                "group": "4",
                "points": 4.0,
                "points_earned": -1.0,
                "is_correct": False,
            },
            {
                "id": "g4-3",
                "year": "2008",
                "group": "4",
                "points": 5.0,
                "points_earned": -1.25,
                "is_correct": False,
            },
        ],
    )
    _write_mock_run(
        runs_dir / run2,
        results=[
            {
                "id": "g3-1",
                "year": "2008",
                "group": "3",
                "points": 3.0,
                "points_earned": -0.75,
                "is_correct": False,
            },
            {
                "id": "g3-2",
                "year": "2008",
                "group": "3",
                "points": 4.0,
                "points_earned": -1.0,
                "is_correct": False,
            },
            {
                "id": "g3-3",
                "year": "2008",
                "group": "3",
                "points": 5.0,
                "points_earned": -1.25,
                "is_correct": False,
            },
            {
                "id": "g4-1",
                "year": "2008",
                "group": "4",
                "points": 3.0,
                "points_earned": 3.0,
                "is_correct": True,
            },
            {
                "id": "g4-2",
                "year": "2008",
                "group": "4",
                "points": 4.0,
                "points_earned": 4.0,
                "is_correct": True,
            },
            {
                "id": "g4-3",
                "year": "2008",
                "group": "4",
                "points": 5.0,
                "points_earned": -1.25,
                "is_correct": False,
            },
        ],
    )

    app = create_app(
        runs_dir=runs_dir,
        models_path=Path("models.json"),
        templates_dir=Path("web/templates"),
        static_dir=Path("web/static"),
        human_results_dir=humans_dir,
    )
    client = TestClient(app)

    years = client.get("/api/humans/years").json()
    assert years and years[0]["year"] == 2008

    summary = client.get("/api/humans/2008/summary").json()
    assert summary["totals_by_grade"]["3"] == 6

    cdf = client.get("/api/humans/2008/cdf", params={"grade": "3"}).json()
    assert cdf["grade_id"] == "3"
    assert cdf["points"]

    percentile = client.get(
        "/api/humans/percentile",
        params={"year": 2008, "grade": "3", "score": 115.0},
    ).json()
    assert pytest.approx(percentile["percentile"], rel=1e-2) == 0.629

    run_compare = client.get(f"/api/humans/compare/run/{run1}").json()
    assert run_compare["entries"]
    grade3_entry = next(
        item for item in run_compare["entries"] if item["grade_id"] == "3"
    )
    assert pytest.approx(grade3_entry["llm_total"], rel=1e-6) == 29.75
    assert pytest.approx(grade3_entry["llm_start_points"], rel=1e-6) == 24.0
    assert pytest.approx(grade3_entry["llm_points_awarded"], rel=1e-6) == 5.75
    assert pytest.approx(grade3_entry["llm_points_available"], rel=1e-6) == 12.0
    assert grade3_entry["bin_comparison"][2]["llm_share"] == 1.0
    assert (
        "member_overrides" in grade3_entry and "3" in grade3_entry["member_overrides"]
    )
    override3 = grade3_entry["member_overrides"]["3"]
    assert override3["grade_label"] == "3"
    assert override3["human_mean"] is not None

    cohort = client.post(
        "/api/humans/compare/aggregate",
        json={"run_ids": [run1, run2]},
    ).json()
    assert cohort["micro"]["entries"]
    micro_grade3 = next(
        entry for entry in cohort["micro"]["entries"] if entry["grade_id"] == "3"
    )
    assert micro_grade3["sample_count"] == 2
    assert (
        "member_overrides" in micro_grade3 and "3" in micro_grade3["member_overrides"]
    )

    run_summary = client.get(
        f"/api/humans/stats/run/{run1}",
        params={"comparator": "average", "late_year_strategy": "best"},
    ).json()
    summary_payload = run_summary["summary"]
    assert summary_payload["run_ids"] == [run1]
    assert summary_payload["total_cells"] == 2
    assert summary_payload["llm_win_count"] == 0
    assert summary_payload["human_win_count"] == 2
    assert summary_payload["top_llm_wins"] == []
    assert summary_payload["top_human_wins"], "expected at least one strong human win"
    assert summary_payload["best_year"]["year"] == 2008
    assert summary_payload["best_grade"]["grade_id"] in {"3", "4"}

    cohort_summary = client.post(
        "/api/humans/stats/cohort",
        json={
            "run_ids": [run1, run2],
            "comparator": "average",
            "weight_mode": "micro",
            "late_year_strategy": "best",
        },
    ).json()
    cohort_payload = cohort_summary["summary"]
    assert cohort_payload["run_ids"] == sorted([run1, run2])
    assert cohort_payload["total_cells"] == 4

    baseline_summary = client.get(
        "/api/humans/stats/human-baseline",
        params={"comparator": "average"},
    ).json()
    assert baseline_summary["best_year"]["year"] == 2008
    assert baseline_summary["best_grade"]["grade_id"] in {"3", "4"}

    best_comparator_run = client.get(
        f"/api/humans/stats/run/{run1}",
        params={"comparator": "best"},
    ).json()
    assert best_comparator_run["summary"]["comparator"] == "best"
