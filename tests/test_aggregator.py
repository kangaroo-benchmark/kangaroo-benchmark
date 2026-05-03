from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from console.metrics import Aggregator, UsageEvent


def test_aggregator_snapshot_and_projection(tmp_path):
    events_path = tmp_path / "usage_events.jsonl"
    aggregator = Aggregator(
        total_items=10,
        model_id="test/model",
        events_path=events_path,
        recent_items=5,
        min_request_interval=0.5,
    )

    event_success = UsageEvent(
        type="success",
        row_id="row-1",
        year=2024,
        group="A",
        points=5.0,
        problem_number=1,
        multimodal=False,
        latency_ms=1200.0,
        attempts=1,
        prompt_tokens=800,
        completion_tokens=200,
        total_tokens=1000,
        reasoning_tokens=100,
        cached_prompt_tokens=200,
        audio_prompt_tokens=None,
        cost_usd_known=0.12,
        predicted="A",
        correct=True,
        status_code=200,
        warnings=None,
    )

    aggregator.record_event(event_success)

    snapshot = aggregator.snapshot(in_flight=1, worker_count=2)
    assert snapshot.success == 1
    assert snapshot.failure == 0
    assert snapshot.prompt_tokens_total == 800
    assert snapshot.total_tokens_total == 1000
    assert snapshot.cost_projection is not None
    assert snapshot.cost_projection.known_cost == pytest.approx(0.12)
    assert snapshot.cost_projection.projected_total is not None
    # Remaining items: 9, tokens_left ≈ 9000, cost/token ≈ 0.00012 => projected_total ≈ 1.2
    assert snapshot.cost_projection.projected_total == pytest.approx(1.2, rel=1e-3)
    assert snapshot.mean_tokens_per_item == pytest.approx(1000.0)
    assert snapshot.recent_items[0].row_id == "row-1"

    # Failure event without usage should still be recorded
    event_failure = UsageEvent(
        type="failure",
        row_id="row-2",
        year=2024,
        group="B",
        points=3.0,
        problem_number=2,
        multimodal=None,
        latency_ms=800.0,
        attempts=2,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        reasoning_tokens=None,
        cached_prompt_tokens=None,
        audio_prompt_tokens=None,
        cost_usd_known=None,
        predicted=None,
        correct=None,
        status_code=500,
        warnings=["http_status_500"],
    )

    aggregator.record_event(event_failure)
    snapshot2 = aggregator.snapshot(in_flight=0, worker_count=2)
    assert snapshot2.failure == 1
    assert snapshot2.throttles == 0
    assert len(snapshot2.recent_items) >= 2

    aggregator.close()
    assert events_path.exists()
    with events_path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 2
