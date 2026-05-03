from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from console.eta import SmartEta


def test_eta_basic_projection():
    eta = SmartEta()
    base = 0.0
    # Feed stable measurements: 120 tokens in 1.5s => 80 toks/s
    for idx in range(6):
        eta.update(tokens=120, latency_ms=1500, monotonic_ts=base + idx * 0.9)

    estimate = eta.estimate(
        remaining_items=4,
        tokens_left=480.0,
        mean_tokens_per_item=120.0,
        worker_count=2,
        min_request_interval=None,
    )

    assert estimate.seconds is not None
    # ETA should be around 2.76 seconds
    assert 2.7 <= estimate.seconds <= 2.8
    assert estimate.tokens_per_second is not None and estimate.tokens_per_second > 0
    assert estimate.confidence == "low"


def test_eta_outlier_rejection():
    eta = SmartEta()
    for idx in range(8):
        eta.update(tokens=100, latency_ms=1000, monotonic_ts=idx * 1.0)

    before = eta.sample_count
    # Extreme outlier should be ignored once MAD is established
    eta.update(tokens=5000, latency_ms=100, monotonic_ts=8.0)
    after = eta.sample_count

    assert after == before
    estimate = eta.estimate(
        remaining_items=2,
        tokens_left=200.0,
        mean_tokens_per_item=100.0,
        worker_count=1,
        min_request_interval=0.2,
    )
    assert estimate.seconds is not None
    assert estimate.requests_per_second is not None
    # With min_interval=0.2 and worker=1 => <=5 req/s
    assert estimate.requests_per_second <= 5.0 + 1e-6
