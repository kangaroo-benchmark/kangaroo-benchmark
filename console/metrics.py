from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Deque, List, Optional

from .eta import EtaEstimate, SmartEta


@dataclass
class UsageEvent:
    type: str
    row_id: Optional[str]
    year: Optional[int]
    group: Optional[str]
    points: Optional[float]
    problem_number: Optional[int]
    multimodal: Optional[bool]
    latency_ms: Optional[float]
    attempts: int = 1
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    explicit_reasoning_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    audio_prompt_tokens: Optional[int] = None
    cost_usd_known: Optional[float] = None
    cost_usd_calc: Optional[float] = None
    predicted: Optional[str] = None
    correct: Optional[bool] = None
    status_code: Optional[int] = None
    warnings: Optional[List[str]] = None
    timestamp: float = field(default_factory=time.time)
    monotonic_ts: float = field(default_factory=time.monotonic)

    def normalize_tokens(self) -> None:
        if (
            self.total_tokens is None
            and self.prompt_tokens is not None
            and self.completion_tokens is not None
        ):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class RecentItem:
    row_id: Optional[str]
    latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    explicit_reasoning_tokens: Optional[int]
    cached_prompt_tokens: Optional[int]
    cost_usd_known: Optional[float]
    cost_usd_calc: Optional[float]
    predicted: Optional[str]
    correct: Optional[bool]
    status: str
    timestamp: float


@dataclass
class CostProjection:
    known_cost: float
    projected_total: Optional[float]
    projected_low: Optional[float]
    projected_high: Optional[float]
    average_cost_per_item: Optional[float]
    samples: int
    currency: str
    confidence: str


@dataclass
class DashboardSnapshot:
    timestamp: float
    model_id: str
    total_items: int
    completed_items: int
    success: int
    failure: int
    skipped: int
    in_flight: int
    worker_count: int
    eta_seconds: Optional[float]
    eta_confidence: str
    eta_capped: bool
    tokens_per_second: Optional[float]
    requests_per_second: Optional[float]
    prompt_tokens_total: int
    completion_tokens_total: int
    total_tokens_total: int
    reasoning_tokens_total: int
    explicit_reasoning_tokens_total: int
    cached_prompt_tokens_total: int
    items_with_usage: int
    mean_tokens_per_item: Optional[float]
    median_tokens_per_item: Optional[float]
    p90_tokens_per_item: Optional[float]
    mean_latency_ms: Optional[float]
    mean_attempts: Optional[float]
    throttles: int
    last_status: Optional[int]
    min_request_interval: Optional[float]
    cost_projection: Optional[CostProjection]
    recent_items: List[RecentItem]
    elapsed_time_seconds: Optional[float] = None


class Aggregator:
    def __init__(
        self,
        total_items: int,
        *,
        model_id: str,
        events_path: Optional[Path] = None,
        recent_items: int = 20,
        min_request_interval: Optional[float] = None,
    ) -> None:
        self.total_items = total_items
        self.model_id = model_id
        self._events_path = Path(events_path) if events_path else None
        self._recent_capacity = max(5, recent_items)
        self._min_request_interval = min_request_interval

        self._start_time = time.time()
        self._lock = Lock()

        self._success = 0
        self._failure = 0
        self._skipped = 0
        self._completed = 0

        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._total_tokens_total = 0
        self._reasoning_tokens_total = 0
        self._explicit_reasoning_tokens_total = 0
        self._cached_prompt_tokens_total = 0

        self._items_with_usage = 0
        self._token_samples: Deque[int] = deque(maxlen=512)

        self._latency_sum = 0.0
        self._latency_count = 0
        self._attempts_sum = 0

        self._recent: Deque[RecentItem] = deque(maxlen=self._recent_capacity)

        self._cost_known_total = 0.0
        self._cost_with_tokens_total = 0.0
        self._tokens_with_cost = 0
        self._tokens_without_cost = 0
        self._items_with_cost = 0
        self._items_without_cost = 0

        self._eta = SmartEta()
        self._throttles = 0
        self._last_status: Optional[int] = None

        self._events_file = None
        if self._events_path is not None:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            self._events_file = self._events_path.open("a", encoding="utf-8")

    # Lifecycle -----------------------------------------------------------------
    def close(self) -> None:
        if self._events_file is not None:
            self._events_file.close()
            self._events_file = None

    # Event ingestion -----------------------------------------------------------
    def record_event(self, event: UsageEvent) -> None:
        event.normalize_tokens()
        with self._lock:
            self._record_event_locked(event)

    def _record_event_locked(self, event: UsageEvent) -> None:
        status = event.type
        if status == "success":
            self._success += 1
        elif status == "failure":
            self._failure += 1
        elif status == "skipped":
            self._skipped += 1
        self._completed = self._success + self._failure + self._skipped

        latency_ms = event.latency_ms if event.latency_ms is not None else None
        if latency_ms is not None and latency_ms > 0:
            self._latency_sum += latency_ms
            self._latency_count += 1
        self._attempts_sum += max(event.attempts, 1)

        prompt_tokens = event.prompt_tokens or 0
        completion_tokens = event.completion_tokens or 0
        total_tokens = event.total_tokens or (prompt_tokens + completion_tokens) or 0
        reasoning_tokens = event.reasoning_tokens or 0
        explicit_reasoning_tokens = event.explicit_reasoning_tokens or 0
        cached_prompt_tokens = event.cached_prompt_tokens or 0

        if total_tokens > 0:
            self._total_tokens_total += total_tokens
            self._items_with_usage += 1
            self._token_samples.append(total_tokens)
        if prompt_tokens > 0:
            self._prompt_tokens_total += prompt_tokens
        if completion_tokens > 0:
            self._completion_tokens_total += completion_tokens
        if reasoning_tokens > 0:
            self._reasoning_tokens_total += reasoning_tokens
        if explicit_reasoning_tokens > 0:
            self._explicit_reasoning_tokens_total += explicit_reasoning_tokens
        if cached_prompt_tokens > 0:
            self._cached_prompt_tokens_total += cached_prompt_tokens

        if event.cost_usd_known is not None:
            cost_val = float(event.cost_usd_known)
            self._cost_known_total += cost_val
            self._items_with_cost += 1
            if total_tokens > 0:
                self._cost_with_tokens_total += cost_val
                self._tokens_with_cost += total_tokens
        else:
            self._items_without_cost += 1
            if total_tokens > 0:
                self._tokens_without_cost += total_tokens

        self._eta.update(
            tokens=total_tokens if total_tokens > 0 else None,
            latency_ms=latency_ms,
            monotonic_ts=event.monotonic_ts,
        )

        self._recent.appendleft(
            RecentItem(
                row_id=event.row_id,
                latency_ms=latency_ms,
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                total_tokens=event.total_tokens,
                reasoning_tokens=event.reasoning_tokens,
                explicit_reasoning_tokens=event.explicit_reasoning_tokens,
                cached_prompt_tokens=event.cached_prompt_tokens,
                cost_usd_known=event.cost_usd_known,
                cost_usd_calc=event.cost_usd_calc,
                predicted=event.predicted,
                correct=event.correct,
                status=status,
                timestamp=event.timestamp,
            )
        )

        if event.status_code is not None:
            self._last_status = event.status_code

        if self._events_file is not None:
            json.dump(asdict(event), self._events_file, ensure_ascii=False)
            self._events_file.write("\n")
            self._events_file.flush()

    # Derived data --------------------------------------------------------------
    def record_throttle(self) -> None:
        with self._lock:
            self._throttles += 1

    def record_status(self, status_code: int) -> None:
        with self._lock:
            self._last_status = status_code

    def _mean_tokens(self) -> Optional[float]:
        if self._items_with_usage == 0:
            return None
        return self._total_tokens_total / float(self._items_with_usage)

    def _median_tokens(self) -> Optional[float]:
        if not self._token_samples:
            return None
        ordered = sorted(self._token_samples)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return float(ordered[mid])
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _p90_tokens(self) -> Optional[float]:
        if not self._token_samples:
            return None
        ordered = sorted(self._token_samples)
        index = max(0, int(math.ceil(0.9 * len(ordered)) - 1))
        return float(ordered[index])

    def project_cost(
        self, *, tokens_left: Optional[float], remaining_items: int
    ) -> CostProjection:
        mean_cost_per_token = None
        if self._tokens_with_cost > 0:
            mean_cost_per_token = self._cost_with_tokens_total / float(
                self._tokens_with_cost
            )

        mean_cost_per_item = None
        if self._items_with_cost > 0:
            mean_cost_per_item = self._cost_known_total / float(self._items_with_cost)

        projected_total = None
        # Estimate based on total items and mean cost per item
        if mean_cost_per_item is not None:
            projected_total = mean_cost_per_item * self.total_items

        # Refine projection with token-based estimate if available
        if mean_cost_per_token is not None:
            # Estimate cost for items that have already completed
            cost_of_completed = self._cost_known_total
            if self._tokens_without_cost > 0 and mean_cost_per_token is not None:
                cost_of_completed += self._tokens_without_cost * mean_cost_per_token

            # Estimate cost for remaining items
            cost_of_remaining = 0.0
            if remaining_items > 0:
                if tokens_left is not None:
                    cost_of_remaining = tokens_left * mean_cost_per_token
                elif self._mean_tokens() is not None:
                    cost_of_remaining = (
                        remaining_items * self._mean_tokens() * mean_cost_per_token
                    )

            projected_total = cost_of_completed + cost_of_remaining

        # Fallback if no projections could be made
        if projected_total is None:
            projected_total = self._cost_known_total

        # Confidence bands - adjust band size based on sample size
        if projected_total > self._cost_known_total:
            # Reduce band size if we have more samples
            base_band = 0.1
            if self._items_with_cost >= 50:
                base_band = 0.05  # 5% band for high confidence
            elif self._items_with_cost >= 20:
                base_band = 0.07  # 7% band for medium confidence
            band = max(projected_total * base_band, 0.01)
            projected_low = max(self._cost_known_total, projected_total - band)
            projected_high = projected_total + band
        else:
            projected_low = None
            projected_high = None

        confidence = "low"
        if self._items_with_cost >= 50:
            confidence = "high"
        elif self._items_with_cost >= 10:
            confidence = "medium"

        if (
            mean_cost_per_token is None
            and mean_cost_per_item is not None
            and confidence == "high"
        ):
            confidence = "medium"

        return CostProjection(
            known_cost=self._cost_known_total,
            projected_total=projected_total if projected_total > 0 else None,
            projected_low=projected_low,
            projected_high=projected_high,
            average_cost_per_item=mean_cost_per_item,
            samples=self._items_with_cost,
            currency="USD",
            confidence=confidence,
        )

    def project_eta(self, *, worker_count: int) -> EtaEstimate:
        remaining = max(self.total_items - self._completed, 0)
        tokens_left = None
        mean_tokens = self._mean_tokens()
        if mean_tokens is not None:
            tokens_left = mean_tokens * remaining
        return self._eta.estimate(
            remaining_items=remaining,
            tokens_left=tokens_left,
            mean_tokens_per_item=mean_tokens,
            worker_count=worker_count,
            min_request_interval=self._min_request_interval,
        )

    def snapshot(self, *, in_flight: int, worker_count: int) -> DashboardSnapshot:
        with self._lock:
            current_time = time.time()
            elapsed_time = current_time - self._start_time
            eta = self.project_eta(worker_count=worker_count)
            remaining = max(self.total_items - self._completed, 0)
            mean_tokens = self._mean_tokens()
            tokens_left = mean_tokens * remaining if mean_tokens is not None else None
            cost_projection = self.project_cost(
                tokens_left=tokens_left, remaining_items=remaining
            )
            mean_latency = (
                (self._latency_sum / self._latency_count)
                if self._latency_count
                else None
            )
            mean_attempts = (
                (self._attempts_sum / self._completed) if self._completed else None
            )
            snapshot = DashboardSnapshot(
                timestamp=current_time,
                model_id=self.model_id,
                total_items=self.total_items,
                completed_items=self._completed,
                success=self._success,
                failure=self._failure,
                skipped=self._skipped,
                in_flight=in_flight,
                worker_count=worker_count,
                eta_seconds=eta.seconds,
                eta_confidence=eta.confidence,
                eta_capped=eta.capped,
                tokens_per_second=eta.tokens_per_second,
                requests_per_second=eta.requests_per_second,
                prompt_tokens_total=self._prompt_tokens_total,
                completion_tokens_total=self._completion_tokens_total,
                total_tokens_total=self._total_tokens_total,
                reasoning_tokens_total=self._reasoning_tokens_total,
                explicit_reasoning_tokens_total=self._explicit_reasoning_tokens_total,
                cached_prompt_tokens_total=self._cached_prompt_tokens_total,
                items_with_usage=self._items_with_usage,
                mean_tokens_per_item=mean_tokens,
                median_tokens_per_item=self._median_tokens(),
                p90_tokens_per_item=self._p90_tokens(),
                mean_latency_ms=mean_latency,
                mean_attempts=mean_attempts,
                throttles=self._throttles,
                last_status=self._last_status,
                min_request_interval=self._min_request_interval,
                cost_projection=cost_projection,
                recent_items=list(self._recent),
                elapsed_time_seconds=elapsed_time,
            )
            return snapshot


__all__ = [
    "UsageEvent",
    "RecentItem",
    "CostProjection",
    "DashboardSnapshot",
    "Aggregator",
]
