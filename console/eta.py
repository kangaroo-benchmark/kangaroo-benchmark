from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class EtaEstimate:
    seconds: Optional[float]
    confidence: str
    tokens_per_second: Optional[float]
    requests_per_second: Optional[float]
    capped: bool = False


class SmartEta:
    def __init__(self, *, window: int = 128, alpha: float = 0.2) -> None:
        self._token_rates: Deque[float] = deque(maxlen=window)
        self._req_intervals: Deque[float] = deque(maxlen=window)
        self._alpha = alpha
        self._token_rate_ema: Optional[float] = None
        self._req_rate_ema: Optional[float] = None
        self._samples = 0
        self._last_ts: Optional[float] = None
        self._mad: Optional[float] = None
        self._median: Optional[float] = None

    # Update ------------------------------------------------------------------
    def update(
        self,
        *,
        tokens: Optional[int],
        latency_ms: Optional[float],
        monotonic_ts: Optional[float],
    ) -> None:
        if monotonic_ts is not None:
            if self._last_ts is not None:
                interval = max(monotonic_ts - self._last_ts, 1e-6)
                self._req_intervals.append(interval)
                req_rate = 1.0 / interval
                self._req_rate_ema = self._ema(self._req_rate_ema, req_rate)
            self._last_ts = monotonic_ts

        if tokens is None or latency_ms is None or latency_ms <= 0:
            return

        rate = float(tokens) / (latency_ms / 1000.0)
        if self._is_outlier(rate):
            return

        self._token_rates.append(rate)
        self._token_rate_ema = self._ema(self._token_rate_ema, rate)
        self._samples += 1
        self._recompute_spread()

    # Helpers -----------------------------------------------------------------
    def _ema(self, current: Optional[float], sample: float) -> float:
        if current is None:
            return sample
        return (1.0 - self._alpha) * current + self._alpha * sample

    def _is_outlier(self, rate: float) -> bool:
        if len(self._token_rates) < 5 or self._mad is None or self._median is None:
            return False
        deviation = abs(rate - self._median)
        threshold = 4.0 * self._mad
        return deviation > threshold

    def _recompute_spread(self) -> None:
        if not self._token_rates:
            self._mad = None
            self._median = None
            return
        rates = list(self._token_rates)
        self._median = statistics.median(rates)
        if len(rates) < 5:
            self._mad = None
            return
        deviations = [abs(r - self._median) for r in rates]
        mad = statistics.median(deviations)
        # Consistency constant to approximate std dev
        self._mad = 1.4826 * mad if mad else 0.0

    # Public API ---------------------------------------------------------------
    def estimate(
        self,
        *,
        remaining_items: int,
        tokens_left: Optional[float],
        mean_tokens_per_item: Optional[float],
        worker_count: int,
        min_request_interval: Optional[float],
    ) -> EtaEstimate:
        request_rate = self._request_rate()
        token_rate = self._token_rate()
        capped = False

        if worker_count > 1:
            if token_rate is not None:
                token_rate *= worker_count
            if request_rate is not None:
                request_rate *= worker_count

        if min_request_interval and min_request_interval > 0 and worker_count > 0:
            req_limit = worker_count / max(min_request_interval, 1e-6)
            if request_rate is None or request_rate <= 0:
                request_rate = req_limit
                capped = True
            elif request_rate > req_limit:
                request_rate = req_limit
                capped = True
            if token_rate is not None and mean_tokens_per_item:
                token_limit = req_limit * mean_tokens_per_item
                if token_rate > token_limit:
                    token_rate = token_limit
                    capped = True

        eta_from_tokens: Optional[float] = None
        if tokens_left is not None and token_rate and token_rate > 0:
            eta_from_tokens = tokens_left / token_rate

        eta_from_items: Optional[float] = None
        if remaining_items > 0 and request_rate and request_rate > 0:
            eta_from_items = remaining_items / request_rate

        eta_seconds = None
        if eta_from_tokens is not None and eta_from_items is not None:
            # Weighted average, giving more weight to token-based estimate
            eta_seconds = 0.8 * eta_from_tokens + 0.2 * eta_from_items
        elif eta_from_tokens is not None:
            eta_seconds = eta_from_tokens
        elif eta_from_items is not None:
            eta_seconds = eta_from_items

        confidence = self._confidence(remaining_items)
        if eta_seconds is None:
            confidence = "low"

        return EtaEstimate(
            seconds=eta_seconds,
            confidence=confidence,
            tokens_per_second=token_rate,
            requests_per_second=request_rate,
            capped=capped,
        )

    def _token_rate(self) -> Optional[float]:
        return self._token_rate_ema

    def _request_rate(self) -> Optional[float]:
        return self._req_rate_ema

    def _confidence(self, remaining_items: int) -> str:
        if self._samples < 10:
            return "low"
        if self._samples < 30 or (
            self._mad is not None
            and self._median is not None
            and self._mad > 0.4 * self._median
        ):
            return "medium"
        if remaining_items < 10:
            return "high"
        return "high"

    @property
    def sample_count(self) -> int:
        return self._samples


__all__ = ["SmartEta", "EtaEstimate"]
