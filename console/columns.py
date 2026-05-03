from __future__ import annotations

import math
from typing import Optional

from rich.progress import ProgressColumn, Task
from rich.text import Text


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 99:
        return ">99h"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TokenRateColumn(ProgressColumn):
    header = "Tokens/s"

    def render(self, task: Task) -> Text:
        rate = task.fields.get("token_rate")
        if not rate or rate <= 0:
            return Text("-- toks/s", style="progress.remaining")
        if rate >= 1000:
            value = f"{rate / 1000:.1f}k"
        else:
            value = f"{rate:.0f}"
        style = "green" if rate >= 500 else "cyan"
        return Text(f"{value} toks/s", style=style)


class ProjectedCostColumn(ProgressColumn):
    header = "Cost"

    def render(self, task: Task) -> Text:
        known = task.fields.get("cost_known")
        projected = task.fields.get("cost_projected")
        if known is None and projected is None:
            return Text("$0.00", style="progress.remaining")
        known_s = f"${known:,.2f}" if known is not None else "$0.00"
        if projected is None or projected <= known:
            return Text(known_s, style="magenta")
        projected_s = f"${projected:,.2f}"
        return Text(f"{known_s} → {projected_s}", style="magenta")


class SmartEtaColumn(ProgressColumn):
    header = "ETA"

    def render(self, task: Task) -> Text:
        eta_seconds = task.fields.get("eta_seconds")
        confidence = task.fields.get("eta_confidence", "low")
        capped = task.fields.get("eta_capped", False)
        style = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
        }.get(confidence, "red")
        if capped:
            style = "yellow"
        eta_str = _format_seconds(eta_seconds)
        suffix = {
            "high": "↑",
            "medium": "≈",
            "low": "?",
        }.get(confidence, "?")
        if capped:
            suffix = "⚑"
        return Text(f"{eta_str} {suffix}", style=style)


__all__ = [
    "TokenRateColumn",
    "ProjectedCostColumn",
    "SmartEtaColumn",
]
