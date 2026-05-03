from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .columns import ProjectedCostColumn, SmartEtaColumn, TokenRateColumn
from .metrics import Aggregator, CostProjection, DashboardSnapshot


class Dashboard:
    def __init__(
        self,
        aggregator: Aggregator,
        *,
        console: Optional[Console] = None,
        refresh_hz: float = 5.0,
        compact: bool = False,
        recent_rows: int = 20,
    ) -> None:
        self.aggregator = aggregator
        self.console = console or Console()
        self.refresh_hz = max(1.0, refresh_hz)
        self.compact = compact
        self.recent_rows = max(5, recent_rows)

        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        self._layout: Optional[Layout] = None
        self._live: Optional[Live] = None

        self._compact_mode = compact
        self._show_cost_panel = True
        self._show_tokens_panel = True
        self._show_recent = not compact

    # Lifecycle ----------------------------------------------------------------
    def start(self) -> None:
        if self._live is not None:
            return

        term_size = self.console.size
        term_height = getattr(term_size, "height", 0) or 0
        auto_compact = not self.compact and term_height and term_height < 34
        self._compact_mode = self.compact or auto_compact

        if self._compact_mode:
            self._show_cost_panel = False
            self._show_tokens_panel = False
            self._show_recent = False
        else:
            self._show_cost_panel = term_height == 0 or term_height >= 20
            self._show_tokens_panel = term_height == 0 or term_height >= 24
            self._show_recent = term_height == 0 or term_height >= 32

        self._progress = Progress(
            TextColumn("{task.description}", style="bold"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}", style="cyan"),
            TokenRateColumn(),
            SmartEtaColumn(),
            ProjectedCostColumn(),
            console=self.console,
            expand=True,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "Progress", total=self.aggregator.total_items
        )

        # Reduce minimum sizes to make dashboard more compact
        min_top = 3 if self._compact_mode else 4
        min_mid = 2
        min_bottom = 1
        # Don't set a fixed size on the root layout - let it adapt to terminal space
        layout = Layout(name="root")

        layout.split_column(
            Layout(name="row_top", minimum_size=min_top, ratio=3),
            Layout(name="row_mid", minimum_size=min_mid, ratio=2),
            Layout(name="row_bottom", minimum_size=min_bottom, ratio=1),
        )

        top_children = [Layout(name="overview")]
        if self._show_cost_panel:
            top_children.append(Layout(name="cost"))
        if self._show_tokens_panel:
            top_children.append(Layout(name="tokens"))
        layout["row_top"].split_row(*top_children)

        mid_children = [Layout(name="rates")]
        if self._show_recent:
            mid_children.append(Layout(name="recent"))
        layout["row_mid"].split_row(*mid_children)

        layout["row_bottom"].update(self._progress)

        self._layout = layout
        self._live = Live(
            layout,
            console=self.console,
            refresh_per_second=self.refresh_hz,
            transient=False,
            auto_refresh=False,
            vertical_overflow="ellipsis",
            screen=True,
        )
        self._live.__enter__()

    def stop(self) -> None:
        if self._progress and self._task_id is not None:
            self._progress.stop()
        if self._live is not None:
            self._live.__exit__(None, None, None)
            self._live = None
        self._progress = None
        self._task_id = None
        self._layout = None

    # Update -------------------------------------------------------------------
    def update(self, snapshot: DashboardSnapshot) -> None:
        if (
            self._live is None
            or self._progress is None
            or self._task_id is None
            or self._layout is None
        ):
            raise RuntimeError("Dashboard.update called before start()")

        eta_seconds = snapshot.eta_seconds
        token_rate = snapshot.tokens_per_second
        requests_per_second = snapshot.requests_per_second
        cost_projection = snapshot.cost_projection

        self._progress.update(
            self._task_id,
            completed=snapshot.completed_items,
            total=max(snapshot.total_items, 1),
            token_rate=token_rate,
            cost_known=cost_projection.known_cost if cost_projection else 0.0,
            cost_projected=cost_projection.projected_total if cost_projection else None,
            eta_seconds=eta_seconds,
            eta_confidence=snapshot.eta_confidence,
            eta_capped=snapshot.eta_capped,
            refresh=True,
        )

        self._layout["overview"].update(
            self._render_overview(snapshot, cost_projection)
        )
        if self._show_cost_panel:
            self._layout["cost"].update(self._render_cost(snapshot))
        if self._show_tokens_panel:
            self._layout["tokens"].update(self._render_tokens(snapshot))
        self._layout["rates"].update(
            self._render_rates(snapshot, token_rate, requests_per_second)
        )
        if self._show_recent:
            self._layout["recent"].update(self._render_recent(snapshot))

        self._live.refresh()

    # Panels -------------------------------------------------------------------
    def _render_overview(
        self, snapshot: DashboardSnapshot, cost_projection: Optional[CostProjection]
    ) -> Panel:
        table = Table.grid(expand=True)
        pct = (
            (snapshot.completed_items / snapshot.total_items * 100.0)
            if snapshot.total_items
            else 0.0
        )

        # Format elapsed time
        elapsed_str = self._format_elapsed_time(snapshot.elapsed_time_seconds)

        table.add_row(Text(f"Model: {snapshot.model_id}", style="yellow bold"))
        table.add_row(
            Text(
                f"Done {snapshot.completed_items}/{snapshot.total_items} ({pct:.1f}%)",
                style="bold",
            ),
            Text(f"Elapsed: {elapsed_str}", style="green"),
        )
        table.add_row(
            Text(f"In-flight: {snapshot.in_flight}", style="cyan"),
            Text(f"Concurrency: {snapshot.worker_count}", style="cyan"),
        )
        table.add_row(
            Text(f"ETA: {self._format_eta(snapshot)}", style="magenta"),
        )
        if not self._show_cost_panel and cost_projection is not None:
            projected = cost_projection.projected_total
            proj_str = f"${projected:,.2f}" if projected is not None else "n/a"
            table.add_row(
                Text(
                    f"Known cost: ${cost_projection.known_cost:,.2f}", style="magenta"
                ),
                Text(f"Projected: {proj_str}", style="magenta"),
            )
        if not self._show_tokens_panel:
            tokens = snapshot.total_tokens_total
            mean_tokens = snapshot.mean_tokens_per_item
            mean_str = f"{mean_tokens:.0f}" if mean_tokens else "--"
            table.add_row(
                Text(f"Tokens: {tokens:,}", style="cyan"),
                Text(f"Mean/Q: {mean_str}", style="cyan"),
            )
        return Panel(table, title="Overview", padding=(0, 1))

    def _format_elapsed_time(self, elapsed_seconds: Optional[float]) -> str:
        if elapsed_seconds is None:
            return "--:--:--"
        elapsed_seconds = int(round(elapsed_seconds))
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def _render_cost(self, snapshot: DashboardSnapshot) -> Panel:
        table = Table.grid(expand=True)
        projection = snapshot.cost_projection
        known = projection.known_cost if projection else 0.0
        table.add_row(Text(f"Known: ${known:,.2f}", style="magenta"))
        if projection and projection.projected_total is not None:
            low = projection.projected_low or projection.projected_total
            high = projection.projected_high or projection.projected_total
            table.add_row(
                Text(f"Projected: ${low:,.2f} – ${high:,.2f}", style="magenta")
            )
            if projection.average_cost_per_item is not None:
                table.add_row(
                    Text(
                        f"/Q mean: ${projection.average_cost_per_item:.3f}",
                        style="magenta",
                    )
                )
            table.add_row(
                Text(
                    f"Samples: {projection.samples} ({projection.confidence})",
                    style="dim",
                )
            )
        else:
            table.add_row(Text("Projection warming up", style="dim"))
        return Panel(table, title="Cost", padding=(0, 1))

    def _render_tokens(self, snapshot: DashboardSnapshot) -> Panel:
        table = Table.grid(expand=True)
        table.add_row(
            Text(f"Prompt: {snapshot.prompt_tokens_total:,}", style="cyan"),
            Text(f"Completion: {snapshot.completion_tokens_total:,}", style="cyan"),
        )
        explicit_reasoning_total = snapshot.explicit_reasoning_tokens_total
        reasoning_label = f"Reasoning: {snapshot.reasoning_tokens_total:,}"
        if explicit_reasoning_total:
            reasoning_label += f" (explicit {explicit_reasoning_total:,})"
        table.add_row(
            Text(f"Total: {snapshot.total_tokens_total:,}", style="bold cyan"),
            Text(reasoning_label, style="cyan"),
        )
        table.add_row(
            Text(f"Cached: {snapshot.cached_prompt_tokens_total:,}", style="cyan"),
            Text(
                "p50/p90: "
                + (
                    f"{int(snapshot.median_tokens_per_item)} / {int(snapshot.p90_tokens_per_item)}"
                    if snapshot.median_tokens_per_item is not None
                    and snapshot.p90_tokens_per_item is not None
                    else "--"
                ),
                style="dim",
            ),
        )
        return Panel(table, title="Tokens", padding=(0, 1))

    def _render_rates(
        self,
        snapshot: DashboardSnapshot,
        token_rate: Optional[float],
        request_rate: Optional[float],
    ) -> Panel:
        table = Table.grid(expand=True)
        tr = f"{token_rate:,.0f} toks/s" if token_rate else "-- toks/s"
        rr = f"{request_rate:.2f} req/s" if request_rate else "-- req/s"
        table.add_row(Text(tr, style="green"))
        table.add_row(Text(rr, style="green"))
        if snapshot.min_request_interval:
            rl = (
                1.0 / snapshot.min_request_interval
                if snapshot.min_request_interval > 0
                else None
            )
            if rl:
                table.add_row(
                    Text(
                        f"Ceiling: {rl * snapshot.worker_count:.2f} req/s",
                        style="yellow",
                    )
                )
        table.add_row(Text(f"Throttles: {snapshot.throttles}", style="yellow"))
        if snapshot.last_status is not None:
            table.add_row(Text(f"Last status: {snapshot.last_status}", style="dim"))
        if snapshot.mean_latency_ms is not None:
            table.add_row(
                Text(f"Mean latency: {snapshot.mean_latency_ms:.0f} ms", style="cyan")
            )
        if snapshot.mean_attempts is not None and snapshot.mean_attempts > 1.0:
            table.add_row(
                Text(f"Avg attempts: {snapshot.mean_attempts:.2f}", style="cyan")
            )
        return Panel(table, title="Rate & Health", padding=(0, 1))

    def _render_recent(self, snapshot: DashboardSnapshot) -> Panel:
        table = Table(expand=True)
        table.add_column("id", style="bold")
        table.add_column("ms", justify="right")
        table.add_column("pt", justify="right")
        table.add_column("ct", justify="right")
        table.add_column("tot", justify="right")
        table.add_column("rt", justify="right")
        table.add_column("cost", justify="right")
        table.add_column("pred", justify="center")
        table.add_column("ok", justify="center")

        for item in list(snapshot.recent_items)[: self.recent_rows]:
            cost_val = (
                item.cost_usd_known
                if item.cost_usd_known is not None
                else item.cost_usd_calc
            )
            cost_s = f"${cost_val:.3f}" if cost_val is not None else "--"
            if item.reasoning_tokens is not None:
                if item.explicit_reasoning_tokens is not None:
                    reasoning_s = (
                        f"{item.reasoning_tokens}/{item.explicit_reasoning_tokens}"
                    )
                else:
                    reasoning_s = f"{item.reasoning_tokens}"
            else:
                reasoning_s = "--"
            table.add_row(
                str(item.row_id or "--"),
                f"{int(item.latency_ms)}" if item.latency_ms is not None else "--",
                f"{item.prompt_tokens}" if item.prompt_tokens is not None else "--",
                f"{item.completion_tokens}"
                if item.completion_tokens is not None
                else "--",
                f"{item.total_tokens}" if item.total_tokens is not None else "--",
                reasoning_s,
                cost_s,
                item.predicted or "--",
                "✓" if item.correct else ("×" if item.correct is False else ""),
            )
        return Panel(table, title="Recent Items", padding=(0, 0))

    # Helpers ------------------------------------------------------------------
    def _format_eta(self, snapshot: DashboardSnapshot) -> str:
        seconds = snapshot.eta_seconds
        if seconds is None:
            return "--"
        hours, rem = divmod(int(seconds), 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d} ({snapshot.eta_confidence})"


__all__ = ["Dashboard"]
