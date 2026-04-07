"""Queue screen — roborev-style job ledger."""

from datetime import datetime, timezone
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from geovibes.cli.ledger import load_jobs, load_reviews


def _format_elapsed(started_at, finished_at) -> str:
    if started_at is None:
        return ""
    start = started_at
    end = finished_at if finished_at is not None else datetime.now(timezone.utc)
    if hasattr(start, "to_pydatetime"):
        start = start.to_pydatetime()
    if hasattr(end, "to_pydatetime"):
        end = end.to_pydatetime()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    delta = end - start
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60}s"
    return f"{secs // 3600}h{(secs % 3600) // 60}m"


def _status_markup(status: str) -> str:
    colors = {
        "done": "[green]done[/]",
        "running": "[yellow]running[/]",
        "failed": "[red]failed[/]",
        "queued": "[dim]queued[/]",
    }
    return colors.get(status, status)


class QueueScreen(Screen):
    """Job queue showing all operations performed on the project."""

    BINDINGS = [
        Binding("r", "refresh_table", "Refresh"),
        Binding("f", "filter", "Filter"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="queue-header")
        yield DataTable(id="jobs-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Job", "Type", "Iter", "Status", "Elapsed", "Summary")
        self._load_data()

    def _load_data(self) -> None:
        project_dir = self.app.project_dir
        jobs = load_jobs(project_dir)
        reviews = load_reviews(project_dir)

        pending = len(reviews[reviews["status"] == "pending"]) if not reviews.empty else 0
        accepted = len(reviews[reviews["status"] == "accepted"]) if not reviews.empty else 0
        rejected = len(reviews[reviews["status"] == "rejected"]) if not reviews.empty else 0
        total_det = len(reviews)

        header = self.query_one("#queue-header", Static)
        header.update(
            f"[bold]{self.app.config.get('name', '')}[/] — "
            f"Jobs: {len(jobs)} | "
            f"Detections: {total_det} "
            f"([green]A:{accepted}[/] [red]R:{rejected}[/] [yellow]P:{pending}[/])"
        )

        table = self.query_one("#jobs-table", DataTable)
        table.clear()

        for _, row in jobs.iloc[::-1].iterrows():
            table.add_row(
                str(int(row["job_id"])),
                row["job_type"],
                str(int(row["iteration"])) if row["iteration"] is not None else "",
                _status_markup(row["status"]),
                _format_elapsed(row.get("started_at"), row.get("finished_at")),
                str(row.get("summary", "") or ""),
            )

    def action_refresh_table(self) -> None:
        self._load_data()

    def action_filter(self) -> None:
        pass
