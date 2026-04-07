"""Review screen — one-at-a-time detection review with satellite tile."""

import asyncio
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import pandas as pd

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, WorkerState

from geovibes.cli.ledger import (
    add_comment,
    load_comments,
    load_reviews,
    save_reviews,
    update_review,
    create_job,
    update_job,
)

HAS_IMAGE = False


GOOGLE_HYBRID_TEMPLATE = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"


class ReviewScreen(Screen):
    """One-at-a-time detection review with satellite imagery."""

    BINDINGS = [
        Binding("a", "accept", "Accept", priority=True),
        Binding("r", "reject", "Reject", priority=True),
        Binding("s", "skip", "Skip", priority=True),
        Binding("c", "comment", "Comment", priority=True),
        Binding("u", "undo", "Undo", priority=True),
        Binding("right", "next_detection", "Next", show=False, priority=True),
        Binding("left", "prev_detection", "Prev", show=False, priority=True),
        Binding("p", "filter_pending", "Pending only", priority=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._index = 0
        self._detection_ids: List[int] = []
        self._reviews = None
        self._review_job_id: Optional[int] = None
        self._undo_stack: List[dict] = []
        self._filter_pending = True
        self._reviewed_count = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="review-container"):
            yield Static("[dim]Loading tile...[/]", id="tile-panel")
            yield Static("", id="meta-panel")
        yield Static("", id="progress-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_detections()
        self._start_review_job()
        self._show_current()

    def _load_detections(self) -> None:
        project_dir = self.app.project_dir
        self._reviews = load_reviews(project_dir)

        if self._reviews.empty:
            self._detection_ids = []
            return

        df = self._reviews.sort_values("score", ascending=False)
        if self._filter_pending:
            df = df[df["status"] == "pending"]

        self._detection_ids = df["detection_id"].tolist()
        self._index = 0

    def _start_review_job(self) -> None:
        if self._reviews is None or self._reviews.empty:
            return
        project_dir = self.app.project_dir
        iteration = self.app.config.get("iteration", 0)
        self._review_job_id = create_job(
            project_dir,
            job_type="review:human",
            iteration=iteration,
            reviewer="human",
        )

    def _current_detection(self) -> Optional[dict]:
        if not self._detection_ids or self._index >= len(self._detection_ids):
            return None
        det_id = self._detection_ids[self._index]
        row = self._reviews[self._reviews["detection_id"] == det_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def _show_current(self) -> None:
        det = self._current_detection()
        meta_panel = self.query_one("#meta-panel", Static)
        tile_panel = self.query_one("#tile-panel", Static)
        progress = self.query_one("#progress-bar", Static)

        if det is None:
            meta_panel.update("[bold]No detections to review[/]")
            tile_panel.update("[dim]No detections loaded[/]")
            progress.update("")
            return

        total = len(self._detection_ids)
        pos = self._index + 1

        score = det.get("score", 0)
        score_color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
        status = det.get("status", "pending")
        status_colors = {"pending": "yellow", "accepted": "green", "rejected": "red", "skipped": "dim"}
        status_color = status_colors.get(status, "white")

        cluster_id = det.get("cluster_id")
        if pd.notna(cluster_id) and int(cluster_id) >= 0:
            cluster_str = f"C-{int(cluster_id):03d}"
        elif pd.notna(cluster_id) and int(cluster_id) == -1:
            cluster_str = "noise"
        else:
            cluster_str = "—"

        lat, lon = self._geometry_to_latlon(det.get("geometry"))
        lat_str = f"{lat:.4f}" if lat else "—"
        lon_str = f"{lon:.4f}" if lon else "—"

        comments = load_comments(self.app.project_dir, detection_id=det["detection_id"])
        comment_lines = ""
        if not comments.empty:
            for _, c in comments.iterrows():
                comment_lines += f"  [{c['reviewer']}] {c['comment']}\n"
        else:
            comment_lines = "  [dim](none)[/]"

        meta_panel.update(
            f"[bold]Detection #{pos}/{total}[/]\n"
            f"\n"
            f"[dim]ID:[/] {int(det['detection_id'])}\n"
            f"[dim]Score:[/] [{score_color}]{score:.3f}[/]\n"
            f"[dim]Cluster:[/] {cluster_str}\n"
            f"[dim]Lat:[/] {lat_str}\n"
            f"[dim]Lon:[/] {lon_str}\n"
            f"[dim]Status:[/] [{status_color}]{status}[/]\n"
            f"\n"
            f"[bold]Comments:[/]\n{comment_lines}"
        )

        filled = int((pos / total) * 30) if total > 0 else 0
        bar = "█" * filled + "░" * (30 - filled)

        reviewed = self._reviewed_count
        progress.update(
            f" {bar} {pos}/{total}  "
            f"[green]Accepted: {self._count_status('accepted')}[/]  "
            f"[red]Rejected: {self._count_status('rejected')}[/]  "
            f"Session: {reviewed}"
        )

        self._fetch_tile(lat, lon)

    def _count_status(self, status: str) -> int:
        if self._reviews is None or self._reviews.empty:
            return 0
        return int((self._reviews["status"] == status).sum())

    def _geometry_to_latlon(self, geom_bytes) -> tuple:
        if geom_bytes is None:
            return (None, None)
        try:
            import shapely.wkb
            point = shapely.wkb.loads(geom_bytes)
            return (point.y, point.x)
        except Exception:
            return (None, None)

    def _fetch_tile(self, lat: Optional[float], lon: Optional[float]) -> None:
        tile_panel = self.query_one("#tile-panel", Static)
        if lat is None or lon is None:
            tile_panel.update("[dim]No coordinates[/]")
            return

        tile_panel.update(f"[dim]Loading tile at {lat:.4f}, {lon:.4f}...[/]")
        self._run_tile_worker(lat, lon)

    def _run_tile_worker(self, lat: float, lon: float) -> None:
        self.run_worker(
            self._fetch_tile_async(lat, lon),
            name="tile_fetch",
            exclusive=True,
        )

    async def _fetch_tile_async(self, lat: float, lon: float) -> bytes:
        from geovibes.ui.xyz import get_map_image
        return await asyncio.to_thread(
            get_map_image,
            source="GOOGLE_HYBRID",
            lon=lon,
            lat=lat,
            zoom=16,
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "tile_fetch":
            return
        if event.state == WorkerState.SUCCESS:
            tile_bytes = event.worker.result
            tile_panel = self.query_one("#tile-panel", Static)
            if HAS_IMAGE and tile_bytes:
                from PIL import Image as PILImage

                img = PILImage.open(BytesIO(tile_bytes))
                tile_panel.remove_children()
                new_widget = TImage(img)
                self.call_after_refresh(lambda w=new_widget, p=tile_panel: p.mount(w))
            elif tile_bytes:
                tile_panel.update(f"[green]Tile loaded[/] ({len(tile_bytes)} bytes)")
            else:
                tile_panel.update("[red]Failed to load tile[/]")
        elif event.state == WorkerState.ERROR:
            tile_panel = self.query_one("#tile-panel", Static)
            tile_panel.update(f"[red]Tile error: {event.worker.error}[/]")

    def _apply_verdict(self, status: str) -> None:
        det = self._current_detection()
        if det is None:
            return

        self._undo_stack.append({
            "detection_id": det["detection_id"],
            "old_status": det["status"],
            "index": self._index,
        })

        project_dir = self.app.project_dir
        update_review(
            project_dir,
            detection_id=int(det["detection_id"]),
            status=status,
            reviewer="human",
            review_job_id=self._review_job_id,
        )

        self._reviews = load_reviews(project_dir)
        self._reviewed_count += 1

        if self._filter_pending:
            self._detection_ids = (
                self._reviews[self._reviews["status"] == "pending"]
                .sort_values("score", ascending=False)["detection_id"]
                .tolist()
            )
            self._index = min(self._index, max(0, len(self._detection_ids) - 1))
        else:
            self._advance()

        self._show_current()

    def _advance(self) -> None:
        if self._index < len(self._detection_ids) - 1:
            self._index += 1

    def action_accept(self) -> None:
        self._apply_verdict("accepted")

    def action_reject(self) -> None:
        self._apply_verdict("rejected")

    def action_skip(self) -> None:
        self._apply_verdict("skipped")

    def action_undo(self) -> None:
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        project_dir = self.app.project_dir
        update_review(
            project_dir,
            detection_id=entry["detection_id"],
            status=entry["old_status"],
            reviewer="human",
            review_job_id=self._review_job_id,
        )
        self._reviews = load_reviews(project_dir)
        self._reviewed_count = max(0, self._reviewed_count - 1)

        if self._filter_pending:
            self._detection_ids = (
                self._reviews[self._reviews["status"] == "pending"]
                .sort_values("score", ascending=False)["detection_id"]
                .tolist()
            )
        self._index = min(entry["index"], max(0, len(self._detection_ids) - 1))
        self._show_current()

    def action_next_detection(self) -> None:
        if self._index < len(self._detection_ids) - 1:
            self._index += 1
            self._show_current()

    def action_prev_detection(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def action_filter_pending(self) -> None:
        self._filter_pending = not self._filter_pending
        self._load_detections()
        self._show_current()

    def action_comment(self) -> None:
        pass

    def on_screen_resume(self) -> None:
        self._reviews = load_reviews(self.app.project_dir)
        self._load_detections()
        self._show_current()
