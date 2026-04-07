"""Review screen — one-at-a-time detection review with satellite tile."""

import io
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

# Import tile fetcher at module level to avoid triggering warnings inside threads
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _stderr = sys.stderr
    sys.stderr = io.StringIO()
    from geovibes.ui.xyz import get_map_image as _get_map_image
    sys.stderr = _stderr

from geovibes.cli.ledger import (
    add_comment,
    load_comments,
    load_reviews,
    save_reviews,
    update_review,
    create_job,
    update_job,
)

from itertools import count
from random import randint

from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.segment import Segment
from rich.style import Style

from textual_image.renderable.tgp import (
    _NUMBER_TO_DIACRITIC,
    _PLACEHOLDER,
    _TGP_MESSAGE_START,
    _TGP_MESSAGE_END,
)

_tgp_id_counter = count(randint(1, 2**32))


def _fetch_tile_grid(lat: float, lon: float, zoom: int = 18, grid: int = 3) -> bytes:
    """Fetch a grid x grid mosaic of tiles centered on lat/lon, return as PNG bytes."""
    from geovibes.ui.xyz import deg2num, _fetch_tile_bytes, _xyz_sources
    from PIL import Image as PILImage

    template = _xyz_sources()["GOOGLE_HYBRID"]
    cx, cy = deg2num(lat, lon, zoom)
    half = grid // 2
    mosaic = PILImage.new("RGB", (grid * 256, grid * 256))
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            tile_bytes = _fetch_tile_bytes("GOOGLE_HYBRID", template, zoom, cx + dx, cy + dy)
            tile_img = PILImage.open(io.BytesIO(tile_bytes))
            mosaic.paste(tile_img, ((dx + half) * 256, (dy + half) * 256))
    buf = io.BytesIO()
    mosaic.save(buf, format="PNG")
    return buf.getvalue()


def _send_tgp_to_tty(*, payload: str | None = None, **kwargs: int | str | None) -> None:
    """Send a Kitty graphics protocol message directly to /dev/tty."""
    parts = [
        _TGP_MESSAGE_START,
        ",".join(f"{k}={v}" for k, v in kwargs.items() if v is not None),
        f";{payload}" if payload else "",
        _TGP_MESSAGE_END,
    ]
    sequence = "".join(parts)
    fd = os.open("/dev/tty", os.O_WRONLY)
    os.write(fd, sequence.encode())
    os.close(fd)


def _transmit_image(pil_image, cell_width, cell_height):
    """Transmit image to terminal via /dev/tty as PNG, return image_id."""
    import base64
    image_id = next(_tgp_id_counter)

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_data = base64.standard_b64encode(buf.getvalue()).decode("ascii")

    first = True
    while image_data:
        chunk, image_data = image_data[:4096], image_data[4096:]
        kwargs = {"m": 1 if image_data else 0, "q": 2, "payload": chunk}
        if first:
            kwargs.update(a="T", i=image_id, f=100, U=1, c=cell_width, r=cell_height)
            first = False
        _send_tgp_to_tty(**kwargs)

    return image_id


class TilePlaceholder:
    """Rich renderable that emits only Kitty unicode placeholder diacritics."""

    def __init__(self, image_id: int, width: int, height: int):
        self.image_id = image_id
        self.width = width
        self.height = height

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        style = Style(
            color=f"rgb({(self.image_id >> 16) & 255},{(self.image_id >> 8) & 255},{self.image_id & 255})"
        )
        id_char = _NUMBER_TO_DIACRITIC[(self.image_id >> 24) & 255]
        for r in range(self.height):
            line = "".join(
                f"{chr(_PLACEHOLDER)}{chr(_NUMBER_TO_DIACRITIC[r])}{chr(_NUMBER_TO_DIACRITIC[c])}{chr(id_char)}"
                for c in range(self.width)
            )
            yield Segment(line + "\n", style=style)

    def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
        return Measurement(self.width, self.width)


GOOGLE_HYBRID_TEMPLATE = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"


from textual.message import Message


class TileReady(Message):
    def __init__(self, content) -> None:
        super().__init__()
        self.content = content


class ReviewScreen(Screen):
    """One-at-a-time detection review with satellite imagery."""

    can_focus = True

    SORT_MODES = ["confident", "cluster", "uncertain"]

    BINDINGS = [
        Binding("a", "accept", "Accept", priority=True),
        Binding("r", "reject", "Reject", priority=True),
        Binding("s", "skip", "Skip", priority=True),
        Binding("c", "comment", "Comment", priority=True),
        Binding("u", "undo", "Undo", priority=True),
        Binding("right", "next_detection", "Next", show=False, priority=True),
        Binding("left", "prev_detection", "Prev", show=False, priority=True),
        Binding("p", "filter_pending", "Pending only", priority=True),
        Binding("o", "open_in_browser", "Open", priority=True),
        Binding("m", "cycle_sort_mode", "Mode", priority=True),
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
        self._sort_mode = "cluster"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="review-container"):
            yield Static("[dim]Loading tile...[/]", id="tile-panel")
            yield Static("", id="meta-panel")
        yield Static("", id="progress-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_detections()
        self._show_current(defer_tile=True)

    def _deferred_tile_fetch(self) -> None:
        """Fetch tile after a short delay so the screen's message loop is active."""
        det = self._current_detection()
        if det is None:
            return
        lat, lon = self._geometry_to_latlon(det.get("geometry"))
        self._fetch_tile(lat, lon)

    def on_unmount(self) -> None:
        self._finish_review_job()

    def _load_detections(self) -> None:
        project_dir = self.app.project_dir
        self._reviews = load_reviews(project_dir)

        if self._reviews.empty:
            self._detection_ids = []
            return

        df = self._reviews.copy()
        if self._filter_pending:
            df = df[df["status"] == "pending"]

        if self._sort_mode == "cluster":
            has_cluster = df["cluster_id"].notna() & (df["cluster_id"] >= 0)
            clustered = df[has_cluster].sort_values("score", ascending=False)
            reps = clustered.drop_duplicates(subset="cluster_id", keep="first")
            noise = df[~has_cluster].sort_values("score", ascending=False)
            df = pd.concat([reps, noise], ignore_index=True)
            df = df.sort_values("score", ascending=False)
        elif self._sort_mode == "uncertain":
            df["uncertainty"] = (df["score"] - 0.5).abs()
            df = df.sort_values("uncertainty", ascending=True)
        else:
            df = df.sort_values("score", ascending=False)

        self._detection_ids = df["detection_id"].tolist()
        self._index = 0

    def _ensure_review_job(self) -> None:
        """Create the review job lazily on first actual review."""
        if self._review_job_id is not None:
            return
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

    def _finish_review_job(self) -> None:
        """Mark the review job as done when leaving the screen."""
        if self._review_job_id is None:
            return
        project_dir = self.app.project_dir
        update_job(
            project_dir, self._review_job_id,
            status="done",
            summary=f"{self._reviewed_count} reviewed",
        )

    def _current_detection(self) -> Optional[dict]:
        if not self._detection_ids or self._index >= len(self._detection_ids):
            return None
        det_id = self._detection_ids[self._index]
        row = self._reviews[self._reviews["detection_id"] == det_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def _show_current(self, defer_tile: bool = False) -> None:
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

        mode_info = {
            "cluster": ("[cyan]Cluster rep[/]", "1 per cluster, by score"),
            "confident": ("[green]Confident[/]", "highest score first"),
            "uncertain": ("[yellow]Uncertain[/]", "near decision boundary"),
        }
        mode_label, mode_desc = mode_info[self._sort_mode]

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
            f"[dim]Mode:[/] {mode_label}\n"
            f"[dim italic]  {mode_desc} (m to change)[/]\n"
            f"\n"
            f"[bold]Comments:[/]\n{comment_lines}"
        )

        filled = int((pos / total) * 30) if total > 0 else 0
        bar = "█" * filled + "░" * (30 - filled)

        reviewed = self._reviewed_count
        mode_label = {"confident": "Confident", "cluster": "Cluster rep", "uncertain": "Uncertain"}
        progress.update(
            f" {bar} {pos}/{total}  "
            f"[green]A:{self._count_status('accepted')}[/]  "
            f"[red]R:{self._count_status('rejected')}[/]  "
            f"Session: {reviewed}  "
            f"[bold cyan]Mode: {mode_label[self._sort_mode]}[/] (m)"
        )

        if defer_tile:
            self.set_timer(0.3, self._deferred_tile_fetch)
        else:
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

        self._do_fetch_tile(lat, lon)

    def _do_fetch_tile(self, lat: float, lon: float) -> None:
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            tile_bytes = _fetch_tile_grid(lat, lon, zoom=18, grid=3)
        except Exception:
            sys.stderr = old_stderr
            self.query_one("#tile-panel", Static).update("[red]Failed to load tile[/]")
            return
        sys.stderr = old_stderr

        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(tile_bytes))
        panel = self.query_one("#tile-panel", Static)
        w = min(panel.size.width - 2, len(_NUMBER_TO_DIACRITIC)) if panel.size.width > 10 else 80
        h = min(panel.size.height - 2, len(_NUMBER_TO_DIACRITIC)) if panel.size.height > 10 else 40
        try:
            image_id = _transmit_image(img, w, h)
            renderable = TilePlaceholder(image_id, w, h)
        except Exception:
            from textual_image.renderable.halfcell import Image as HalfcellImage
            renderable = HalfcellImage(img, width=w, height=h)
        panel.update(renderable)



    def _apply_verdict(self, status: str) -> None:
        det = self._current_detection()
        if det is None:
            return

        self._ensure_review_job()

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

    def action_open_in_browser(self) -> None:
        det = self._current_detection()
        if det is None:
            return
        lat, lon = self._geometry_to_latlon(det.get("geometry"))
        if lat is None:
            return
        import webbrowser
        webbrowser.open(f"https://www.google.com/maps/@{lat},{lon},18z/data=!3m1!1e3")

    def action_filter_pending(self) -> None:
        self._filter_pending = not self._filter_pending
        self._load_detections()
        self._show_current()

    def action_cycle_sort_mode(self) -> None:
        idx = self.SORT_MODES.index(self._sort_mode)
        self._sort_mode = self.SORT_MODES[(idx + 1) % len(self.SORT_MODES)]
        self._load_detections()
        self._show_current()

    def action_comment(self) -> None:
        pass

    def on_screen_resume(self) -> None:
        self._reviews = load_reviews(self.app.project_dir)
        self._load_detections()
        self._show_current()
