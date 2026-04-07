"""Search screen — FAISS similarity search + labelling."""

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, WorkerState

from geovibes.cli.search import (
    connect_db,
    load_faiss_index,
    compute_query_vector,
    search_faiss,
    search_metadata,
    fetch_embeddings,
)

try:
    from textual_image.widget import Image as TImage

    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False


class ResultTile(Static):
    """A single search result tile with score and label state."""

    def __init__(self, result_id: int, score: float, rank: int, **kwargs):
        super().__init__(**kwargs)
        self.result_id = result_id
        self.score = score
        self.rank = rank
        self.label: Optional[str] = None


class SearchScreen(Screen):
    """FAISS similarity search with inline labelling."""

    BINDINGS = [
        Binding("enter", "run_search", "Search"),
        Binding("p", "label_positive", "Positive"),
        Binding("n", "label_negative", "Negative"),
        Binding("x", "clear_label", "Clear"),
        Binding("d", "toggle_sort", "Dissimilar"),
        Binding("right", "next_result", "Next", show=False),
        Binding("left", "prev_result", "Prev", show=False),
        Binding("e", "export_labels", "Export"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._conn = None
        self._index = None
        self._results: List[Dict] = []
        self._selected = 0
        self._pos_ids: List[int] = []
        self._neg_ids: List[int] = []
        self._cached_embeddings: Dict[int, np.ndarray] = {}
        self._sort_ascending = False
        self._n_neighbors = 1000

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="search-header")
        with Horizontal(id="search-container"):
            yield Static("[dim]Select a tile to begin labelling[/]", id="search-tile")
            yield Static("", id="search-meta")
        yield Static("", id="search-results-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_database()
        self._update_header()

    def _load_database(self) -> None:
        config = self.app.config
        project_dir = self.app.project_dir
        db_path = project_dir / config["database"]["path"]
        faiss_path = project_dir / config["database"]["faiss_path"]

        if not db_path.exists():
            return
        if not faiss_path.exists():
            return

        self._conn = connect_db(db_path)
        self._index = load_faiss_index(faiss_path)

    def _update_header(self) -> None:
        header = self.query_one("#search-header", Static)
        n_pos = len(self._pos_ids)
        n_neg = len(self._neg_ids)
        sort_label = "Dissimilar" if self._sort_ascending else "Similar"
        db_loaded = "ready" if self._conn and self._index else "[red]no database[/]"

        header.update(
            f"[bold]Search[/] | "
            f"[green]+{n_pos}[/] [red]-{n_neg}[/] | "
            f"Neighbors: {self._n_neighbors} | "
            f"Sort: {sort_label} | "
            f"DB: {db_loaded}"
        )

    def _update_results_bar(self) -> None:
        bar = self.query_one("#search-results-bar", Static)
        if not self._results:
            bar.update("[dim]No results. Press Enter to search after labelling points.[/]")
            return

        total = len(self._results)
        pos = self._selected + 1

        parts = []
        for i, r in enumerate(self._results[:20]):
            rid = r["id"]
            score = r.get("score", 0)
            marker = ""
            if rid in self._pos_ids:
                marker = "[green]+[/]"
            elif rid in self._neg_ids:
                marker = "[red]-[/]"

            if i == self._selected:
                parts.append(f"[bold reverse] #{i+1} {score:.2f}{marker} [/]")
            else:
                parts.append(f" #{i+1} {score:.2f}{marker} ")

        bar.update(
            " ".join(parts) + f"  ({pos}/{total})"
        )

    def _show_selected(self) -> None:
        if not self._results or self._selected >= len(self._results):
            return

        result = self._results[self._selected]
        rid = result["id"]
        score = result.get("score", 0)
        lat = result.get("lat", 0)
        lon = result.get("lon", 0)

        label = "none"
        if rid in self._pos_ids:
            label = "[green]positive[/]"
        elif rid in self._neg_ids:
            label = "[red]negative[/]"

        meta = self.query_one("#search-meta", Static)
        meta.update(
            f"[bold]Result #{self._selected + 1}/{len(self._results)}[/]\n"
            f"\n"
            f"[dim]ID:[/] {rid}\n"
            f"[dim]Score:[/] {score:.4f}\n"
            f"[dim]Lat:[/] {lat:.4f}\n"
            f"[dim]Lon:[/] {lon:.4f}\n"
            f"[dim]Label:[/] {label}\n"
        )

        self._update_results_bar()

        tile_panel = self.query_one("#search-tile", Static)
        tile_panel.update(f"[dim]Loading tile at {lat:.4f}, {lon:.4f}...[/]")
        self.run_worker(
            self._fetch_tile_async(lat, lon),
            name="search_tile_fetch",
            exclusive=True,
        )

    async def _fetch_tile_async(self, lat: float, lon: float) -> bytes:
        from geovibes.ui.xyz import get_map_image
        return get_map_image(source="GOOGLE_HYBRID", lon=lon, lat=lat, zoom=16)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "search_tile_fetch":
            return
        if event.state == WorkerState.SUCCESS:
            tile_bytes = event.worker.result
            tile_panel = self.query_one("#search-tile", Static)
            if HAS_IMAGE and tile_bytes:
                from PIL import Image as PILImage
                img = PILImage.open(BytesIO(tile_bytes))
                try:
                    tile_panel.remove_children()
                except Exception:
                    pass
                new_widget = TImage(img)
                self.call_after_refresh(lambda: tile_panel.mount(new_widget))
            elif tile_bytes:
                tile_panel.update(f"[green]Tile loaded[/] ({len(tile_bytes)} bytes)")

    def action_run_search(self) -> None:
        if not self._conn or not self._index:
            return

        pos_embeds = [self._cached_embeddings[pid] for pid in self._pos_ids if pid in self._cached_embeddings]
        neg_embeds = [self._cached_embeddings[nid] for nid in self._neg_ids if nid in self._cached_embeddings]

        query = compute_query_vector(pos_embeds, neg_embeds)
        if query is None:
            bar = self.query_one("#search-results-bar", Static)
            bar.update("[yellow]Label at least one positive point first[/]")
            return

        distances, ids = search_faiss(self._index, query, self._n_neighbors)

        labeled_ids = set(self._pos_ids) | set(self._neg_ids)
        valid = [(d, int(i)) for d, i in zip(distances, ids) if i >= 0 and int(i) not in labeled_ids]

        if self._sort_ascending:
            valid.sort(key=lambda x: -x[0])
        else:
            valid.sort(key=lambda x: x[0])

        faiss_ids = [i for _, i in valid]
        meta_df = search_metadata(self._conn, faiss_ids)

        if meta_df.empty:
            self._results = []
        else:
            id_to_meta = {row["id"]: row for _, row in meta_df.iterrows()}
            self._results = []
            for dist, fid in valid:
                if fid in id_to_meta:
                    row = id_to_meta[fid]
                    self._results.append({
                        "id": int(fid),
                        "score": float(dist),
                        "lat": float(row.get("lat", 0)),
                        "lon": float(row.get("lon", 0)),
                    })

        self._selected = 0
        self._update_header()
        self._show_selected()

    def _label_current(self, label: str) -> None:
        if not self._results or self._selected >= len(self._results):
            return

        result = self._results[self._selected]
        rid = result["id"]

        if rid in self._pos_ids:
            self._pos_ids.remove(rid)
        if rid in self._neg_ids:
            self._neg_ids.remove(rid)

        if label == "positive":
            self._pos_ids.append(rid)
        elif label == "negative":
            self._neg_ids.append(rid)

        if rid not in self._cached_embeddings and self._conn:
            df = fetch_embeddings(self._conn, [rid])
            if not df.empty:
                emb = df.iloc[0]["embedding"]
                self._cached_embeddings[rid] = np.array(emb, dtype=np.float32)

        self._update_header()
        self._show_selected()

    def action_label_positive(self) -> None:
        self._label_current("positive")

    def action_label_negative(self) -> None:
        self._label_current("negative")

    def action_clear_label(self) -> None:
        self._label_current("clear")

    def action_toggle_sort(self) -> None:
        self._sort_ascending = not self._sort_ascending
        if self._results:
            if self._sort_ascending:
                self._results.sort(key=lambda x: -x["score"])
            else:
                self._results.sort(key=lambda x: x["score"])
            self._selected = 0
        self._update_header()
        self._show_selected()

    def action_next_result(self) -> None:
        if self._selected < len(self._results) - 1:
            self._selected += 1
            self._show_selected()

    def action_prev_result(self) -> None:
        if self._selected > 0:
            self._selected -= 1
            self._show_selected()

    def action_export_labels(self) -> None:
        if not self._pos_ids and not self._neg_ids:
            return

        from datetime import datetime, timezone
        import pandas as pd
        from geovibes.cli.ledger import save_labels
        from geovibes.cli.project import load_project, save_project

        project_dir = self.app.project_dir
        config = self.app.config
        iteration = config.get("iteration", 0) + 1

        rows = []
        for pid in self._pos_ids:
            rows.append({"id": pid, "geometry": None, "label": 1, "source": "manual", "iteration": iteration, "created_at": datetime.now(timezone.utc)})
        for nid in self._neg_ids:
            rows.append({"id": nid, "geometry": None, "label": 0, "source": "hard_negative", "iteration": iteration, "created_at": datetime.now(timezone.utc)})

        df = pd.DataFrame(rows)
        path = save_labels(project_dir, df, iteration)

        config["iteration"] = iteration
        save_project(config)

        bar = self.query_one("#search-results-bar", Static)
        bar.update(f"[green]Saved {len(rows)} labels to {path.name}[/]")

    def on_screen_resume(self) -> None:
        self._update_header()
        if self._results:
            self._show_selected()
