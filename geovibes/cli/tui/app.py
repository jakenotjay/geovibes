"""GeoVibes TUI application."""

from typing import Dict

from textual.app import App, ComposeResult
from textual.binding import Binding

from geovibes.cli.tui.screens.queue import QueueScreen
from geovibes.cli.tui.screens.review import ReviewScreen


class GeoVibesTUI(App):
    """Interactive TUI for reviewing detections and labelling embeddings."""

    CSS_PATH = "styles.tcss"
    TITLE = "GeoVibes"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("1", "switch_screen('queue')", "Jobs", show=True),
        Binding("2", "switch_screen('review')", "Review", show=True),
    ]

    SCREENS = {
        "queue": QueueScreen,
        "review": ReviewScreen,
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.project_dir = config["_project_dir"]

    def on_mount(self) -> None:
        self.sub_title = self.config.get("name", "")
        self.push_screen("queue")

    def action_switch_screen(self, screen_name: str) -> None:
        self._finish_active_review()
        self.switch_screen(screen_name)

    def action_quit(self) -> None:
        self._finish_active_review()
        self.exit()

    def _finish_active_review(self) -> None:
        """Finish any active review job before leaving the screen."""
        from geovibes.cli.tui.screens.review import ReviewScreen
        if isinstance(self.screen, ReviewScreen):
            self.screen._finish_review_job()
