"""Tests for TUI app and screens."""

import pytest
import pandas as pd
from pathlib import Path

from geovibes.cli.project import init_project
from geovibes.cli.ledger import create_job, update_job, save_reviews, load_reviews

pytestmark = pytest.mark.asyncio(loop_scope="function")


def _make_project_with_data(tmp_path: Path) -> dict:
    project_dir = init_project(
        name="test",
        bbox=[0, 0, 1, 1],
        years=[2024],
        base_dir=tmp_path,
    )

    job_id = create_job(project_dir, "init", iteration=0)
    update_job(project_dir, job_id, status="done", summary="project created")

    job_id = create_job(project_dir, "infer", iteration=1)
    update_job(project_dir, job_id, status="done", summary="500 detections")

    reviews = pd.DataFrame([
        {
            "detection_id": i,
            "embedding_id": i * 10,
            "geometry": None,
            "cluster_id": i // 5,
            "iteration": 1,
            "job_id": 2,
            "review_job_id": None,
            "score": 0.95 - (i * 0.01),
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
        }
        for i in range(20)
    ])
    save_reviews(project_dir, reviews)

    return {
        "name": "test",
        "bbox": [0, 0, 1, 1],
        "years": [2024],
        "resolution": 80,
        "iteration": 1,
        "tile_source": "GOOGLE_HYBRID",
        "database": {"path": "database/embeddings.db", "faiss_path": "database/embeddings.index"},
        "_project_dir": project_dir,
    }


async def test_queue_screen_loads(tmp_path):
    from geovibes.cli.tui.app import GeoVibesTUI

    config = _make_project_with_data(tmp_path)
    app = GeoVibesTUI(config)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        table = screen.query_one("#jobs-table")
        assert table is not None


async def test_queue_shows_job_count(tmp_path):
    from geovibes.cli.tui.app import GeoVibesTUI

    config = _make_project_with_data(tmp_path)
    app = GeoVibesTUI(config)

    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        header = screen.query_one("#queue-header")
        content = str(header.render())
        assert "Jobs: 2" in content


async def test_switch_to_review_screen(tmp_path):
    from geovibes.cli.tui.app import GeoVibesTUI

    config = _make_project_with_data(tmp_path)
    app = GeoVibesTUI(config)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        screen = app.screen
        meta = screen.query_one("#meta-panel")
        assert meta is not None
        content = str(meta.render())
        assert "Detection #1/20" in content


async def test_review_accept_advances(tmp_path):
    from geovibes.cli.tui.app import GeoVibesTUI

    config = _make_project_with_data(tmp_path)
    app = GeoVibesTUI(config)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()

        reviews = load_reviews(config["_project_dir"])
        accepted = reviews[reviews["status"] == "accepted"]
        assert len(accepted) == 1


async def test_review_undo(tmp_path):
    from geovibes.cli.tui.app import GeoVibesTUI

    config = _make_project_with_data(tmp_path)
    app = GeoVibesTUI(config)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()

        reviews = load_reviews(config["_project_dir"])
        assert len(reviews[reviews["status"] == "accepted"]) == 1

        await pilot.press("u")
        await pilot.pause()

        reviews = load_reviews(config["_project_dir"])
        assert len(reviews[reviews["status"] == "accepted"]) == 0
