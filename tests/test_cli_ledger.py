"""Tests for CLI ledger (jobs, reviews, comments)."""

from pathlib import Path

import pandas as pd

from geovibes.cli.project import init_project
from geovibes.cli.ledger import (
    create_job,
    update_job,
    load_jobs,
    load_reviews,
    save_reviews,
    update_review,
    add_comment,
    load_comments,
    save_labels,
    load_all_labels,
    REVIEWS_SCHEMA,
)


def _make_project(tmp_path: Path) -> Path:
    return init_project(
        name="test",
        bbox=[0, 0, 1, 1],
        years=[2024],
        base_dir=tmp_path,
    )


def test_create_and_load_jobs(tmp_path):
    project = _make_project(tmp_path)

    job_id = create_job(project, "init", iteration=0, config={"bbox": [0, 0, 1, 1]})
    assert job_id == 1

    jobs = load_jobs(project)
    assert len(jobs) == 1
    assert jobs.iloc[0]["job_type"] == "init"
    assert jobs.iloc[0]["status"] == "running"


def test_create_multiple_jobs_increments_id(tmp_path):
    project = _make_project(tmp_path)

    id1 = create_job(project, "init")
    id2 = create_job(project, "load")
    id3 = create_job(project, "train")

    assert id1 == 1
    assert id2 == 2
    assert id3 == 3

    jobs = load_jobs(project)
    assert len(jobs) == 3


def test_update_job_status(tmp_path):
    project = _make_project(tmp_path)

    job_id = create_job(project, "train")
    update_job(project, job_id, status="done", summary="F1=0.94")

    jobs = load_jobs(project)
    row = jobs[jobs["job_id"] == job_id].iloc[0]
    assert row["status"] == "done"
    assert row["summary"] == "F1=0.94"
    assert pd.notna(row["finished_at"])


def test_empty_reviews(tmp_path):
    project = _make_project(tmp_path)
    reviews = load_reviews(project)
    assert reviews.empty
    assert "detection_id" in reviews.columns


def test_save_and_load_reviews(tmp_path):
    project = _make_project(tmp_path)

    df = pd.DataFrame([
        {
            "detection_id": 1,
            "embedding_id": 100,
            "geometry": b"\x00\x01",
            "cluster_id": None,
            "iteration": 1,
            "job_id": 1,
            "review_job_id": None,
            "score": 0.87,
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
        },
        {
            "detection_id": 2,
            "embedding_id": 200,
            "geometry": b"\x00\x02",
            "cluster_id": None,
            "iteration": 1,
            "job_id": 1,
            "review_job_id": None,
            "score": 0.65,
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
        },
    ])

    save_reviews(project, df)
    loaded = load_reviews(project)
    assert len(loaded) == 2
    assert loaded.iloc[0]["score"] == 0.87


def test_update_review(tmp_path):
    project = _make_project(tmp_path)

    df = pd.DataFrame([{
        "detection_id": 1,
        "embedding_id": 100,
        "geometry": b"\x00",
        "cluster_id": None,
        "iteration": 1,
        "job_id": 1,
        "review_job_id": None,
        "score": 0.9,
        "status": "pending",
        "reviewer": None,
        "reviewed_at": None,
    }])
    save_reviews(project, df)

    from datetime import datetime, timezone
    update_review(
        project,
        detection_id=1,
        status="accepted",
        reviewer="human",
        reviewed_at=datetime.now(timezone.utc),
    )

    loaded = load_reviews(project)
    assert loaded.iloc[0]["status"] == "accepted"
    assert loaded.iloc[0]["reviewer"] == "human"
    assert pd.notna(loaded.iloc[0]["reviewed_at"])


def test_add_and_load_comments(tmp_path):
    project = _make_project(tmp_path)

    cid1 = add_comment(project, detection_id=1, reviewer="human", comment="Looks correct")
    cid2 = add_comment(project, detection_id=1, reviewer="agent:gemini", comment="Confirmed via Places API")
    cid3 = add_comment(project, detection_id=2, reviewer="human", comment="River, not lagoon")

    assert cid1 == 1
    assert cid2 == 2
    assert cid3 == 3

    all_comments = load_comments(project)
    assert len(all_comments) == 3

    det1_comments = load_comments(project, detection_id=1)
    assert len(det1_comments) == 2

    det2_comments = load_comments(project, detection_id=2)
    assert len(det2_comments) == 1
    assert det2_comments.iloc[0]["comment"] == "River, not lagoon"


def test_save_and_load_labels(tmp_path):
    project = _make_project(tmp_path)

    df = pd.DataFrame([
        {"id": 1, "geometry": b"\x00", "label": 1, "source": "manual", "iteration": 1, "created_at": None},
        {"id": 2, "geometry": b"\x01", "label": 0, "source": "sampled", "iteration": 1, "created_at": None},
    ])

    path = save_labels(project, df, iteration=1)
    assert path.exists()
    assert "iteration_001" in path.name

    loaded = load_all_labels(project)
    assert len(loaded) == 2


def test_load_labels_across_iterations(tmp_path):
    project = _make_project(tmp_path)

    df1 = pd.DataFrame([
        {"id": 1, "geometry": b"\x00", "label": 1, "source": "manual", "iteration": 1, "created_at": None},
    ])
    df2 = pd.DataFrame([
        {"id": 2, "geometry": b"\x01", "label": 0, "source": "hard_negative", "iteration": 2, "created_at": None},
    ])

    save_labels(project, df1, iteration=1)
    save_labels(project, df2, iteration=2)

    loaded = load_all_labels(project)
    assert len(loaded) == 2
    assert set(loaded["iteration"]) == {1, 2}
