"""Persistent job/detection/comment ledger backed by parquet files."""

import json
import warnings
from contextlib import contextmanager

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


JOBS_FILE = "jobs.parquet"
REVIEWS_FILE = "reviews.parquet"
COMMENTS_FILE = "comments.parquet"

JOBS_SCHEMA = pa.schema([
    ("job_id", pa.int64()),
    ("job_type", pa.string()),
    ("parent_job_id", pa.int64()),
    ("iteration", pa.int64()),
    ("status", pa.string()),
    ("config", pa.string()),
    ("summary", pa.string()),
    ("started_at", pa.timestamp("us", tz="UTC")),
    ("finished_at", pa.timestamp("us", tz="UTC")),
    ("reviewer", pa.string()),
])

REVIEWS_SCHEMA = pa.schema([
    ("detection_id", pa.int64()),
    ("embedding_id", pa.int64()),
    ("geometry", pa.binary()),
    ("cluster_id", pa.int64()),
    ("iteration", pa.int64()),
    ("job_id", pa.int64()),
    ("review_job_id", pa.int64()),
    ("score", pa.float32()),
    ("status", pa.string()),
    ("reviewer", pa.string()),
    ("reviewed_at", pa.timestamp("us", tz="UTC")),
])

COMMENTS_SCHEMA = pa.schema([
    ("comment_id", pa.int64()),
    ("detection_id", pa.int64()),
    ("job_id", pa.int64()),
    ("reviewer", pa.string()),
    ("comment", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
])


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _file_lock(project_dir: Path, name: str = "ledger"):
    if not _HAS_FCNTL:
        warnings.warn(
            "File locking unavailable on this platform; concurrent writes are not protected",
            stacklevel=2,
        )
        yield
        return
    lock_path = project_dir / f".{name}.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _jobs_path(project_dir: Path) -> Path:
    return project_dir / JOBS_FILE


def _reviews_path(project_dir: Path) -> Path:
    return project_dir / REVIEWS_FILE


def _comments_path(project_dir: Path) -> Path:
    return project_dir / COMMENTS_FILE


def _next_id(df: pd.DataFrame, id_col: str) -> int:
    if df.empty:
        return 1
    max_val = df[id_col].dropna().max()
    if pd.isna(max_val):
        return 1
    return int(max_val) + 1


# --- Jobs ---

def load_jobs(project_dir: Path) -> pd.DataFrame:
    path = _jobs_path(project_dir)
    if not path.exists():
        return pd.DataFrame(columns=[f.name for f in JOBS_SCHEMA])
    return pd.read_parquet(path)


def create_job(
    project_dir: Path,
    job_type: str,
    iteration: int = 0,
    config: Optional[Dict[str, Any]] = None,
    parent_job_id: Optional[int] = None,
    reviewer: Optional[str] = None,
) -> int:
    with _file_lock(project_dir, "jobs"):
        return _create_job_locked(project_dir, job_type, iteration, config, parent_job_id, reviewer)


def _create_job_locked(project_dir, job_type, iteration, config, parent_job_id, reviewer) -> int:
    jobs = load_jobs(project_dir)
    job_id = _next_id(jobs, "job_id")

    new_row = {
        "job_id": job_id,
        "job_type": job_type,
        "parent_job_id": parent_job_id,
        "iteration": iteration,
        "status": "running",
        "config": json.dumps(config or {}),
        "summary": "",
        "started_at": _now(),
        "finished_at": None,
        "reviewer": reviewer,
    }

    new_df = pd.DataFrame([new_row])
    if jobs.empty:
        jobs = new_df
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            jobs = pd.concat([jobs, new_df], ignore_index=True)
    _write_jobs(project_dir, jobs)
    return job_id


def update_job(
    project_dir: Path,
    job_id: int,
    status: Optional[str] = None,
    summary: Optional[str] = None,
) -> None:
    with _file_lock(project_dir, "jobs"):
        _update_job_locked(project_dir, job_id, status, summary)


@contextmanager
def track_job(project_dir: Path, job_id: int):
    """Context manager that marks a job as failed on unhandled exception."""
    try:
        yield job_id
    except BaseException as exc:
        msg = str(exc)[:200] if str(exc) else type(exc).__name__
        update_job(project_dir, job_id, status="failed", summary=msg)
        raise


def _update_job_locked(project_dir, job_id, status, summary):
    jobs = load_jobs(project_dir)
    mask = jobs["job_id"] == job_id
    if not mask.any():
        raise ValueError(f"Job {job_id} not found")

    if status is not None:
        jobs.loc[mask, "status"] = status
    if summary is not None:
        jobs.loc[mask, "summary"] = summary
    if status in ("done", "failed"):
        jobs.loc[mask, "finished_at"] = _now()

    _write_jobs(project_dir, jobs)


def _write_jobs(project_dir: Path, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df, schema=JOBS_SCHEMA, preserve_index=False)
    pq.write_table(table, _jobs_path(project_dir))


# --- Reviews (detections) ---

def load_reviews(project_dir: Path) -> pd.DataFrame:
    path = _reviews_path(project_dir)
    if not path.exists():
        return pd.DataFrame(columns=[f.name for f in REVIEWS_SCHEMA])
    return pd.read_parquet(path)


def _write_reviews(project_dir: Path, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df, schema=REVIEWS_SCHEMA, preserve_index=False)
    pq.write_table(table, _reviews_path(project_dir))


def save_reviews(project_dir: Path, df: pd.DataFrame) -> None:
    with _file_lock(project_dir, "reviews"):
        _write_reviews(project_dir, df)


def update_review(
    project_dir: Path,
    detection_id: int,
    status: str,
    reviewer: Optional[str],
    review_job_id: Optional[int] = None,
    reviewed_at: Optional[datetime] = None,
) -> None:
    with _file_lock(project_dir, "reviews"):
        _update_review_locked(project_dir, detection_id, status, reviewer, review_job_id, reviewed_at)


def _update_review_locked(project_dir, detection_id, status, reviewer, review_job_id, reviewed_at):
    reviews = load_reviews(project_dir)
    mask = reviews["detection_id"] == detection_id
    if not mask.any():
        raise ValueError(f"Detection {detection_id} not found")

    reviews.loc[mask, "status"] = status
    reviews.loc[mask, "reviewer"] = reviewer
    reviews.loc[mask, "reviewed_at"] = reviewed_at if reviewed_at is not None else _now()
    if review_job_id is not None:
        reviews.loc[mask, "review_job_id"] = review_job_id

    _write_reviews(project_dir, reviews)


# --- Comments ---

def load_comments(
    project_dir: Path,
    detection_id: Optional[int] = None,
) -> pd.DataFrame:
    path = _comments_path(project_dir)
    if not path.exists():
        return pd.DataFrame(columns=[f.name for f in COMMENTS_SCHEMA])
    df = pd.read_parquet(path)
    if detection_id is not None:
        df = df[df["detection_id"] == detection_id]
    return df


def add_comment(
    project_dir: Path,
    detection_id: int,
    reviewer: str,
    comment: str,
    job_id: Optional[int] = None,
) -> int:
    with _file_lock(project_dir, "comments"):
        return _add_comment_locked(project_dir, detection_id, reviewer, comment, job_id)


def _add_comment_locked(project_dir, detection_id, reviewer, comment, job_id) -> int:
    comments = load_comments(project_dir)
    comment_id = _next_id(comments, "comment_id")

    new_row = {
        "comment_id": comment_id,
        "detection_id": detection_id,
        "job_id": job_id,
        "reviewer": reviewer,
        "comment": comment,
        "created_at": _now(),
    }

    new_df = pd.DataFrame([new_row])
    if comments.empty:
        comments = new_df
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            comments = pd.concat([comments, new_df], ignore_index=True)
    table = pa.Table.from_pandas(comments, schema=COMMENTS_SCHEMA, preserve_index=False)
    pq.write_table(table, _comments_path(project_dir))
    return comment_id


# --- Labels ---

LABELS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("geometry", pa.binary()),
    ("label", pa.int8()),
    ("source", pa.string()),
    ("iteration", pa.int64()),
    ("created_at", pa.timestamp("us", tz="UTC")),
])


def labels_dir(project_dir: Path) -> Path:
    d = project_dir / "labels"
    d.mkdir(exist_ok=True)
    return d


def save_labels(
    project_dir: Path,
    df: pd.DataFrame,
    iteration: int,
) -> Path:
    path = labels_dir(project_dir) / f"iteration_{iteration:03d}.parquet"
    table = pa.Table.from_pandas(df, schema=LABELS_SCHEMA, preserve_index=False)
    pq.write_table(table, path)
    return path


def load_all_labels(project_dir: Path) -> pd.DataFrame:
    d = labels_dir(project_dir)
    files = sorted(d.glob("iteration_*.parquet"))
    if not files:
        return pd.DataFrame(columns=[f.name for f in LABELS_SCHEMA])
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
