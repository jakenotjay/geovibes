"""CLI wrappers for train/infer/cluster pipeline steps."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import duckdb
import numpy as np
import pandas as pd

from geovibes.cli.ledger import create_job, load_reviews, update_job, save_reviews
from geovibes.cli.project import load_project, save_project


def run_train(
    project_dir: Path,
    positives: List[Path],
    negatives: List[Path],
    classifier_type: str = "xgboost",
    threshold: float = 0.5,
    test_fraction: float = 0.2,
) -> Dict:
    config = load_project(project_dir)
    db_path = project_dir / config["database"]["path"]
    iteration = config.get("iteration", 0)

    job_id = create_job(
        project_dir,
        job_type="train",
        iteration=iteration,
        config={
            "positives": [str(p) for p in positives],
            "negatives": [str(n) for n in negatives],
            "classifier": classifier_type,
            "threshold": threshold,
        },
    )

    start = time.perf_counter()

    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("SET memory_limit='24GB'")

    pos_dfs = [_load_label_file(f) for f in positives]
    neg_dfs = [_load_label_file(f) for f in negatives] if negatives else []

    all_labels = pd.concat(pos_dfs + neg_dfs, ignore_index=True)
    pos_count = int((all_labels["label"] == 1).sum())
    neg_count = int((all_labels["label"] == 0).sum())
    click.echo(f"Training data: {pos_count} positive, {neg_count} negative")

    point_ids = all_labels["id"].tolist()
    embeddings = _fetch_all_embeddings(conn, point_ids)

    all_labels = all_labels.merge(embeddings, on="id", how="inner")
    click.echo(f"Matched {len(all_labels)} embeddings from database")

    X = np.vstack(all_labels["embedding"].values).astype(np.float32)
    y = all_labels["label"].values.astype(np.int32)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=y
    )

    if classifier_type == "linear-svm":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=42)
        model = CalibratedClassifierCV(base, cv=3)
    else:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
            use_label_encoder=False, eval_metric="logloss",
        )

    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    from sklearn.metrics import f1_score, roc_auc_score
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    click.echo(f"Train time: {train_time:.2f}s | F1: {f1:.3f} | AUC: {auc:.3f}")

    import joblib
    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{classifier_type}_v{iteration}.pkl"
    joblib.dump(model, model_path)
    click.echo(f"Model saved to {model_path}")

    conn.close()

    update_job(
        project_dir, job_id,
        status="done",
        summary=f"F1={f1:.3f} AUC={auc:.3f} ({pos_count}+/{neg_count}-)",
    )

    return {
        "model_path": model_path,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "job_id": job_id,
    }


def run_infer(
    project_dir: Path,
    model_path: Optional[Path] = None,
    threshold: float = 0.5,
    batch_size: int = 100_000,
) -> Dict:
    config = load_project(project_dir)
    db_path = project_dir / config["database"]["path"]
    iteration = config.get("iteration", 0)

    if model_path is None:
        model_dir = project_dir / "models"
        models = sorted(model_dir.glob("*.pkl"))
        if not models:
            raise FileNotFoundError("No trained models found in models/")
        model_path = models[-1]

    job_id = create_job(
        project_dir,
        job_type="infer",
        iteration=iteration,
        config={"model_path": str(model_path), "threshold": threshold, "batch_size": batch_size},
    )

    start = time.perf_counter()

    import joblib
    model = joblib.load(model_path)
    click.echo(f"Loaded model from {model_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("SET memory_limit='24GB'")

    total_count = conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
    click.echo(f"Scoring {total_count:,} embeddings...")

    detections = []
    scored = 0

    for offset in range(0, total_count, batch_size):
        batch = conn.execute(
            """
            SELECT id, CAST(embedding AS FLOAT[]) as embedding,
                   ST_AsBinary(geometry) as geometry
            FROM geo_embeddings
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            [batch_size, offset],
        ).fetchdf()

        if batch.empty:
            break

        X_batch = np.vstack(batch["embedding"].values).astype(np.float32)
        proba = model.predict_proba(X_batch)[:, 1]
        scored += len(batch)

        mask = proba >= threshold
        if mask.any():
            det_batch = batch[mask].copy()
            det_batch["score"] = proba[mask]
            detections.append(det_batch[["id", "geometry", "score"]])

        elapsed = time.perf_counter() - start
        click.echo(f"  {scored:,}/{total_count:,} ({elapsed:.1f}s)", nl=False)
        click.echo("\r", nl=False)

    click.echo()

    conn.close()
    infer_time = time.perf_counter() - start

    if detections:
        all_detections = pd.concat(detections, ignore_index=True)
    else:
        all_detections = pd.DataFrame(columns=["id", "geometry", "score"])

    n_det = len(all_detections)
    click.echo(f"Found {n_det:,} detections above threshold {threshold} in {infer_time:.1f}s")

    det_path = project_dir / "detections" / f"iteration_{iteration:03d}.parquet"
    det_path.parent.mkdir(exist_ok=True)
    all_detections.to_parquet(det_path)

    reviews_df = pd.DataFrame({
        "detection_id": range(1, n_det + 1),
        "embedding_id": all_detections["id"].values,
        "geometry": all_detections["geometry"].values,
        "cluster_id": pd.array([None] * n_det, dtype="Int64"),
        "iteration": iteration,
        "job_id": job_id,
        "review_job_id": pd.array([None] * n_det, dtype="Int64"),
        "score": all_detections["score"].values.astype(np.float32),
        "status": "pending",
        "reviewer": pd.array([None] * n_det, dtype="string"),
        "reviewed_at": pd.array([None] * n_det, dtype="datetime64[us, UTC]"),
    })
    save_reviews(project_dir, reviews_df)

    update_job(
        project_dir, job_id,
        status="done",
        summary=f"{n_det:,} detections (t={threshold})",
    )

    return {"detections_path": det_path, "n_detections": n_det, "job_id": job_id}


def run_cluster(
    project_dir: Path,
    eps_m: int = 500,
    min_samples: int = 2,
) -> Dict:
    config = load_project(project_dir)
    iteration = config.get("iteration", 0)

    job_id = create_job(
        project_dir,
        job_type="cluster",
        iteration=iteration,
        config={"eps_m": eps_m, "min_samples": min_samples},
    )

    start = time.perf_counter()

    reviews = load_reviews(project_dir)
    if reviews.empty:
        update_job(project_dir, job_id, status="done", summary="No detections to cluster")
        return {"n_clusters": 0}

    import shapely.wkb
    valid_mask = reviews["geometry"].notna()
    valid_reviews = reviews[valid_mask].copy()

    if valid_reviews.empty:
        update_job(project_dir, job_id, status="done", summary="No detections with geometry")
        return {"n_clusters": 0}

    coords = []
    for geom_bytes in valid_reviews["geometry"]:
        point = shapely.wkb.loads(geom_bytes)
        coords.append([point.y, point.x])
    coords = np.array(coords)

    coords_rad = np.radians(coords)
    eps_rad = eps_m / 6_378_137.0

    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    cluster_labels = db.fit_predict(coords_rad)

    reviews["cluster_id"] = -1
    reviews.loc[valid_mask, "cluster_id"] = cluster_labels
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())

    save_reviews(project_dir, reviews)

    elapsed = time.perf_counter() - start
    click.echo(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({elapsed:.1f}s)")

    update_job(
        project_dir, job_id,
        status="done",
        summary=f"{n_clusters} clusters from {len(reviews)} detections",
    )

    return {"n_clusters": n_clusters, "n_noise": n_noise, "job_id": job_id}


def _load_label_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".geojson" or suffix == ".json":
        import geopandas as gpd
        gdf = gpd.read_file(path)
        if "label" not in gdf.columns and "class" in gdf.columns:
            gdf["label"] = gdf["class"].map(
                lambda c: 1 if "pos" in str(c).lower() else 0
            )
        return gdf
    raise ValueError(f"Unsupported label file format: {suffix}")


def _fetch_all_embeddings(conn, point_ids: List) -> pd.DataFrame:
    chunk_size = 10_000
    dfs = []
    for i in range(0, len(point_ids), chunk_size):
        chunk = point_ids[i:i + chunk_size]
        placeholders = ",".join(["?" for _ in chunk])
        sql = f"""
        SELECT id, CAST(embedding AS FLOAT[]) as embedding
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        dfs.append(conn.execute(sql, chunk).fetchdf())
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["id", "embedding"])
