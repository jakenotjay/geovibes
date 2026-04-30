"""CLI wrappers for train/infer/cluster pipeline steps."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import duckdb
import numpy as np
import pandas as pd

from geovibes.cli.ledger import create_job, load_reviews, track_job, update_job, save_reviews
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

    with track_job(project_dir, job_id):
        start = time.perf_counter()

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
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

            all_labels = all_labels.drop(columns=["embedding"], errors="ignore")
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
                    eval_metric="logloss",
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
        finally:
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

    with track_job(project_dir, job_id):
        start = time.perf_counter()

        import joblib
        model = joblib.load(model_path)
        click.echo(f"Loaded model from {model_path}")

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            conn.execute("INSTALL spatial; LOAD spatial;")
            conn.execute("SET memory_limit='24GB'")
            conn.execute("SET preserve_insertion_order=false")
            conn.execute("SET threads=4")

            total_count = conn.execute("SELECT COUNT(*) FROM geo_embeddings").fetchone()[0]
            click.echo(f"Scoring {total_count:,} embeddings...")

            detections = []
            scored = 0
            last_id = -1

            while scored < total_count:
                batch = conn.execute(
                    """
                    SELECT id, CAST(embedding AS FLOAT[]) as embedding,
                           ST_AsWKB(geometry) as geometry
                    FROM geo_embeddings
                    WHERE id > ?
                    ORDER BY id
                    LIMIT ?
                    """,
                    [last_id, batch_size],
                ).fetchdf()

                if batch.empty:
                    break

                last_id = int(batch["id"].iloc[-1])
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
        finally:
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

        from geovibes.cli.ledger import load_reviews

        existing = load_reviews(project_dir)
        if not existing.empty:
            existing = existing[existing["iteration"] != iteration]
            next_det_id = int(existing["detection_id"].max()) + 1 if not existing.empty else 1
        else:
            next_det_id = 1

        new_rows = pd.DataFrame({
            "detection_id": range(next_det_id, next_det_id + n_det),
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

        if not existing.empty:
            existing_cols = set(existing.columns)
            new_cols = set(new_rows.columns)
            if existing_cols != new_cols:
                added = sorted(new_cols - existing_cols)
                removed = sorted(existing_cols - new_cols)
                reviews_path = project_dir / "reviews.parquet"
                raise ValueError(
                    f"Schema drift between existing reviews and new detections "
                    f"(added: {added}, removed: {removed}). "
                    f"To rebuild from scratch, delete {reviews_path} and re-run infer; "
                    f"to migrate, add the new columns to existing rows with safe defaults "
                    f"before running infer again."
                )
            new_rows = new_rows[existing.columns]
            reviews_df = pd.concat([existing, new_rows], ignore_index=True)
        else:
            reviews_df = new_rows
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

    with track_job(project_dir, job_id):
        start = time.perf_counter()

        reviews = load_reviews(project_dir)
        if reviews.empty:
            update_job(project_dir, job_id, status="done", summary="No detections to cluster")
            return {"n_clusters": 0}

        import shapely.wkb
        iter_mask = reviews["iteration"] == iteration
        if not iter_mask.any():
            update_job(project_dir, job_id, status="done", summary=f"No detections at iteration {iteration}")
            return {"n_clusters": 0}

        valid_mask = iter_mask & reviews["geometry"].notna()
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

        reviews.loc[iter_mask, "cluster_id"] = pd.NA
        reviews.loc[valid_mask, "cluster_id"] = pd.array(
            cluster_labels.tolist(), dtype="Int64"
        )
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


def run_validate(
    project_dir: Path,
    truth_path: Path,
    buffer_m: int = 500,
    use_clusters: bool = True,
) -> Dict:
    import geopandas as gpd
    import shapely.wkb

    start = time.perf_counter()

    truth = gpd.read_file(truth_path)
    if truth.crs and truth.crs.to_epsg() != 4326:
        truth = truth.to_crs(4326)
    click.echo(f"Ground truth: {len(truth)} points")

    reviews = load_reviews(project_dir)
    if reviews.empty:
        click.echo("No detections to validate")
        return {}

    valid_mask = reviews["geometry"].notna()
    det_points = [shapely.wkb.loads(b) for b in reviews[valid_mask]["geometry"]]

    if use_clusters:
        cluster_ids = reviews.loc[valid_mask, "cluster_id"]
        det_df = gpd.GeoDataFrame(
            {"cluster_id": cluster_ids.values, "geometry": det_points},
            crs="EPSG:4326",
        )
        clustered = det_df[det_df["cluster_id"].notna() & (det_df["cluster_id"] >= 0)]
        if clustered.empty:
            det_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            utm_for_dissolve = clustered.estimate_utm_crs()
            dissolved = clustered.to_crs(utm_for_dissolve).dissolve(by="cluster_id")
            centroid_geoms = dissolved.centroid.to_crs(4326)
            det_gdf = gpd.GeoDataFrame(geometry=centroid_geoms.values, crs="EPSG:4326")
        click.echo(f"Detections: {len(det_gdf)} cluster centroids")
    else:
        det_gdf = gpd.GeoDataFrame({"geometry": det_points}, crs="EPSG:4326")
        click.echo(f"Detections: {len(det_gdf)} raw points")

    if det_gdf.empty:
        n_truth = len(truth)
        click.echo(f"\nValidation results (buffer={buffer_m}m): no detections")
        click.echo(f"  TP=0  FP=0  FN={n_truth}  Precision=0.000  Recall=0.000  F1=0.000")
        return {"tp": 0, "fp": 0, "fn": n_truth, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    utm_crs = det_gdf.estimate_utm_crs()
    det_proj = det_gdf.to_crs(utm_crs)
    truth_proj = truth.to_crs(utm_crs)

    from shapely import STRtree

    truth_tree = STRtree(truth_proj.geometry.values)
    matched_truth: set = set()
    fp = 0
    for det_geom in det_proj.geometry.values:
        nearest_idx = int(truth_tree.nearest(det_geom))
        nearest_truth = truth_proj.geometry.values[nearest_idx]
        if det_geom.distance(nearest_truth) <= buffer_m:
            matched_truth.add(nearest_idx)
        else:
            fp += 1

    tp = len(matched_truth)
    fn = len(truth) - tp
    n_det = len(det_proj)
    precision = tp / n_det if n_det > 0 else 0.0
    recall = tp / len(truth) if len(truth) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elapsed = time.perf_counter() - start

    click.echo(f"\nValidation results (buffer={buffer_m}m, CRS={utm_crs.to_string()}):")
    click.echo(f"  True positives:  {tp}  (unique truth points matched)")
    click.echo(f"  False positives: {fp}  (detections with no truth in buffer)")
    click.echo(f"  False negatives: {fn}  (truth points unmatched)")
    click.echo(f"  Precision: {precision:.3f}")
    click.echo(f"  Recall:    {recall:.3f}")
    click.echo(f"  F1:        {f1:.3f}")
    click.echo(f"  ({elapsed:.1f}s)")

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _load_label_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".geojson" or suffix == ".json":
        import geopandas as gpd
        df = gpd.read_file(path)
        if "label" not in df.columns and "class" in df.columns:
            class_to_label = {
                "geovibes_pos": 1, "positive": 1, "relabel_pos": 1,
                "geovibes_neg": 0, "negative": 0, "relabel_neg": 0,
                "geovibes_sampled_neg": 0, "sampled": 0,
            }
            normalised = df["class"].str.lower()
            n_missing = int(normalised.isna().sum())
            if n_missing:
                raise ValueError(
                    f"Label file {path.name} has {n_missing} rows with missing 'class'. "
                    f"Drop them, fill them, or supply an explicit 'label' column."
                )
            unknown = set(normalised[~normalised.isin(class_to_label)].unique())
            if unknown:
                raise ValueError(
                    f"Label file {path.name} has unmapped class values: {sorted(unknown)}. "
                    f"Add them to class_to_label or supply an explicit 'label' column."
                )
            df["label"] = normalised.map(class_to_label).astype(int)
    else:
        raise ValueError(f"Unsupported label file format: {suffix}")

    if "id" not in df.columns and "tile_id" in df.columns:
        df["id"] = df["tile_id"]

    missing = [col for col in ("id", "label") if col not in df.columns]
    if missing:
        raise ValueError(f"Label file {path.name} missing required columns: {missing}")

    df["id"] = df["id"].astype(int)
    return df


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
