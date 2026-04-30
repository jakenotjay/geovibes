"""FAISS search and query vector computation for the CLI."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import faiss
import numpy as np
import pandas as pd


def connect_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("SET memory_limit='24GB'")
    return conn


def load_faiss_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def nearest_point(
    conn: duckdb.DuckDBPyConnection, lon: float, lat: float
) -> Optional[Tuple]:
    """Return the nearest embedding to (lon, lat).

    Assumes `geo_embeddings.geometry` is stored in WGS84 (EPSG:4326);
    `ST_Distance_Sphere` requires lon/lat inputs and returns meters.
    The CLI loader writes 4326; if a future loader stores projected
    coordinates, this query must be updated.
    """
    sql = """
    SELECT id, ST_AsGeoJSON(geometry) as geometry,
           ST_Distance_Sphere(geometry, ST_Point(?, ?)) AS dist_m,
           CAST(embedding AS FLOAT[]) as embedding
    FROM geo_embeddings
    ORDER BY dist_m
    LIMIT 1
    """
    return conn.execute(sql, [lon, lat]).fetchone()


def fetch_embeddings(
    conn: duckdb.DuckDBPyConnection, point_ids: List[int]
) -> pd.DataFrame:
    if not point_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?" for _ in point_ids])
    sql = f"""
    SELECT id, CAST(embedding AS FLOAT[]) as embedding,
           ST_X(geometry) as lon, ST_Y(geometry) as lat
    FROM geo_embeddings
    WHERE id IN ({placeholders})
    """
    return conn.execute(sql, point_ids).fetchdf()


def compute_query_vector(
    pos_embeddings: List[np.ndarray],
    neg_embeddings: Optional[List[np.ndarray]] = None,
) -> Optional[np.ndarray]:
    if not pos_embeddings:
        return None
    pos_vec = np.mean(pos_embeddings, axis=0)
    if neg_embeddings:
        neg_vec = np.mean(neg_embeddings, axis=0)
    else:
        neg_vec = np.zeros_like(pos_vec)
    return 2 * pos_vec - neg_vec


def search_faiss(
    index: faiss.Index,
    query_vector: np.ndarray,
    n_neighbors: int = 1000,
    nprobe: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    query = query_vector.reshape(1, -1).astype(np.float32)
    distances, ids = index.search(query, n_neighbors)
    return distances[0], ids[0]


def search_metadata(
    conn: duckdb.DuckDBPyConnection,
    faiss_ids: List[int],
) -> pd.DataFrame:
    if not faiss_ids:
        return pd.DataFrame()
    valid_ids = [int(i) for i in faiss_ids if i >= 0]
    if not valid_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?" for _ in valid_ids])
    sql = f"""
    SELECT id, ST_AsGeoJSON(geometry) as geometry_json,
           ST_X(geometry) as lon, ST_Y(geometry) as lat
    FROM geo_embeddings
    WHERE id IN ({placeholders})
    """
    return conn.execute(sql, valid_ids).fetchdf()
