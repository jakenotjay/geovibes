"""Load AEF embeddings into a GeoVibes project database.

Uses aef-loader to fetch COG tiles from Source Cooperative,
then extracts embedding vectors and builds DuckDB + FAISS index.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Optional

import click
import duckdb
import faiss
import numpy as np

from geovibes.cli.ledger import create_job, update_job
from geovibes.cli.project import load_project


RES_TO_IFD = {10: 0, 20: 1, 40: 2, 80: 3}


def res_to_ifd(resolution: int) -> int:
    if resolution in RES_TO_IFD:
        return RES_TO_IFD[resolution]
    raise ValueError(f"Unsupported resolution {resolution}m. Use one of: {list(RES_TO_IFD.keys())}")


async def _fetch_embeddings(bbox, years, ifd, source):
    """Fetch AEF embeddings via aef-loader and return a reprojected Dataset."""
    from aef_loader import AEFIndex, VirtualTiffReader, DataSource
    from aef_loader.utils import reproject_datatree, int8_to_float32
    from odc.geo.geobox import GeoBox

    ds_enum = DataSource.SOURCE_COOP if source == "source-coop" else DataSource.GCS

    click.echo(f"Downloading AEF index...")
    index = AEFIndex(source=ds_enum)
    await index.download()
    index.load()

    click.echo(f"Querying tiles for bbox={bbox}, years={years}...")
    tiles = await index.query(bbox=tuple(bbox), years=tuple(years))
    click.echo(f"Found {len(tiles)} tiles across zones")

    if not tiles:
        raise ValueError("No tiles found for the given bbox/years")

    async with VirtualTiffReader(source=ds_enum) as reader:
        click.echo(f"Opening tiles at IFD {ifd}...")
        tree = await reader.open_tiles_by_zone(tiles, ifd=ifd, chunks=None)

    click.echo(f"Reprojecting to EPSG:4326...")
    target = GeoBox.from_bbox(
        bbox=tuple(bbox),
        crs="EPSG:4326",
        resolution=0.001,
    )
    combined = reproject_datatree(tree, target)
    combined = int8_to_float32(combined)

    return combined


def _extract_vectors_from_dataset(ds, var_name="embeddings"):
    """Extract embedding vectors and centroids from an xarray Dataset.

    Returns:
        embeddings: np.ndarray shape (N, D) float32
        lons: np.ndarray shape (N,)
        lats: np.ndarray shape (N,)
    """
    da = ds[var_name]

    if "time" in da.dims:
        da = da.isel(time=0)

    click.echo(f"Dataset shape: {dict(da.sizes)}")
    click.echo(f"Computing embeddings (this may take a while)...")

    start = time.perf_counter()
    data = da.values
    elapsed = time.perf_counter() - start
    click.echo(f"Computed in {elapsed:.1f}s")

    n_bands, n_y, n_x = data.shape
    n_pixels = n_y * n_x

    embeddings = data.reshape(n_bands, n_pixels).T.astype(np.float32)

    xs = da.coords["x"].values
    ys = da.coords["y"].values
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    lons = lon_grid.reshape(n_pixels)
    lats = lat_grid.reshape(n_pixels)

    nodata_mask = np.isnan(embeddings).any(axis=1)
    valid = ~nodata_mask
    n_valid = valid.sum()
    click.echo(f"Valid pixels: {n_valid:,} / {n_pixels:,} ({100*n_valid/n_pixels:.1f}%)")

    return embeddings[valid], lons[valid], lats[valid]


def _create_duckdb(db_path: Path, embeddings: np.ndarray, lons: np.ndarray, lats: np.ndarray):
    """Create DuckDB database with geo_embeddings table."""
    click.echo(f"Creating DuckDB at {db_path}...")
    start = time.perf_counter()

    conn = duckdb.connect(str(db_path))
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("SET memory_limit='24GB'")

    n, dim = embeddings.shape
    conn.execute(f"""
        CREATE TABLE geo_embeddings (
            id BIGINT PRIMARY KEY,
            embedding FLOAT[{dim}],
            geometry GEOMETRY
        )
    """)

    chunk_size = 50_000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk_emb = embeddings[i:end]
        chunk_lon = lons[i:end]
        chunk_lat = lats[i:end]

        values = []
        for j in range(end - i):
            emb_list = chunk_emb[j].tolist()
            values.append((i + j, emb_list, chunk_lon[j], chunk_lat[j]))

        conn.executemany(
            "INSERT INTO geo_embeddings VALUES (?, ?::FLOAT[], ST_Point(?, ?))",
            values,
        )
        click.echo(f"  Inserted {end:,}/{n:,}", nl=False)
        click.echo("\r", nl=False)

    click.echo()

    conn.execute("CREATE INDEX geom_spatial_idx ON geo_embeddings USING RTREE (geometry)")

    elapsed = time.perf_counter() - start
    click.echo(f"DuckDB created with {n:,} embeddings in {elapsed:.1f}s")
    conn.close()


def _create_faiss_index(index_path: Path, embeddings: np.ndarray):
    """Build FAISS IVF-PQ index from embeddings."""
    click.echo(f"Building FAISS index...")
    start = time.perf_counter()

    n, dim = embeddings.shape

    if n < 10_000:
        click.echo("  Using flat index (small dataset)")
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
    else:
        nlist = min(4096, n // 100)
        m = min(64, dim)
        nbits = 8

        click.echo(f"  Training IVF-PQ: nlist={nlist}, m={m}, nbits={nbits}")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        train_size = min(n, max(nlist * 40, 200_000))
        train_indices = np.random.choice(n, train_size, replace=False)
        train_data = embeddings[train_indices]

        index.train(train_data)
        click.echo(f"  Trained on {train_size:,} vectors")

        batch_size = 500_000
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            index.add(embeddings[i:end])
            click.echo(f"  Added {end:,}/{n:,}", nl=False)
            click.echo("\r", nl=False)

        click.echo()

    faiss.write_index(index, str(index_path))
    elapsed = time.perf_counter() - start
    click.echo(f"FAISS index saved ({index.ntotal:,} vectors) in {elapsed:.1f}s")


def run_load(project_dir: Path, source: str = "source-coop") -> Dict:
    """Run the full load pipeline: aef-loader → DuckDB + FAISS."""
    config = load_project(project_dir)

    bbox = config["bbox"]
    years = config["years"]
    resolution = config.get("resolution", 80)
    ifd = res_to_ifd(resolution)

    job_id = create_job(
        project_dir,
        job_type="load",
        iteration=0,
        config={"bbox": bbox, "years": years, "resolution": resolution, "ifd": ifd, "source": source},
    )

    start = time.perf_counter()

    ds = asyncio.run(_fetch_embeddings(bbox, years, ifd, source))

    embeddings, lons, lats = _extract_vectors_from_dataset(ds)

    db_path = project_dir / config["database"]["path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _create_duckdb(db_path, embeddings, lons, lats)

    faiss_path = project_dir / config["database"]["faiss_path"]
    _create_faiss_index(faiss_path, embeddings)

    elapsed = time.perf_counter() - start

    update_job(
        project_dir, job_id,
        status="done",
        summary=f"{len(embeddings):,} embeddings at {resolution}m ({elapsed:.0f}s)",
    )

    click.echo(f"Load complete: {len(embeddings):,} embeddings in {elapsed:.1f}s")
    return {"n_embeddings": len(embeddings), "job_id": job_id}
