# GeoVibes CLI/TUI — End-to-End Test Plan

## For a Fresh Agent

This document tells you everything you need to run the GeoVibes CLI/TUI end-to-end with real Sumatra palm oil mill detection data, replicating the workflow from the Epoch blog post.

## Background: What Is This

GeoVibes is a geospatial similarity search tool. The original UI is a Jupyter notebook. We've built a CLI + Textual TUI alternative on the `jake/tui` branch that supports the full workflow:

```
seed labels → FAISS search → train classifier → inference → cluster → review
```

The CLI lives at `geovibes/cli/` and provides:
- `geovibes init` — create project directory with config
- `geovibes load` — fetch AEF satellite embeddings (not needed for this test — we have pre-built data)
- `geovibes train` — train Linear SVM or XGBoost on labelled data
- `geovibes infer` — batch score all embeddings, populate review ledger
- `geovibes cluster` — DBSCAN clustering of detections
- `geovibes tui` — Textual TUI with Queue (1), Review (2), Search (3) screens

Data is persisted in parquet files: `jobs.parquet` (append-only job ledger), `reviews.parquet` (mutable detection state), `comments.parquet` (append-only), `labels/iteration_NNN.parquet`.

## Prerequisites

```bash
cd ~/projects/geovibes
git checkout jake/tui
uv pip install -e .
```

Verify the CLI works:
```bash
uv run geovibes --version
uv run geovibes --help
```

## Available Data

All data lives in the epoch fork at `~/epoch/geovibes/`:

### Sumatra Embedding Database
- **DuckDB**: `~/epoch/geovibes/sumatra_db/sumatra_metadata.db`
  - Table: `geo_embeddings` with columns: `id` (BIGINT), `lon` (DOUBLE), `lat` (DOUBLE), `geometry` (GEOMETRY), `embedding` (FLOAT[64])
  - 75,312,294 rows (75M embeddings at 80m resolution)
  - Covers Sumatra, Indonesia (~95°E to 106°E, ~6°S to 6°N)
- **FAISS index**: `~/epoch/geovibes/sumatra_db/sumatra_faiss.index`
  - 75,312,294 vectors, dimension 64
  - IVF-PQ index

### Training Labels
- **Seed labels** (26 positive POME lagoons, 44 manual negatives):
  `~/epoch/geovibes/labeled_dataset_20260119_111222.geojson`
  - Properties: `id`, `label` (0/1), `class` (geovibes_pos/geovibes_neg), `embedding`, `source`

- **Clean labels with more positives** (116 total: ~50 pos, ~66 neg):
  `~/epoch/geovibes/labeled_dataset_20260120_132604_clean.geojson`

- **Sampled negatives** (3,499 stratified by landcover):
  `~/epoch/geovibes/sampled_negatives.geojson`
  - Properties: `tile_id` (NOT `id`), `class` (water/crops/etc), `label` (0)
  - NOTE: uses `tile_id` not `id` — the pipeline's `_load_label_file` expects `id`. This file needs the column renamed or the loader needs to handle `tile_id`.

- **Augmented labels** (with hard negatives from iteration 2):
  `~/epoch/geovibes/augmented_dataset_20260120_152630.geojson`
  - Properties: `id`, `tile_id`, `label`, `class`, `source`, `embedding`

### Ground Truth
- **Trase Universal Mill List** (867 known palm oil mills in Sumatra):
  `~/epoch/geovibes/geometries/sumatra-palm-oil-mills.geojson`
  - Properties: `name`, `trase_code`, `uml_id`, `group`, `company`, `mill_name`, `latitude`, `longitude`, `capacity_tonnes_ffb_hour`, `active`

### Classification Outputs (for reference)
- `~/epoch/geovibes/classification_output_linear_pools/` — Linear SVM outputs with DBSCAN clustering
- `~/epoch/geovibes/classification_output_xgb_pools_v2/` — XGBoost v2 outputs

## E2E Test Steps

### Step 1: Create Project and Link Database

```bash
cd ~/projects/geovibes
uv run geovibes init sumatra-e2e \
  --bbox "95,-6,106,6" \
  --years 2024 \
  --res 80 \
  --target "palm oil mill POME lagoons"
```

Then symlink the pre-built database (skip the slow `geovibes load` step):

```bash
cd sumatra-e2e
ln -sf ~/epoch/geovibes/sumatra_db/sumatra_metadata.db database/embeddings.db
ln -sf ~/epoch/geovibes/sumatra_db/sumatra_faiss.index database/embeddings.index
```

Verify the database is accessible:
```bash
uv run python -c "
import duckdb
conn = duckdb.connect('database/embeddings.db', read_only=True)
conn.execute('INSTALL spatial; LOAD spatial;')
print(conn.execute('SELECT COUNT(*) FROM geo_embeddings').fetchone())
conn.close()
"
```

Expected: `(75312294,)`

### Step 2: Launch TUI and Verify Queue Screen

```bash
uv run geovibes tui
```

- Press `1` for Queue screen — should show the `init` job
- Press `q` to quit

### Step 3: Train Iteration 1 (Linear SVM)

Use the clean labelled dataset (116 positives + negatives) combined with sampled negatives:

**IMPORTANT**: The `sampled_negatives.geojson` file uses `tile_id` instead of `id`. Before training, either:
a) Rename the column in the file, OR
b) Modify `_load_label_file` in `pipeline.py` to accept `tile_id` as an alias for `id`

Option (b) is better. In `geovibes/cli/pipeline.py`, in the `_load_label_file` function, after loading the GeoJSON, add:
```python
if "id" not in df.columns and "tile_id" in df.columns:
    df["id"] = df["tile_id"]
```

Then train:

```bash
uv run geovibes train \
  --positives ~/epoch/geovibes/labeled_dataset_20260120_132604_clean.geojson \
  --negatives ~/epoch/geovibes/sampled_negatives.geojson \
  --classifier linear-svm \
  --threshold 0.5
```

Expected output:
- Training data: ~116 positive, ~3499 negative (after merge with DB embeddings)
- Train time: ~1-2 seconds
- F1: ~0.9+ on test set
- Model saved to `models/linear-svm_v0.pkl`

### Step 4: Run Inference

```bash
uv run geovibes infer --threshold 0.5
```

This scores all 75M embeddings. Expected:
- Runtime: ~2-4 minutes on M1 Mac with 32GB RAM
- Detections: ~100,000+ above threshold (first iteration is noisy — this is expected)
- Detections saved to `detections/iteration_000.parquet`
- `reviews.parquet` populated with all detections as `status=pending`

### Step 5: Cluster Detections

```bash
uv run geovibes cluster --eps 500 --min-samples 2
```

Expected:
- Reduces ~100K detections to ~2,000-5,000 clusters
- Updates `reviews.parquet` with `cluster_id`

### Step 6: Review Detections in TUI

```bash
uv run geovibes tui
```

- Press `1` — Queue screen should show init, train, infer, cluster jobs
- Press `2` — Review screen
  - Should show detection #1 with satellite tile (Google Hybrid), score, cluster ID, coordinates
  - Press `a` to accept, `r` to reject, `s` to skip
  - Arrow keys to navigate
  - Press `p` to toggle pending-only filter
  - Review ~20-50 detections to identify false positives
- Press `q` to quit

**Key things to verify:**
- Satellite tile images load (requires network access, Google Hybrid is unauthenticated)
- Score values are similarity scores (higher = more similar, via `1/(1+dist)`)
- Cluster IDs display correctly
- Accept/reject persists to `reviews.parquet`
- Progress bar updates

### Step 7: Validate Against Ground Truth

This step requires implementing `geovibes validate`. Reference implementation at `~/epoch/geovibes/scripts/validate_detections.py`.

The validation should:
1. Load detections (or clustered centroids)
2. Load truth GeoJSON (`~/epoch/geovibes/geometries/sumatra-palm-oil-mills.geojson`)
3. For each truth point, check if any detection is within `--buffer` meters
4. Compute precision, recall, F1

Expected results (from the blog): ~70% recall at 500m buffer with 2 iterations.

### Step 8: Iteration 2 — Hard Negative Mining (Optional)

If the TUI Search screen is working:
1. Press `3` for Search screen
2. Label some of the false positive detections as negatives
3. Press `e` to export labels
4. Re-train with augmented labels:
   ```bash
   uv run geovibes train \
     --positives ~/epoch/geovibes/labeled_dataset_20260120_132604_clean.geojson \
     --positives labels/iteration_001.parquet \
     --negatives ~/epoch/geovibes/sampled_negatives.geojson \
     --classifier linear-svm \
     --threshold 0.5
   ```
5. Re-infer and re-cluster
6. Detection count should drop ~10x

## Known Issues to Watch For

1. **`sampled_negatives.geojson` uses `tile_id` not `id`** — needs column rename or loader fix
2. **Label `id` is string in GeoJSON, int in DuckDB** — `_load_label_file` loads GeoJSON ids as strings; `_fetch_all_embeddings` may fail on the join. Cast to int.
3. **75M embeddings at 64 dimensions** — inference takes ~2-4 min. The batch INSERT in loader is row-by-row (slow for this size, but we're symlink-ing, not loading).
4. **Tile images in TUI** — depend on terminal supporting Kitty/Sixel/halfblock via `textual-image`. Test in iTerm2 or WezTerm for best results. In a basic terminal, you'll see `Tile loaded (N bytes)` instead of the image.
5. **Memory** — 75M x 64-dim float32 = ~18GB when all loaded. DuckDB streams in batches of 100K, so should be fine with 32GB RAM.

## Success Criteria

The e2e test passes if:
- [ ] `geovibes init` creates project structure
- [ ] Database symlinks work and CLI can query them
- [ ] `geovibes train` produces a model with F1 > 0.9
- [ ] `geovibes infer` scores all 75M embeddings without OOM
- [ ] `geovibes cluster` reduces detections to ~2-5K clusters
- [ ] TUI Queue screen shows all jobs with correct status
- [ ] TUI Review screen displays tiles and allows accept/reject/skip
- [ ] Accept/reject persists across TUI sessions
- [ ] Jobs ledger (`jobs.parquet`) has complete audit trail
