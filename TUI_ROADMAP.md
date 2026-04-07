# GeoVibes CLI/TUI — Roadmap

> Branch: `jake/tui` (15 commits, 29 tests)
> Plan: `.claude/plans/polished-coalescing-eagle.md`

## What's Built (PoC)

The PoC implements the full CLI + Textual TUI skeleton from the plan:

```
geovibes init   → project directory + geovibes.yaml
geovibes load   → aef-loader COGs → DuckDB + FAISS (Source Cooperative)
geovibes train  → Linear SVM or XGBoost on labelled parquet/GeoJSON
geovibes infer  → batch score all embeddings, populate reviews ledger
geovibes cluster → DBSCAN with haversine distance
geovibes tui    → 3 screens: Queue (1), Review (2), Search (3)
```

Persistent parquet ledger (jobs, reviews, comments, labels) with file locking.
Reviewed through 8 roborev iterations.

## Remaining Work

### 1. Review Backlog (from roborev findings)

Issues identified in the final branch review that were deferred:

| Finding | Severity | File | Description |
|---------|----------|------|-------------|
| Degree-based distance | Medium | `search.py:27` | `ST_Distance` returns degrees on WGS84; use `ST_Distance_Sphere` for meters per CLAUDE.md |
| Cluster overwrites all | High | `pipeline.py:270` | `run_cluster` resets cluster_id for the entire reviews file; should scope to current iteration only |
| Point ID type safety | Medium | `pipeline.py` | `_fetch_all_embeddings` should cast point_ids to `int` to fail fast on bad label data |
| Review job lifecycle | Low | `review.py:76` | `_start_review_job` creates a new job on every mount; should create once and complete on exit |
| PQ fallback for prime dims | Low | `loader.py:169` | When `dim` is prime, `m=1` gives poor quantization; consider `IndexIVFFlat` fallback |
| Bulk insert perf | Low | `loader.py:132` | Row-by-row DuckDB inserts are slow; use `INSERT INTO ... SELECT` from pyarrow |
| Pipeline test coverage | Low | `tests/` | `run_train`, `run_infer`, `run_cluster` have no tests |

### 2. End-to-End Validation

Test the full workflow with real data before building more features.

- **Symlink or copy** the Sumatra database from `~/epoch/geovibes/sumatra_db/` into a test project
- Run: `init → (symlink DB) → tui search → label 50 seeds → train → infer → cluster → tui review`
- Validate against Trase UML ground truth using `geovibes validate`
- Document any data format mismatches or UX issues

### 3. Missing CLI Commands

From the original plan, not yet implemented:

- **`geovibes validate`** — Compare clustered detections against a ground truth GeoJSON with configurable buffer distance. Output precision/recall/F1. The epoch fork has `scripts/validate_detections.py` as reference.

### 4. TUI Gaps

- **Comment input dialog** — `c` key in review screen is stubbed. Need a Textual `Input` modal that writes to `comments.parquet` via `add_comment()`.
- **Search screen coordinate input** — Currently can only label FAISS results. Need a way to seed initial points (lat/lon input or load from file).
- **Tile image quality** — `textual-image` renders via Kitty/Sixel/halfblock. Test on different terminals (iTerm2, WezTerm, kitty) and assess if the tile quality is sufficient for labelling decisions.

### 5. Linear SVM Classifier

The epoch fork (`~/epoch/geovibes/geovibes/classification/classifier.py`) has `LinearSVMClassifier` with:
- `CalibratedClassifierCV` for probability estimates
- `get_coefficients()` for extracting the learned direction vector
- `get_normalized_direction()` for cosine similarity

The PoC's `pipeline.py` already supports `--classifier linear-svm` using raw sklearn, but should be upgraded to the epoch fork's version which supports coefficient extraction (needed for exporting the vector back to Earth Engine).

### 6. Agentic Review (Phase 4 from plan)

The epoch fork has a complete LangGraph verification agent at `~/epoch/geovibes/geovibes/agents/`:

- `verification_agent.py` — Multi-step pipeline: satellite image analysis (Gemini vision) → Google Places lookup → second verification with context → facility enrichment
- `clustering.py` — DBSCAN clustering of detections
- `places.py` — Google Places API integration
- `thumbnail.py` — Satellite image extraction for agent input
- `pipeline.py` — End-to-end orchestration

Integration plan:
1. Add `geovibes verify` CLI command that queues detections for agent processing
2. Agent writes structured comments to `comments.parquet` with reasoning
3. Agent sets review status (accepted/rejected) with `reviewer=agent:gemini`
4. Human spot-check screen in TUI that shows agent decisions + reasoning
5. Confidence-gated routing: score > 0.9 → auto-accept pool, 0.5-0.9 → human queue, < 0.5 → auto-reject

### 7. Package Extraction

The plan calls for building in `geovibes/cli/` with clean boundaries, then extracting to a standalone package. Dependencies to decouple:

- `geovibes.ui.xyz` — tile fetching functions should be copied into `cli/tiles.py`
- `geovibes.ui_config` — `BasemapConfig.BASEMAP_TILES` constant used for Google Hybrid URL
- Own `pyproject.toml` with minimal deps: `click`, `textual`, `textual-image`, `duckdb`, `faiss-cpu`, `scikit-learn`, `aef-loader`, `pyarrow`, `shapely`, `pillow`

### 8. Data Loading Improvements

The `geovibes load` command currently:
- Fetches all tiles for the bbox at a given IFD resolution
- Reprojects to EPSG:4326
- Extracts vectors row-by-row into DuckDB

Improvements needed:
- **Bulk insert** via pyarrow table (10-100x faster for large regions)
- **Progress reporting** during aef-loader fetch (currently silent during COG download)
- **Resume support** — if load fails partway, don't re-download tiles
- **Multiple data sources** — the plan calls for `init` then `load` separately to support future sources beyond AEF

## Architecture Reference

See `.claude/plans/polished-coalescing-eagle.md` for the full architecture including:
- Parquet schema definitions (jobs, reviews, comments, labels)
- Project directory structure
- TUI screen layouts
- Roborev-inspired design patterns (persistent ledger, confidence-gated routing, iterative refine)
