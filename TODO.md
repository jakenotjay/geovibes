# GeoVibes TUI — Active TODO

Branch: `jake/tui`. See `TUI_ROADMAP.md` for the longer-form context. This doc tracks the **current** work streams.

## Done

- ✅ Roborev review backlog closed: jobs 610 (compact), 612 (full), 613, 614, 615, 616, 617. Net effect: pipeline iteration scoping (`run_infer`/`run_cluster`), UTM projection in `run_validate`, ST_Distance_Sphere, IndexIVFFlat fallback, tile-cache thread safety, idempotent review-job lifecycle, undo audit trail, queue filter binding, NaN class detection, schema-drift error w/ migration guidance, ledger sentinel→explicit reviewed_at.
- ✅ Search screen cut from TUI app (key `3` removed, files left in tree).
- ✅ Queue filter binding (`f`) wired to a working status-cycle.

## Cuts / parking lot

### Deep rendering review — deferred
The TGP / halfcell tile pipeline in `ReviewScreen` works but has accumulated a lot of churn (~15 commits). We want to be **decisive about look and feel now** and do the deep rendering rework later.
- [ ] Lock the visual contract for the review page (panel layout, colors, info density, progress bar).
- [ ] Capture the agreed look in a screenshot or markdown spec under `docs/tui-look.md` so future rendering work is graded against it, not "looks fine."
- [ ] Schedule the deep rendering review after the look-and-feel is locked.

## Active work streams

### 1. Render embedding bounding boxes on the review page
Each detection corresponds to an 80m×80m AEF tile (or whatever resolution the project is using). The review tile shows a 3×3 mosaic of GOOGLE_HYBRID 256px tiles centered on the embedding centroid; we currently show no indication of which pixels the embedding actually covers.
- [ ] Compute the embedding tile bbox from `(lat, lon, resolution_m)` (resolution comes from `geovibes.yaml`).
- [ ] Convert bbox corners to pixel coordinates inside the rendered mosaic before TGP transmission.
- [ ] Draw a 1–2px outline (PIL `ImageDraw.rectangle`) on the mosaic before transmit.
- [ ] Make sure halfcell fallback also gets the overlay.
- [ ] Test at multiple resolutions (10/20/40/80m) — the box should stay visible at 80m and not dominate at 10m.

### 2. `geovibes verify` — agentic review (full feature)
Going beyond the roadmap stub. We want the agent to be properly briefed and pluggable.
- [ ] Prompt input: each project needs an **example image** (or set) + **text description** of the target. Add to `geovibes.yaml`:
  - `verify.target_description: str`
  - `verify.example_images: [path]`
  - `verify.negative_examples: [path]` (optional)
- [ ] Pluggable agent backends:
  - Gemini vision (epoch fork's existing impl, port from `~/epoch/geovibes/geovibes/agents/`).
  - Claude with vision (via `claude-api` skill, prompt caching for the example images).
  - **Local coding agent pickup** — roborev-style: detect a running local agent and route verification jobs through it. Investigate roborev's local-agent-discovery mechanism for the pattern.
- [ ] CLI: `geovibes verify [--agent gemini|claude|local] [--limit N]` queues pending detections and writes structured comments + status to the ledger.
- [ ] Confidence-gated routing: `score > 0.9` → auto-accept pool, `0.5–0.9` → human queue, `< 0.5` → auto-reject.
- [ ] TUI surface: review screen should show agent reasoning in the comments panel; agent verdicts should be visually distinct from human ones.

### 3. LinearSVM → Earth Engine export
Upgrade `pipeline.py`'s linear-svm branch to expose the learned direction vector + intercept, then auto-render an EE script and open it.
- [ ] Replace raw sklearn LinearSVC + CalibratedClassifierCV with epoch fork's `LinearSVMClassifier` (`~/epoch/geovibes/geovibes/classification/classifier.py`) so we get `get_coefficients()` and `get_normalized_direction()`.
- [ ] After train, write `models/linear-svm_v{N}.json` with `{w: [...], b: float, n_pos, n_neg, f1, auc, target_description}`.
- [ ] Add `geovibes export-ee [--model PATH] [--center lat,lon] [--zoom N] [--lo F] [--hi F]`:
  - Render the EE JS template (see `docs/ee_template.js`) with the model's `w` and `b` baked in.
  - Build the URL fragment string (`title=...;year=...;lat=...;lon=...;zoom=...;lo=...;hi=...;w=...;b=...`).
  - URL-encode and concatenate to `https://code.earthengine.google.com/<paste-id>?code#<fragment>`.
  - Decision needed: do we paste the script body to a new EE snippet (unauthenticated `gist`-style), or do we copy the loader URL to the clipboard and require the user to paste the script body once into a saved Code Editor file?
  - Open the resulting URL in the browser via `webbrowser.open`.
- [ ] Center default = project bbox centroid; lo/hi default = score percentile (e.g. p1/p99 from a sample of inferred scores).

### 4. Project manager page in the TUI
A new top-level screen (key `0` or before `1`) listing all known projects, not just the one we launched into.
- [ ] Maintain a registry at `~/.geovibes/projects.json` updated by every `geovibes init` and `geovibes load`.
  - Schema: `{path, name, target, bbox, years, resolution, last_seen, status_summary}`.
- [ ] On TUI start (and via a `r` refresh binding), validate each registered path:
  - Path exists and contains `geovibes.yaml`?
  - If yes → live; refresh `last_seen` and pull a quick summary (jobs count, detections count, latest iteration).
  - If no → mark `missing`.
- [ ] **Tracking moved projects**: this is non-trivial. Options:
  - (a) Best-effort only — show as missing, let the user `geovibes register PATH` to update.
  - (b) On launch, write a unique `.geovibes-id` file inside the project dir; the registry indexes by id, so a `find ~ -name .geovibes-id` rescan can re-anchor moved dirs.
  - (c) Watch the user's home dir with `fsevents` / `inotify` — too complex for now.
  - **Decision:** start with (a) + (b). Add `geovibes scan [--root PATH]` to walk and refresh the registry by id.
- [ ] Project page columns: name, path, target, resolution, jobs, detections (A/R/P), last activity, status (live/missing).
- [ ] Selecting a row + Enter switches `App.config` and `App.project_dir` to that project and pushes the queue screen.

### 5. End-to-end fresh project run
Once the cleanup above lands, run the workflow from zero on a new region — not the symlinked Sumatra DB.
- [ ] Pick a small bbox (small enough that `load` finishes in < 1h on the target machine).
- [ ] `geovibes init` → `geovibes load` → seed labels → `train` → `infer` → `cluster` → `tui review`.
- [ ] **Loader UX questions to resolve during this run:**
  - How do we actually invoke the load? Document `geovibes load --source source-coop` in `docs/loader.md`.
  - Progress meter: `aef-loader` is silent during COG download. We currently print `"Downloading AEF index..."`, `"Querying tiles..."`, `"Opening tiles..."` and a per-batch DuckDB insert counter. The COG fetch + reproject step is the long pole and has no progress.
  - Add tqdm-style progress reporting around `reproject_datatree` / the tile fetch. May need to pass a callback into `aef-loader` or wrap the underlying iterator.
  - Resume support — if load fails partway, don't re-download tiles. Probably needs aef-loader changes.
- [ ] Bulk-insert path (pyarrow `INSERT INTO ... SELECT` instead of `executemany`) — currently flagged in the roadmap as a perf TODO; this run will tell us if it's a blocker or a nice-to-have.
- [ ] Document any rough edges in this file as we go.

## Reference

- Earth Engine loader template: `docs/ee_template.js`
- Existing roadmap (longer-form, includes deferred roborev findings): `TUI_ROADMAP.md`
- E2E test plan against Sumatra ground truth: `E2E_TEST_PLAN.md`
