"""Project initialization and configuration management."""

from pathlib import Path
from typing import Dict, List, Optional

import yaml


DIRS = ["database", "labels", "models", "detections"]
CONFIG_FILE = "geovibes.yaml"


def parse_bbox(bbox_str: str) -> List[float]:
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values (minx,miny,maxx,maxy), got {len(parts)}")
    return parts


def parse_years(years_str: str) -> List[int]:
    return [int(y.strip()) for y in years_str.split(",")]


def init_project(
    name: str,
    bbox: List[float],
    years: List[int],
    resolution: int = 80,
    target: str = "",
    base_dir: Optional[Path] = None,
) -> Path:
    project_dir = (base_dir or Path.cwd()) / name
    project_dir.mkdir(parents=True, exist_ok=False)

    for d in DIRS:
        (project_dir / d).mkdir()

    config = {
        "name": name,
        "target": target,
        "bbox": bbox,
        "years": years,
        "resolution": resolution,
        "database": {
            "path": "database/embeddings.db",
            "faiss_path": "database/embeddings.index",
        },
        "tile_source": "GOOGLE_HYBRID",
        "iteration": 0,
    }

    config_path = project_dir / CONFIG_FILE
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return project_dir


def load_project(path: Optional[Path] = None) -> Dict:
    project_dir = _find_project_root(path or Path.cwd())
    config_path = project_dir / CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"No {CONFIG_FILE} found in {project_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["_project_dir"] = project_dir
    return config


def save_project(config: Dict) -> None:
    project_dir = config.pop("_project_dir", None)
    if project_dir is None:
        raise ValueError("Config missing _project_dir; use load_project() first")

    config_path = Path(project_dir) / CONFIG_FILE
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    config["_project_dir"] = project_dir


def _find_project_root(start: Path) -> Path:
    current = start.resolve()
    for p in [current, *current.parents]:
        if (p / CONFIG_FILE).exists():
            return p
    raise FileNotFoundError(
        f"No {CONFIG_FILE} found in {start} or any parent directory"
    )
