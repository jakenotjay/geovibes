"""Tests for CLI project initialization and configuration."""

import yaml
from pathlib import Path

from geovibes.cli.project import (
    init_project,
    load_project,
    save_project,
    parse_bbox,
    parse_years,
    CONFIG_FILE,
    DIRS,
)


def test_parse_bbox():
    result = parse_bbox("95.0,-6.0,106.0,0.0")
    assert result == [95.0, -6.0, 106.0, 0.0]


def test_parse_bbox_with_spaces():
    result = parse_bbox("95.0, -6.0, 106.0, 0.0")
    assert result == [95.0, -6.0, 106.0, 0.0]


def test_parse_years():
    result = parse_years("2023,2024")
    assert result == [2023, 2024]


def test_init_project_creates_structure(tmp_path):
    project_dir = init_project(
        name="test-project",
        bbox=[95.0, -6.0, 106.0, 0.0],
        years=[2024],
        resolution=80,
        target="palm oil mills",
        base_dir=tmp_path,
    )

    assert project_dir.exists()
    assert (project_dir / CONFIG_FILE).exists()
    for d in DIRS:
        assert (project_dir / d).is_dir()


def test_init_project_writes_correct_config(tmp_path):
    project_dir = init_project(
        name="test-project",
        bbox=[95.0, -6.0, 106.0, 0.0],
        years=[2024],
        resolution=80,
        target="palm oil mills",
        base_dir=tmp_path,
    )

    with open(project_dir / CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    assert config["name"] == "test-project"
    assert config["target"] == "palm oil mills"
    assert config["bbox"] == [95.0, -6.0, 106.0, 0.0]
    assert config["years"] == [2024]
    assert config["resolution"] == 80
    assert config["tile_source"] == "GOOGLE_HYBRID"
    assert config["iteration"] == 0
    assert config["database"]["path"] == "database/embeddings.db"


def test_load_project(tmp_path):
    project_dir = init_project(
        name="test-project",
        bbox=[1.0, 2.0, 3.0, 4.0],
        years=[2024],
        base_dir=tmp_path,
    )

    config = load_project(project_dir)
    assert config["name"] == "test-project"
    assert config["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert config["_project_dir"] == project_dir


def test_save_project_roundtrip(tmp_path):
    project_dir = init_project(
        name="test-project",
        bbox=[1.0, 2.0, 3.0, 4.0],
        years=[2024],
        base_dir=tmp_path,
    )

    config = load_project(project_dir)
    config["iteration"] = 3
    save_project(config)

    reloaded = load_project(project_dir)
    assert reloaded["iteration"] == 3


def test_load_project_finds_parent(tmp_path):
    project_dir = init_project(
        name="test-project",
        bbox=[1.0, 2.0, 3.0, 4.0],
        years=[2024],
        base_dir=tmp_path,
    )

    sub_dir = project_dir / "labels"
    config = load_project(sub_dir)
    assert config["name"] == "test-project"
