"""Repository metadata and configuration smoke tests."""

from pathlib import Path

import yaml


def test_default_config_requires_chronological_split(repository_root: Path) -> None:
    config_path = repository_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["split"]["strategy"] == "chronological"
    assert sum(
        config["split"][key]
        for key in ("train_fraction", "validation_fraction", "test_fraction")
    ) == 1.0


def test_required_dataset_paths_are_ignored(repository_root: Path) -> None:
    gitignore = (repository_root / ".gitignore").read_text(encoding="utf-8")
    assert "/data/raw/*" in gitignore
    assert "/data/interim/*" in gitignore
    assert "/data/processed/*" in gitignore
    assert "/models/*" in gitignore
    assert "/artifacts/*" in gitignore
