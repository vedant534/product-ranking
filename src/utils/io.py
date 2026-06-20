"""Small, shared helpers for JSON artifacts."""

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read a JSON artifact with path-specific failures."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"JSON artifact not found: '{path}'") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON artifact '{path}' is malformed: {error}") from error


def write_json(payload: Any, path: Path) -> Path:
    """Atomically write a deterministic, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path
