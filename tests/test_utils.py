import json
import logging

import pytest

from utils.io import read_json, write_json
from utils.logging import get_logger


def test_json_artifact_round_trip(tmp_path):
    path = tmp_path / "nested" / "artifact.json"
    payload = {"items": [3, 1], "name": "test"}

    assert write_json(payload, path) == path
    assert read_json(path) == payload
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_read_json_reports_missing_path(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="JSON artifact not found"):
        read_json(path)


def test_read_json_reports_malformed_artifact(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="is malformed"):
        read_json(path)


def test_get_logger_returns_named_logger():
    assert isinstance(get_logger("session-ranking.test"), logging.Logger)
    assert get_logger("session-ranking.test").name == "session-ranking.test"
