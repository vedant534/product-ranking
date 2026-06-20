"""Tests for the raw RetailRocket inspection script."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.check_raw_data import check_raw_data, verify_required_files


@pytest.fixture
def raw_data_dir(
    tmp_path: Path,
    repository_root: Path,
) -> Path:
    data_dir = tmp_path / "retailrocket"
    data_dir.mkdir()
    sample_dir = repository_root / "data" / "sample"

    shutil.copy(sample_dir / "events_sample.csv", data_dir / "events.csv")
    shutil.copy(sample_dir / "category_tree_sample.csv", data_dir / "category_tree.csv")

    properties = pd.read_csv(sample_dir / "item_properties_sample.csv")
    properties.iloc[:7].to_csv(data_dir / "item_properties_part1.csv", index=False)
    properties.iloc[7:].to_csv(data_dir / "item_properties_part2.csv", index=False)
    return data_dir


def test_check_raw_data_prints_requested_statistics(
    raw_data_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    warnings = check_raw_data(raw_data_dir, sample_n=2)

    output = capsys.readouterr().out
    assert warnings == []
    assert "events.csv found" in output
    assert "item_properties_part1.csv found" in output
    assert "item_properties_part2.csv found" in output
    assert "category_tree.csv found" in output
    assert "All headers contain the expected columns" in output
    assert "first 2 rows from events.csv only" in output
    assert "row count (sampled): 2" in output
    assert "column names: ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']" in output
    assert "timestamp range: 2024-01-01 00:00:00+00:00 to 2024-01-01 00:01:00+00:00" in output
    assert "unique visitors: 2" in output
    assert "unique items: 2" in output
    assert "event counts:" in output
    assert "view: 2" in output
    assert "item_properties_part1.csv row count: 7" in output
    assert "item_properties_part2.csv row count: 7" in output
    assert "combined row count: 14" in output
    assert "top property names:" in output
    assert "category_tree.csv\n  row count: 10" in output
    assert "unique categories: 10" in output
    assert "Warnings: none." in output


def test_verify_required_files_lists_every_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error_info:
        verify_required_files(tmp_path)

    message = str(error_info.value)
    assert "events.csv" in message
    assert "item_properties_part1.csv" in message
    assert "item_properties_part2.csv" in message
    assert "category_tree.csv" in message


def test_check_raw_data_rejects_bad_header_before_loading(raw_data_dir: Path) -> None:
    (raw_data_dir / "category_tree.csv").write_text("categoryid\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="category_tree.csv is missing expected column.*parentid"):
        check_raw_data(raw_data_dir, sample_n=2)


def test_check_raw_data_warns_about_unexpected_events_and_null_ids(
    raw_data_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events_path = raw_data_dir / "events.csv"
    events = pd.read_csv(events_path)
    events.loc[0, "event"] = "wishlist"
    events.loc[0, "visitorid"] = None
    events.to_csv(events_path, index=False)

    warnings = check_raw_data(raw_data_dir, sample_n=2)

    output = capsys.readouterr().out
    assert any("Unexpected event values: ['wishlist']" in warning for warning in warnings)
    assert any("visitorid contains 1 nulls" in warning for warning in warnings)
    assert "WARNINGS:" in output
