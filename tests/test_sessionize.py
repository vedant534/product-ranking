"""Tests for anonymous RetailRocket event sessionization."""

import json
from pathlib import Path

import pandas as pd
import pytest

from data.sessionize import OUTPUT_COLUMNS, save_session_summary, sessionize_events
from scripts.prepare_data import run_sessionization_step


def _events(rows: list[tuple[object, int, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "visitorid": visitorid,
                "event": event,
                "itemid": itemid,
                "transactionid": None,
            }
            for timestamp, visitorid, event, itemid in rows
        ]
    )


def test_same_visitor_splits_after_more_than_thirty_minutes() -> None:
    events = _events(
        [
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:10:00+00:00", 1, "view", 102),
            ("2024-01-01 00:41:00+00:00", 1, "view", 103),
        ]
    )

    result = sessionize_events(events, min_session_length=1)

    assert result["session_id"].tolist() == [0, 0, 1]


def test_same_visitor_does_not_split_within_or_at_thirty_minutes() -> None:
    events = _events(
        [
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:30:00+00:00", 1, "view", 102),
            ("2024-01-01 00:59:00+00:00", 1, "view", 103),
        ]
    )

    result = sessionize_events(events)

    assert result["session_id"].nunique() == 1


def test_different_visitors_never_share_a_session() -> None:
    events = _events(
        [
            ("2024-01-01 00:05:00+00:00", 2, "view", 201),
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:10:00+00:00", 2, "view", 202),
            ("2024-01-01 00:05:00+00:00", 1, "view", 102),
        ]
    )

    result = sessionize_events(events)

    visitors_per_session = result.groupby("session_id")["visitorid"].nunique()
    assert result["session_id"].nunique() == 2
    assert visitors_per_session.eq(1).all()


def test_session_position_starts_at_zero_and_increments() -> None:
    events = _events(
        [
            ("2024-01-01 00:10:00+00:00", 1, "view", 103),
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:05:00+00:00", 1, "addtocart", 102),
        ]
    )

    result = sessionize_events(events)

    assert tuple(result.columns) == OUTPUT_COLUMNS
    assert result["itemid"].tolist() == [101, 102, 103]
    assert result["session_position"].tolist() == [0, 1, 2]


def test_short_sessions_are_dropped_and_session_ids_are_compact() -> None:
    events = _events(
        [
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:00:00+00:00", 2, "view", 201),
            ("2024-01-01 00:05:00+00:00", 2, "view", 202),
        ]
    )

    result = sessionize_events(events)

    assert result["visitorid"].tolist() == [2, 2]
    assert result["session_id"].tolist() == [0, 0]
    assert result["session_position"].tolist() == [0, 1]


def test_allowed_events_are_filtered_before_sessionization() -> None:
    events = _events(
        [
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:05:00+00:00", 1, "wishlist", 102),
            ("2024-01-01 00:10:00+00:00", 1, "transaction", 103),
        ]
    )

    result = sessionize_events(events)

    assert result["event"].tolist() == ["view", "transaction"]
    assert result["session_position"].tolist() == [0, 1]


def test_numeric_timestamps_are_interpreted_as_unix_milliseconds() -> None:
    events = _events(
        [
            (1704067200000, 1, "view", 101),
            (1704067500000, 1, "view", 102),
        ]
    )

    result = sessionize_events(events)

    assert result.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_summary_stats_are_saved_on_the_result() -> None:
    events = _events(
        [
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:05:00+00:00", 1, "addtocart", 102),
            ("2024-01-01 00:00:00+00:00", 2, "view", 201),
            ("2024-01-01 00:05:00+00:00", 2, "view", 202),
            ("2024-01-01 00:10:00+00:00", 2, "transaction", 203),
        ]
    )

    result = sessionize_events(events)
    summary = result.attrs["summary_stats"]

    assert summary == {
        "number_of_sessions": 2,
        "average_session_length": 2.5,
        "median_session_length": 2.5,
        "max_session_length": 3,
        "event_type_distribution": {"addtocart": 1, "transaction": 1, "view": 3},
    }


def test_session_summary_can_be_persisted(tmp_path: Path) -> None:
    events = _events(
        [
            ("2024-01-01 00:00:00+00:00", 1, "view", 101),
            ("2024-01-01 00:05:00+00:00", 1, "view", 102),
        ]
    )
    result = sessionize_events(events)

    output_path = save_session_summary(
        result.attrs["summary_stats"],
        tmp_path / "session_summary.json",
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["number_of_sessions"] == 1
    assert saved["average_session_length"] == 2.0


def test_missing_columns_raise_a_helpful_error() -> None:
    with pytest.raises(ValueError, match="missing required column.*transactionid"):
        sessionize_events(
            pd.DataFrame({"timestamp": [], "visitorid": [], "event": [], "itemid": []})
        )


def test_prepare_data_sessionize_step_reports_valid_sample_sessions(
    repository_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample_dir = repository_root / "data" / "sample"
    config_path = tmp_path / "sessionize.yaml"
    config_path.write_text(
        "paths:\n"
        f"  raw_data: {sample_dir}\n"
        f"  interim_data: {tmp_path / 'interim'}\n"
        "data:\n"
        "  events_file: events_sample.csv\n"
        "  session_gap_minutes: 30\n"
        "  min_session_length: 2\n"
        "  event_types: [view, addtocart, transaction]\n",
        encoding="utf-8",
    )

    sessionized = run_sessionization_step(config_path)
    lengths = sessionized.groupby("session_id").size()

    assert lengths.min() >= 2
    assert sessionized.groupby("session_id")["visitorid"].nunique().eq(1).all()
    assert (
        sessionized.groupby("session_id")["timestamp"]
        .apply(lambda timestamps: timestamps.is_monotonic_increasing)
        .all()
    )
    output = capsys.readouterr().out
    assert output.count("PASS") == 3
    assert (tmp_path / "interim" / "session_summary.json").is_file()
