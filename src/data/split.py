"""Chronologically split complete sessions into train, validation, and test."""

import json
import math
from numbers import Real
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

REQUIRED_COLUMNS = ("session_id", "visitorid", "timestamp", "itemid")
SPLIT_NAMES = ("train", "validation", "test")


def _validate_input(events: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(
            "Sessionized dataframe is missing required column(s): "
            f"{', '.join(missing)}. Required columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    null_required = [column for column in REQUIRED_COLUMNS if events[column].isna().any()]
    if null_required:
        raise ValueError(
            "Sessionized dataframe contains null values in required column(s): "
            f"{', '.join(null_required)}"
        )


def _validate_fractions(
    train_fraction: Real,
    validation_fraction: Real,
    test_fraction: Real,
) -> None:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(
        isinstance(fraction, bool) or not isinstance(fraction, Real) or fraction < 0
        for fraction in fractions
    ):
        raise ValueError("Split fractions must be non-negative numbers")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.0")


def _normalize_timestamps(timestamps: pd.Series) -> pd.Series:
    try:
        if is_datetime64_any_dtype(timestamps.dtype):
            return pd.to_datetime(timestamps, utc=True, errors="raise")
        if is_numeric_dtype(timestamps.dtype):
            return pd.to_datetime(timestamps, unit="ms", utc=True, errors="raise")
        return pd.to_datetime(timestamps, utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Sessionized dataframe has invalid timestamps; numeric timestamps "
            "must be Unix milliseconds"
        ) from error


def _range(start: object, end: object) -> dict[str, object]:
    return {"start": start, "end": end}


def summarize_splits(
    split_events: pd.DataFrame,
    session_starts: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Create per-split date ranges and cardinality statistics."""
    summary = {}
    for split_name in SPLIT_NAMES:
        events = split_events[split_events["split"] == split_name]
        starts = session_starts[session_starts["split"] == split_name]["session_start_time"]
        if events.empty:
            event_range = _range(None, None)
            session_start_range = _range(None, None)
        else:
            event_range = _range(events["timestamp"].min(), events["timestamp"].max())
            session_start_range = _range(starts.min(), starts.max())

        summary[split_name] = {
            "date_range": event_range,
            "session_start_time_range": session_start_range,
            "number_of_sessions": int(events["session_id"].nunique()),
            "number_of_events": int(len(events)),
            "number_of_unique_items": int(events["itemid"].nunique()),
            "number_of_unique_visitors": int(events["visitorid"].nunique()),
        }
    return summary


def save_split_summary(
    summary: dict[str, dict[str, object]],
    output_path: Path,
    purge_summary: object = None,
) -> Path:
    """Persist split diagnostics with timestamps encoded as ISO-8601 strings."""
    def json_default(value: object) -> str:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        raise TypeError(f"Unsupported split summary value: {type(value).__name__}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": summary, "purge": purge_summary or {}}
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    return output_path


def chronological_split(
    sessionized_events: pd.DataFrame,
    train_fraction: Real = 0.70,
    validation_fraction: Real = 0.15,
    test_fraction: Real = 0.15,
    purge_overlapping_sessions: bool = True,
) -> pd.DataFrame:
    """Assign complete sessions chronologically and optionally purge boundary overlap.

    Purging removes sessions from the earlier split when their final event reaches
    or crosses the next split's start boundary. This preserves complete retained
    sessions while ensuring strict event-time ordering between splits.
    """
    _validate_input(sessionized_events)
    _validate_fractions(train_fraction, validation_fraction, test_fraction)

    result = sessionized_events.copy()
    result["timestamp"] = _normalize_timestamps(result["timestamp"])

    session_starts = (
        result.groupby("session_id", sort=False, as_index=False)["timestamp"]
        .agg(session_start_time="min", session_end_time="max")
        .sort_values(
            ["session_start_time", "session_id"],
            kind="mergesort",
            ignore_index=True,
        )
    )

    session_count = len(session_starts)
    train_end = round(session_count * float(train_fraction))
    validation_end = round(
        session_count * float(train_fraction + validation_fraction)
    )
    session_starts["split"] = (
        ["train"] * train_end
        + ["validation"] * (validation_end - train_end)
        + ["test"] * (session_count - validation_end)
    )

    purge_summary = {
        "enabled": bool(purge_overlapping_sessions),
        "initial_number_of_sessions": session_count,
        "purged_train_sessions": 0,
        "purged_validation_sessions": 0,
        "retained_number_of_sessions": session_count,
        "validation_boundary": None,
        "test_boundary": None,
    }
    if purge_overlapping_sessions and session_count:
        validation_starts = session_starts.loc[
            session_starts["split"] == "validation", "session_start_time"
        ]
        test_starts = session_starts.loc[
            session_starts["split"] == "test", "session_start_time"
        ]
        purged = pd.Series(False, index=session_starts.index)
        if not validation_starts.empty:
            validation_boundary = validation_starts.min()
            train_overlap = (session_starts["split"] == "train") & session_starts[
                "session_end_time"
            ].ge(validation_boundary)
            purged |= train_overlap
            purge_summary["validation_boundary"] = validation_boundary
            purge_summary["purged_train_sessions"] = int(train_overlap.sum())
        if not test_starts.empty:
            test_boundary = test_starts.min()
            validation_overlap = (
                (session_starts["split"] == "validation")
                & session_starts["session_end_time"].ge(test_boundary)
            )
            purged |= validation_overlap
            purge_summary["test_boundary"] = test_boundary
            purge_summary["purged_validation_sessions"] = int(validation_overlap.sum())
        session_starts = session_starts.loc[~purged].copy()
        purge_summary["retained_number_of_sessions"] = int(len(session_starts))

    split_by_session = session_starts.set_index("session_id")["split"]
    result["split"] = result["session_id"].map(split_by_session)
    result = result[result["split"].notna()].copy()
    result.attrs["split_summary"] = summarize_splits(result, session_starts)
    result.attrs["purge_summary"] = purge_summary
    return result
