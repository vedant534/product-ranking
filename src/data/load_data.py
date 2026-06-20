"""Load and validate RetailRocket CSV files."""

import argparse
from collections.abc import Collection, Sequence
from typing import Optional

import pandas as pd

EVENT_COLUMNS = ("timestamp", "visitorid", "event", "itemid", "transactionid")
ITEM_PROPERTY_COLUMNS = ("timestamp", "itemid", "property", "value")
CATEGORY_TREE_COLUMNS = ("categoryid", "parentid")


def _read_csv(
    path: str,
    dataset_name: str,
    required_columns: Collection[str],
    read_dtypes: Optional[dict[str, str]] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """Read a CSV and fail with dataset-specific schema errors."""
    _validate_sample_n(sample_n)
    try:
        frame = pd.read_csv(path, dtype=read_dtypes, nrows=sample_n)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{dataset_name} file not found: '{path}'") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{dataset_name} file is empty and has no header: '{path}'") from error

    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        required = ", ".join(required_columns)
        raise ValueError(
            f"{dataset_name} file '{path}' is missing required column(s): {missing}. "
            f"Required columns: {required}."
        )

    return frame


def _validate_sample_n(sample_n: Optional[int]) -> None:
    if sample_n is not None and (
        isinstance(sample_n, bool) or not isinstance(sample_n, int) or sample_n <= 0
    ):
        raise ValueError("sample_n must be a positive integer or None")


def _convert_timestamp(frame: pd.DataFrame, path: str, dataset_name: str) -> None:
    """Convert an epoch-millisecond timestamp column to timezone-aware UTC."""
    if frame["timestamp"].isna().any():
        raise ValueError(f"{dataset_name} file '{path}' contains null timestamp values")

    try:
        epoch_milliseconds = pd.to_numeric(frame["timestamp"], errors="raise")
        frame["timestamp"] = pd.to_datetime(
            epoch_milliseconds,
            unit="ms",
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{dataset_name} file '{path}' has an invalid timestamp; "
            "expected Unix epoch milliseconds"
        ) from error


def _convert_dtypes(
    frame: pd.DataFrame,
    path: str,
    dataset_name: str,
    dtypes: dict[str, str],
) -> None:
    """Apply contract dtypes with a path-aware error."""
    try:
        for column, dtype in dtypes.items():
            frame[column] = frame[column].astype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{dataset_name} file '{path}' contains a value incompatible with its schema"
        ) from error


def load_events(path: str, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load events, convert epoch milliseconds to UTC, and order them per visitor."""
    frame = _read_csv(
        path,
        "Events",
        EVENT_COLUMNS,
        read_dtypes={"event": "string", "transactionid": "string"},
        sample_n=sample_n,
    )
    _convert_timestamp(frame, path, "Events")
    _convert_dtypes(
        frame,
        path,
        "Events",
        {
            "visitorid": "int64",
            "event": "string",
            "itemid": "int64",
            "transactionid": "string",
        },
    )

    return frame.sort_values(
        ["visitorid", "timestamp"],
        kind="mergesort",
        ignore_index=True,
    )


def load_item_properties(path: str, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load time-varying item properties and convert timestamps to UTC."""
    frame = _read_csv(
        path,
        "Item properties",
        ITEM_PROPERTY_COLUMNS,
        read_dtypes={"property": "string", "value": "string"},
        sample_n=sample_n,
    )
    _convert_timestamp(frame, path, "Item properties")
    _convert_dtypes(
        frame,
        path,
        "Item properties",
        {"itemid": "int64", "property": "string", "value": "string"},
    )
    return frame


def load_item_properties_parts(
    paths: Sequence[str],
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """Load and concatenate ordered RetailRocket item-property shards."""
    if isinstance(paths, str) or not paths:
        raise ValueError("paths must contain at least one item-properties CSV path")

    parts = [load_item_properties(path, sample_n=sample_n) for path in paths]
    return pd.concat(parts, ignore_index=True)


def load_category_tree(path: str, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load category identifiers and their nullable parent identifiers."""
    frame = _read_csv(
        path,
        "Category tree",
        CATEGORY_TREE_COLUMNS,
        sample_n=sample_n,
    )
    _convert_dtypes(
        frame,
        path,
        "Category tree",
        {"categoryid": "int64", "parentid": "Int64"},
    )
    return frame


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load one CSV from the command line and print a compact validation summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the RetailRocket-like CSV file")
    parser.add_argument(
        "--dataset",
        choices=("events", "item-properties", "category-tree"),
        default="events",
        help="Schema and loader to use (default: events)",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Load at most N rows from the selected file",
    )
    args = parser.parse_args(argv)

    loaders = {
        "events": load_events,
        "item-properties": load_item_properties,
        "category-tree": load_category_tree,
    }
    frame = loaders[args.dataset](args.path, sample_n=args.sample_n)

    print(f"Loaded {len(frame)} {args.dataset} rows from '{args.path}'")
    print("Dtypes:")
    print(frame.dtypes.to_string())
    print("First rows:")
    print(frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
