"""Validate and summarize the raw RetailRocket dataset files."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    source_dir = PROJECT_ROOT / "src"
    sys.path.insert(0, str(source_dir))

DEFAULT_DATA_DIR = Path("data/raw/retailrocket")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "retailrocket.yaml"
REQUIRED_FILENAMES = (
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
)
EXPECTED_COLUMNS = {
    "events.csv": ("timestamp", "visitorid", "event", "itemid", "transactionid"),
    "item_properties_part1.csv": ("timestamp", "itemid", "property", "value"),
    "item_properties_part2.csv": ("timestamp", "itemid", "property", "value"),
    "category_tree.csv": ("categoryid", "parentid"),
}
ALLOWED_EVENTS = {"view", "addtocart", "transaction"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def get_required_paths(data_dir: Path) -> dict[str, Path]:
    """Return the required filenames mapped to paths under ``data_dir``."""
    return {filename: data_dir / filename for filename in REQUIRED_FILENAMES}


def verify_required_files(data_dir: Path) -> dict[str, Path]:
    """Return required paths or raise one error listing every missing file."""
    paths = get_required_paths(data_dir)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required RetailRocket raw file(s):\n{missing_list}")
    return paths


def validate_headers(paths: dict[str, Path]) -> None:
    """Read only CSV headers and validate every required schema before data loading."""
    for filename in REQUIRED_FILENAMES:
        path = paths[filename]
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"{filename} is empty and has no header") from error

        missing = sorted(set(EXPECTED_COLUMNS[filename]) - set(columns))
        if missing:
            raise ValueError(
                f"{filename} is missing expected column(s): {', '.join(missing)}. "
                f"Found columns: {columns}"
            )


def _config_filename_warnings(config_path: Path) -> list[str]:
    """Report filename drift between the checker and RetailRocket configuration."""
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        files = config["dataset"]["files"]
        property_paths = files["item_properties"]
        if isinstance(property_paths, str):
            property_paths = [property_paths]
        configured_names = [
            Path(files["events"]).name,
            *(Path(path).name for path in property_paths),
            Path(files["category_tree"]).name,
        ]
    except (KeyError, TypeError, OSError, yaml.YAMLError) as error:
        return [f"Could not verify filenames against {config_path}: {error}"]

    if sorted(configured_names) != sorted(REQUIRED_FILENAMES):
        return [
            "Raw filenames do not match configs/retailrocket.yaml: "
            f"checker expects {list(REQUIRED_FILENAMES)}, config expects {configured_names}"
        ]
    return []


def _load_events_for_check(path: Path, sample_n: Optional[int]) -> pd.DataFrame:
    """Load event fields without rejecting null IDs so diagnostics can report them."""
    events = pd.read_csv(
        path,
        nrows=sample_n,
        dtype={"event": "string", "transactionid": "string"},
    )
    if events["timestamp"].isna().any():
        raise ValueError("events.csv contains null timestamp values")
    try:
        epoch_milliseconds = pd.to_numeric(events["timestamp"], errors="raise")
        events["timestamp"] = pd.to_datetime(
            epoch_milliseconds,
            unit="ms",
            utc=True,
            errors="raise",
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "events.csv contains invalid timestamps; expected Unix milliseconds"
        ) from error
    return events


def _event_warnings(events: pd.DataFrame) -> list[str]:
    warnings = []
    if events.empty:
        return ["events.csv contains no data rows"]

    timestamp_min = events["timestamp"].min()
    timestamp_max = events["timestamp"].max()
    current_year = pd.Timestamp.now(tz="UTC").year
    if timestamp_min.year < 2000 or timestamp_max.year > current_year + 1:
        warnings.append(
            "Event timestamp conversion produced a suspicious range: "
            f"{timestamp_min} to {timestamp_max}"
        )

    observed_events = set(events["event"].dropna().astype(str).unique())
    unexpected_events = sorted(observed_events - ALLOWED_EVENTS)
    if unexpected_events:
        warnings.append(f"Unexpected event values: {unexpected_events}")

    for column in ("visitorid", "itemid"):
        null_count = int(events[column].isna().sum())
        if null_count:
            null_percentage = 100 * null_count / len(events)
            warnings.append(
                f"{column} contains {null_count:,} nulls ({null_percentage:.2f}% of loaded rows)"
            )
    return warnings


def _print_events(events: pd.DataFrame, sampled: bool) -> None:
    row_label = "row count (sampled)" if sampled else "row count"
    print("\nevents.csv")
    print(f"  {row_label}: {len(events):,}")
    print(f"  column names: {list(events.columns)}")
    if events.empty:
        print("  timestamp range: N/A")
    else:
        print(f"  timestamp range: {events['timestamp'].min()} to {events['timestamp'].max()}")
    print("  event counts:")
    event_counts = events["event"].value_counts(dropna=False).sort_index()
    for event, count in event_counts.items():
        print(f"    {event}: {count:,}")
    print(f"  unique visitors: {events['visitorid'].nunique():,}")
    print(f"  unique items: {events['itemid'].nunique():,}")
    print(f"  visitorid nulls: {events['visitorid'].isna().sum():,}")
    print(f"  itemid nulls: {events['itemid'].isna().sum():,}")


def _print_item_properties(part1: pd.DataFrame, part2: pd.DataFrame) -> None:
    combined_items = pd.concat([part1["itemid"], part2["itemid"]], ignore_index=True)
    property_counts = (
        part1["property"]
        .value_counts()
        .add(part2["property"].value_counts(), fill_value=0)
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nitem properties")
    print(f"  item_properties_part1.csv row count: {len(part1):,}")
    print(f"  item_properties_part2.csv row count: {len(part2):,}")
    print(f"  combined row count: {len(part1) + len(part2):,}")
    print(f"  unique items: {combined_items.nunique():,}")
    print("  top property names:")
    for property_name, count in property_counts.items():
        print(f"    {property_name}: {int(count):,}")


def _print_categories(categories: pd.DataFrame) -> None:
    print("\ncategory_tree.csv")
    print(f"  row count: {len(categories):,}")
    print(f"  column names: {list(categories.columns)}")
    print(f"  unique categories: {categories['categoryid'].nunique():,}")


def check_raw_data(data_dir: Path, sample_n: Optional[int] = None) -> list[str]:
    """Verify required files, load them, and print raw-data summaries."""
    from data.load_data import load_category_tree, load_item_properties

    paths = verify_required_files(data_dir)
    for filename in REQUIRED_FILENAMES:
        print(f"{filename} found")

    validate_headers(paths)
    print("All headers contain the expected columns.")

    if sample_n is not None:
        print(f"Smoke-test mode: loading the first {sample_n:,} rows from events.csv only.")

    events = _load_events_for_check(paths["events.csv"], sample_n=sample_n)
    properties_part1 = load_item_properties(str(paths["item_properties_part1.csv"]))
    properties_part2 = load_item_properties(str(paths["item_properties_part2.csv"]))
    categories = load_category_tree(str(paths["category_tree.csv"]))

    _print_events(events, sampled=sample_n is not None)
    _print_item_properties(properties_part1, properties_part2)
    _print_categories(categories)

    warnings = _event_warnings(events)
    warnings.extend(_config_filename_warnings(DEFAULT_CONFIG_PATH))
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nWarnings: none.")
    return warnings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Raw RetailRocket directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--sample-n",
        type=_positive_int,
        default=None,
        help="Load only the first N rows from events.csv",
    )
    args = parser.parse_args(argv)

    try:
        check_raw_data(args.data_dir, sample_n=args.sample_n)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(status=1, message=f"Raw data check failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
