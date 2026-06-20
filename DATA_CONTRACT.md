# RetailRocket Data Contract

## Purpose

This contract defines the input schema for raw RetailRocket data and the committed synthetic sample.
The sample is structurally representative only; it is not extracted from the RetailRocket dataset and
must not be used for model-quality claims.

## File Mapping

| Dataset | Raw path | Synthetic sample path |
|---|---|---|
| Events | `data/raw/retailrocket/events.csv` | `data/sample/events_sample.csv` |
| Item properties | `data/raw/retailrocket/item_properties_part1.csv` and `item_properties_part2.csv` | `data/sample/item_properties_sample.csv` |
| Category tree | `data/raw/retailrocket/category_tree.csv` | `data/sample/category_tree_sample.csv` |

All files are comma-delimited UTF-8 CSVs with a header row. Column names are case-sensitive. Extra
columns must not affect the required pipeline, but missing required columns are validation errors.

## Timestamp Convention

Every `timestamp` is a non-negative Unix timestamp in **milliseconds since 1970-01-01 00:00:00
UTC**. Load it as a signed 64-bit integer and convert with an explicit `unit="ms", utc=True` setting.
Do not infer the unit from magnitude and do not interpret timestamps in the machine's local timezone.

## `events.csv`

One row represents one visitor interaction with one item.

| Column | Logical dtype | Nullable | Rules |
|---|---|---:|---|
| `timestamp` | `int64` | No | Unix epoch milliseconds in UTC |
| `visitorid` | `int64` | No | Positive anonymous visitor identifier; not a known customer identity |
| `event` | string / categorical | No | Exactly one of `view`, `addtocart`, or `transaction` |
| `itemid` | `int64` | No | Positive item identifier |
| `transactionid` | string | Yes | Required for `transaction`; normally empty for other event types |

An empty `transactionid` field represents null, not an empty transaction identifier. Transaction IDs
are opaque strings: implementations must not assume they are numeric, globally ordered, or unique per
item. Unknown event values are rejected by default rather than silently mapped to a supported event.

Events are ordered within each visitor by `timestamp`, then by original CSV row position for timestamp
ties. Input files do not need to be globally sorted. Multiple visitors may have overlapping timestamps.

## `item_properties.csv`

One row is a time-varying property assignment for an item.

| Column | Logical dtype | Nullable | Rules |
|---|---|---:|---|
| `timestamp` | `int64` | No | Time from which the observed property value is available |
| `itemid` | `int64` | No | Positive item identifier |
| `property` | string | No | Opaque property name, including values such as `categoryid` |
| `value` | string | No | Opaque value; preserve its textual representation |

Properties can change over time and an item can have many properties. For an as-of lookup, use the
last value for `(itemid, property)` whose timestamp is at or before the applicable data cutoff. A
future property row must never be used to enrich an earlier event. If duplicate updates share the same
timestamp, original CSV row position is the deterministic tie-breaker.

Property values must remain strings during general ingestion. A property-specific downstream step may
parse a value, such as interpreting `categoryid` as an integer, only after validating that value.
The two raw property files are validated independently and concatenated in part 1, part 2 order. A
bounded smoke-test read applies the row limit independently to each physical part.

## `category_tree.csv`

One row defines a category and its optional parent.

| Column | Logical dtype | Nullable | Rules |
|---|---|---:|---|
| `categoryid` | `int64` | No | Unique positive category identifier |
| `parentid` | nullable `int64` | Yes | Parent category identifier; null marks a root category |

The category structure may be a forest with multiple roots. Every non-null `parentid` should resolve
to a `categoryid` in the same file. Cycles, self-parenting, and duplicate category IDs are invalid.
Items or categories can be absent from the other files; missing metadata must not remove otherwise
valid events.

## Session and Modeling Assumptions

- `visitorid` is anonymous and must not be treated as a durable authenticated user identity.
- A new session begins when the gap between consecutive events for one visitor is greater than 30
  minutes. The threshold is configurable; a gap of exactly 30 minutes remains in the same session.
- Sessions are built only after deterministic per-visitor event ordering.
- `view`, `addtocart`, and `transaction` all contribute item interactions to the default sequence.
- The target is the next interacted item, regardless of event type.
- Consecutive same-item interactions are retained at ingestion. Their optional collapse is a separate,
  recorded preprocessing decision.
- Repeated items are valid next-item targets and must not be automatically removed from candidates.
- Event rows and raw metadata are immutable inputs. Cleaned or derived data belongs under
  `data/interim/` or `data/processed/`.
- Vocabularies, item statistics, metadata snapshots, and recommendation artifacts are fitted from the
  chronological training partition only.

## Synthetic Sample Characteristics

The committed sample contains four visitors, ten items, all three allowed event types, nullable and
non-null transaction IDs, consecutive same-item interactions, repeated items, multiple session gaps,
several category roots, and a time-varying availability property. It is intended for ingestion,
sessionization, split, and smoke tests.
