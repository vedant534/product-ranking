# Real-Time Session-Based Product Ranking

A reproducible recommendation project for ranking the next product in an anonymous e-commerce
session. The project uses the RetailRocket dataset and compares Most Popular, Item-KNN /
co-visitation, first-order Markov, and a simple PyTorch GRU4Rec-style model.

The primary evaluation uses a chronological train/validation/test split and reports Recall@20,
MRR@20, NDCG@20, HitRate@10, catalog coverage, and single-session recommendation latency. See
[`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the complete data, modeling, evaluation, and acceptance
criteria.

## Current Status

Implemented end-to-end workflow:

- Project layout, dependency metadata, configurations, Make targets, and pytest setup are present.
- Raw-data validation, loading, sessionization, chronological splitting, and next-item dataset
  creation are implemented.
- Popularity, Markov, and Item-KNN baselines plus offline ranking metrics are implemented.
- Unified evaluation, segment analysis, MMR reranking, and the GRU4Rec-style neural model are
  implemented.
- The Streamlit demo loads saved model artifacts and compares all five recommendation variants.

## Requirements

- Python 3.9 or newer
- `make`
- Enough local disk space for the uncompressed RetailRocket CSV files and generated artifacts

GPU support is optional. Training and neural evaluation support CPU and automatically prefer CUDA
or Apple MPS when available.

## Setup

Create and activate an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternatively, install through the convenience requirements file:

```bash
python -m pip install -r requirements.txt
```

Verify the scaffold:

```bash
make test
make lint
```

## Dataset Placement

Obtain the RetailRocket e-commerce dataset separately. Dataset files are intentionally excluded
from version control. Place them at these paths:

```text
data/raw/retailrocket/events.csv
data/raw/retailrocket/item_properties_part1.csv
data/raw/retailrocket/item_properties_part2.csv
data/raw/retailrocket/category_tree.csv
```

Expected columns:

| File | Columns |
|---|---|
| `events.csv` | `timestamp`, `visitorid`, `event`, `itemid`, `transactionid` |
| `item_properties_part1.csv` and `item_properties_part2.csv` | `timestamp`, `itemid`, `property`, `value` |
| `category_tree.csv` | `categoryid`, `parentid` |

Verify the raw files before preprocessing. Use `--sample-n` to read only the first N event rows
during a smoke test; property and category statistics still cover their complete files:

```bash
python scripts/check_raw_data.py --sample-n 1000
```

The loader concatenates item-property parts in part 1, part 2 order after validating both schemas.

Never commit raw, interim, or processed dataset files. Tiny synthetic fixtures are kept under
`data/sample/` for tests and examples.

## Commands

| Command | Purpose | Current behavior |
|---|---|---|
| `make test` | Run pytest | Runs the project test suite |
| `make lint` | Run Ruff | Checks Python source and tests |
| `make prepare-data` | Build model-ready processed data | Sessionizes, purges split overlap, and writes examples |
| `make prepare-demo` | Build serving metadata | Saves fitted baselines and processed categories |
| `make run-baselines` | Run the three required baselines | Evaluates all baselines on validation artifacts |
| `make train` | Train the GRU4Rec-style model | Trains and saves the best validation-MRR checkpoint |
| `make evaluate` | Evaluate a saved recommender | Runs K=5/10/20 unified evaluation and updates reports |
| `make demo` | Launch Streamlit | Opens the artifact-only interactive recommendation demo |

Override the default command configuration when needed:

```bash
make prepare-data CONFIG=configs/retailrocket.yaml
```

Create model-ready next-item examples and write the required Parquet/JSON artifacts:

```bash
python3 scripts/prepare_data.py --step make-dataset --sample-n 100000
```

Omit `--sample-n` for the full dataset. The main split purges sessions whose final event crosses
the next split boundary, preserving complete retained sessions and strict event-time ordering.
Known targets already present in the prefix are retained in the artifacts for session
reconstruction but excluded from the unseen-item training and evaluation objective.

The model-specific defaults are recorded in `configs/model_gru.yaml`. Select another evaluator model
with `make evaluate MODEL=markov SPLIT=validation`.

On Apple Silicon, run training and neural evaluation in an environment where PyTorch reports MPS
available. The full-data config uses sampled cross-entropy during training and exact full-catalog
MRR@20 on a fixed validation sample for checkpoint selection; final reports evaluate the complete
eligible validation split.

## Streamlit Demo

The demo never fits or retrains models. After preparing data and training the GRU, build the fitted
baseline bundle once and launch Streamlit:

```bash
python3 scripts/build_demo_artifacts.py
python3 scripts/build_category_mapping.py
python3 -m streamlit run app/streamlit_app.py
```

`make demo` is an equivalent launcher. At runtime the app reads only `data/processed/`,
`artifacts/baseline_models.pkl`, and `artifacts/gru4rec_best.pt`; it does not require the full raw
RetailRocket CSV files. An optional `data/processed/item_category_mapping.json` file may map model
item indices to category labels. Product names are unavailable because RetailRocket item IDs are
anonymized.

## Repository Layout

```text
configs/       Reproducible project, dataset, and model settings
data/          Local raw/interim/processed data and committed tiny samples
src/           Data, baseline, model, ranking, evaluation, and utility packages
scripts/       Command-line workflow entry points
tests/         Pytest suite
notebooks/     Exploration, baseline result, and failure analysis notebooks
app/           Streamlit application
reports/       Results, limitations, and technical report documents
artifacts/     Generated indexes and trained artifacts (not committed)
```

## Evaluation Policy

The main result must use session-level chronological train/validation/test partitions. Random splits
must not be reported as the primary result. All recommenders must share the same candidate catalog,
test examples, fallback policy, and metric implementation so comparisons remain valid.
