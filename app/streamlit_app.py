"""Interactive artifact-only demo for anonymous session product ranking."""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.evaluate import load_catalog_size, load_processed_examples  # noqa: E402
from evaluation.reporting import (  # noqa: E402
    load_candidate_gru_summary,
    load_candidate_recall_summary,
    load_evaluation_summary,
    load_markdown_report,
    load_quota_candidate_recall_summary,
    summarize_metric_winners,
)
from models.gru4rec import GRU4Rec  # noqa: E402
from ranking.candidate_generation import CandidateGenerator  # noqa: E402
from ranking.recommend import (  # noqa: E402
    RankedRecommendation,
    RecommendationAdapter,
    RecommendationEngine,
)
from ranking.rerank import MMRReranker  # noqa: E402
from utils.model_artifacts import load_baseline_bundle  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_ARTIFACT = PROJECT_ROOT / "artifacts" / "baseline_models.pkl"
GRU_ARTIFACT = PROJECT_ROOT / "artifacts" / "gru4rec_best.pt"
CATEGORY_ARTIFACT = PROCESSED_DIR / "item_category_mapping.json"
RESULTS_ARTIFACTS = {
    split_name: PROJECT_ROOT / "reports" / f"results_{split_name}.json"
    for split_name in ("validation", "test")
}
CANDIDATE_RECALL_ARTIFACTS = {
    split_name: PROJECT_ROOT / "reports" / f"candidate_recall_{split_name}.json"
    for split_name in ("validation", "test")
}
CANDIDATE_GRU_RESULT_ARTIFACTS = {
    split_name: PROJECT_ROOT / "reports" / f"candidate_gru_{split_name}.json"
    for split_name in ("validation", "test")
}
QUOTA_CANDIDATE_RECALL_ARTIFACTS = {
    split_name: PROJECT_ROOT / "reports" / f"candidate_recall_quota_{split_name}.json"
    for split_name in ("validation", "test")
}
CANDIDATE_GRU_DIAGNOSTICS_ARTIFACT = (
    PROJECT_ROOT / "reports" / "candidate_gru_diagnostics.md"
)
RECORDED_OFFLINE_MMR_ALPHA = 0.8
MODEL_NAMES = (
    "Popularity",
    "Markov",
    "Item-KNN",
    "GRU4Rec",
    "GRU4Rec + diversity",
)


@dataclass
class DemoResources:
    test_examples: pd.DataFrame
    sessions: dict[object, list[int]]
    raw_to_model: dict[str, int]
    model_to_raw: dict[int, str]
    categories: dict[int, str]
    engine: RecommendationEngine
    catalog_size: int


def _reconstruct_test_sessions(examples: pd.DataFrame) -> dict[object, list[int]]:
    sessions = {}
    for session_id, session_examples in examples.groupby("session_id", sort=False):
        first_prefix = [int(item) for item in session_examples.iloc[0]["prefix_items"]]
        targets = session_examples["target_item"].astype(int).tolist()
        sessions[session_id] = first_prefix + targets
    return sessions


def _load_item_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed item mapping: '{path}'")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("<UNK_ITEM>") != 0:
        raise ValueError("The processed item mapping must reserve index 0 for <UNK_ITEM>")
    raw_to_model = {str(raw_id): int(model_id) for raw_id, model_id in mapping.items()}
    model_to_raw = {
        model_id: raw_id for raw_id, model_id in raw_to_model.items() if raw_id != "<UNK_ITEM>"
    }
    model_to_raw[0] = "<UNK_ITEM>"
    return raw_to_model, model_to_raw


def _load_categories(path: Path) -> dict[int, str]:
    """Load an optional model-item-index to category mapping."""
    if not path.is_file():
        return {}
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError(f"Category artifact '{path}' must contain a JSON object")
    return {int(item_id): str(category) for item_id, category in mapping.items()}


@st.cache_data(show_spinner=False)
def load_results_summary(path: Path, split_name: str) -> Optional[pd.DataFrame]:
    """Load recorded K=20 metrics without recomputing or fabricating results."""
    return load_evaluation_summary(path, split_name)


@st.cache_data(show_spinner=False)
def load_recorded_candidate_recall(
    path: Path,
    split_name: str,
) -> Optional[pd.DataFrame]:
    """Load recorded candidate recall without running candidate generation."""
    return load_candidate_recall_summary(path, split_name)


@st.cache_data(show_spinner=False)
def load_candidate_gru_report(path: Path, split_name: str) -> Optional[pd.DataFrame]:
    """Load saved candidate-aware metrics without loading the experimental model."""
    return load_candidate_gru_summary(path, split_name)


@st.cache_data(show_spinner=False)
def load_quota_candidate_recall(
    path: Path,
    split_name: str,
) -> Optional[pd.DataFrame]:
    """Load saved quota-candidate recall without regenerating candidates."""
    return load_quota_candidate_recall_summary(path, split_name)


@st.cache_data(show_spinner=False)
def load_candidate_gru_diagnostics(path: Path) -> Optional[str]:
    """Load the saved diagnostics Markdown as report-only content."""
    return load_markdown_report(path)


def _render_interpretation(summary: pd.DataFrame) -> None:
    winners = summarize_metric_winners(summary)
    labels = (
        ("Recall@20", "Best Recall@20 model"),
        ("MRR@20", "Best MRR@20 model"),
        ("NDCG@20", "Best NDCG@20 model"),
        ("Coverage@20", "Best Coverage@20 model"),
        ("Best neural variant", "Best neural variant by Recall@20"),
    )
    lines = []
    for metric, label in labels:
        if metric in winners:
            model_name, value = winners[metric]
            lines.append(f"**{label}:** {model_name} ({value:.6f})")
    if lines:
        st.info("  \n".join(lines))


@st.cache_resource(show_spinner="Loading processed artifacts and fitted models...")
def load_demo_resources() -> DemoResources:
    """Load immutable serving artifacts; this function never fits a model."""
    test_examples = load_processed_examples(PROCESSED_DIR, "test")
    catalog_size = load_catalog_size(PROCESSED_DIR)
    raw_to_model, model_to_raw = _load_item_mapping(
        PROCESSED_DIR / "item_id_mapping.json"
    )
    popularity, markov, item_knn = load_baseline_bundle(
        BASELINE_ARTIFACT,
        expected_catalog_size=catalog_size,
    )
    gru = GRU4Rec.load_checkpoint(GRU_ARTIFACT)
    if gru.num_items != catalog_size + 1:
        raise ValueError(
            "GRU checkpoint and processed item mapping use different vocabularies. "
            "Rebuild matching artifacts before starting the demo."
        )

    candidate_generator = CandidateGenerator(
        popularity,
        markov,
        item_knn,
        candidate_pool_size=500,
    )
    engine = RecommendationEngine(
        candidate_generator,
        popularity,
        markov,
        item_knn,
        neural_model=gru,
    )
    return DemoResources(
        test_examples=test_examples,
        sessions=_reconstruct_test_sessions(test_examples),
        raw_to_model=raw_to_model,
        model_to_raw=model_to_raw,
        categories=_load_categories(CATEGORY_ARTIFACT),
        engine=engine,
        catalog_size=catalog_size,
    )


def _parse_manual_items(value: str) -> list[int]:
    tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
    if not tokens:
        return []
    try:
        return [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("Enter integer item IDs separated by commas or spaces.") from error


def _manual_prefix(
    value: str,
    id_namespace: str,
    resources: DemoResources,
) -> tuple[list[int], list[str]]:
    entered_items = _parse_manual_items(value)
    warnings = []
    if id_namespace == "Original RetailRocket IDs":
        prefix = []
        for item_id in entered_items:
            mapped = resources.raw_to_model.get(str(item_id), 0)
            if mapped == 0:
                warnings.append(f"Original item {item_id} is unseen in the training vocabulary.")
            prefix.append(mapped)
        return prefix, warnings

    prefix = []
    for item_id in entered_items:
        if 0 <= item_id <= resources.catalog_size:
            prefix.append(item_id)
        else:
            prefix.append(0)
            warnings.append(f"Model item {item_id} is outside the vocabulary and maps to UNK.")
    return prefix, warnings


def _explanation(recommendation: RankedRecommendation, model_name: str) -> str:
    """Describe the selected model's reason; candidate provenance is shown separately."""
    if model_name == "GRU4Rec + diversity":
        return "selected after diversity re-ranking"
    if model_name == "GRU4Rec":
        return "high GRU score among generated candidates"
    if model_name == "Markov" and "markov" in recommendation.sources:
        return "often follows the last item"
    if model_name == "Item-KNN" and "item_knn" in recommendation.sources:
        return "co-occurs with session items"
    return "popular item"


def _recommendations_for_model(
    resources: DemoResources,
    prefix: list[int],
    model_name: str,
    k: int,
    diversity_alpha: float,
) -> list[RankedRecommendation]:
    if model_name == "GRU4Rec + diversity":
        reranked_gru = RecommendationAdapter(
            resources.engine,
            "gru",
            reranker=MMRReranker(
                # MMR uses train-only Item-KNN cosine co-visitation similarity,
                # not category similarity.
                resources.engine.item_knn.neighbor_scores,
                alpha=diversity_alpha,
            ),
        )
        return reranked_gru.recommend_with_scores(prefix, k)
    engine_name = {
        "Popularity": "popularity",
        "Markov": "markov",
        "Item-KNN": "item_knn",
        "GRU4Rec": "gru",
    }[model_name]
    return resources.engine.recommend(prefix, engine_name, k)


def _recommendation_table(
    recommendations: list[RankedRecommendation],
    model_name: str,
    resources: DemoResources,
) -> pd.DataFrame:
    rows = []
    for rank, recommendation in enumerate(recommendations, start=1):
        item_id = int(recommendation.item_id)
        rows.append(
            {
                "rank": rank,
                "item_id": resources.model_to_raw.get(item_id, str(item_id)),
                "model_item_id": item_id,
                "score": recommendation.score,
                "model": model_name,
                "candidate_sources": ", ".join(
                    source
                    for source in recommendation.sources
                    if source in {"popularity", "markov", "item_knn"}
                )
                or "none",
                "category": resources.categories.get(item_id, "Unavailable"),
                "explanation": _explanation(recommendation, model_name),
            }
        )
    return pd.DataFrame(rows)


def _prefix_table(prefix: list[int], resources: DemoResources) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "position": position,
                "item_id": resources.model_to_raw.get(item_id, str(item_id)),
                "model_item_id": item_id,
            }
            for position, item_id in enumerate(prefix, start=1)
        ]
    )


st.set_page_config(page_title="Session Product Ranking", layout="wide")
st.title("Real-Time Session-Based Product Ranking")
st.caption("Anonymous RetailRocket sessions · artifact-only inference · no model retraining")
st.warning(
    "RetailRocket item IDs are anonymized and product names are unavailable. "
    "Recommendations therefore display anonymous item IDs."
)
with st.expander("How to read this demo", expanded=True):
    st.markdown(
        "1. RetailRocket item IDs are anonymized; product names are not provided.\n"
        "2. Scores are model-specific and must not be compared across model columns.\n"
        "3. GRU4Rec scores a shared candidate pool generated by Popularity, Markov, and "
        "Item-KNN.\n"
        "4. Diversity uses train-only Item-KNN cosine co-visitation similarity, not category "
        "diversity. Optional categories are display metadata only.\n"
        "5. Lower MMR alpha applies a stronger similarity penalty; higher alpha preserves more "
        "of the GRU relevance order.\n"
        "6. The top-N and alpha sliders affect only displayed interactive recommendations. "
        "Recorded metrics and candidate recall are precomputed and do not change with sliders."
    )

try:
    resources = load_demo_resources()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(str(error))
    st.code("python3 scripts/build_demo_artifacts.py\nstreamlit run app/streamlit_app.py")
    st.stop()

st.subheader("Recorded offline results")
result_tabs = st.tabs(("Validation", "Test"))
for result_tab, split_name in zip(result_tabs, ("validation", "test")):
    with result_tab:
        try:
            results_summary = load_results_summary(
                RESULTS_ARTIFACTS[split_name],
                split_name,
            )
        except (json.JSONDecodeError, ValueError) as error:
            st.warning(f"Could not load {split_name} evaluation results: {error}")
        else:
            if results_summary is None:
                st.warning(
                    f"{split_name.title()} evaluation results not found. Run "
                    f"scripts/evaluate_model.py --split {split_name} before using this section."
                )
            else:
                st.caption(
                    f"Recorded {split_name} metrics at K=20; this app does not recompute them. "
                    f"Recorded MMR used alpha={RECORDED_OFFLINE_MMR_ALPHA:.2f}."
                )
                st.dataframe(
                    results_summary,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        metric: st.column_config.NumberColumn(format="%.6f")
                        for metric in ("Recall@20", "MRR@20", "NDCG@20", "Coverage@20")
                    }
                    | {"Latency/query (ms)": st.column_config.NumberColumn(format="%.4f")},
                )
                _render_interpretation(results_summary)

st.info(
    "Classical session baselines outperform the initial GRU4Rec. A separate candidate-aware "
    "GRU experiment substantially improves neural reranking by training against hard "
    "candidates from the same quota candidate pool."
)

if any(path.is_file() for path in CANDIDATE_RECALL_ARTIFACTS.values()):
    st.subheader("Candidate pool recall")
    st.caption(
        "Recall indicates whether the true next item was available for ranking. Candidate "
        "generators are fitted on train data only; unknown targets count as misses."
    )
    candidate_tabs = st.tabs(("Validation candidates", "Test candidates"))
    for candidate_tab, split_name in zip(candidate_tabs, ("validation", "test")):
        with candidate_tab:
            try:
                candidate_summary = load_recorded_candidate_recall(
                    CANDIDATE_RECALL_ARTIFACTS[split_name],
                    split_name,
                )
            except (json.JSONDecodeError, ValueError) as error:
                st.warning(f"Could not load {split_name} candidate recall: {error}")
            else:
                if candidate_summary is None:
                    st.caption(f"No recorded {split_name} candidate recall report was found.")
                else:
                    st.dataframe(
                        candidate_summary,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            metric: st.column_config.NumberColumn(format="%.6f")
                            for metric in ("Recall@100", "Recall@500", "Recall@1000")
                        },
                    )

st.subheader("Candidate-aware GRU experiment")
st.caption(
    "Report-only experiment results loaded from saved artifacts. Candidate-aware GRU is not "
    "loaded into the live recommendation interface, and this app does not recompute metrics."
)
candidate_gru_tabs = st.tabs(("Validation experiment", "Observed test"))
with candidate_gru_tabs[0]:
    try:
        candidate_gru_validation = load_candidate_gru_report(
            CANDIDATE_GRU_RESULT_ARTIFACTS["validation"],
            "validation",
        )
    except (json.JSONDecodeError, ValueError) as error:
        st.warning(f"Could not load candidate-aware validation report: {error}")
    else:
        if candidate_gru_validation is None:
            st.warning("Candidate-aware GRU validation report not found.")
        else:
            st.dataframe(
                candidate_gru_validation,
                hide_index=True,
                width="stretch",
                column_config={
                    metric: st.column_config.NumberColumn(format="%.6f")
                    for metric in ("Recall@20", "MRR@20", "NDCG@20", "Coverage@20")
                },
            )
            st.info(
                "The initial GRU4Rec underperformed classical baselines. Candidate-aware "
                "hard-negative training substantially improved neural reranking on the same "
                "quota candidate pool, indicating that train–serve mismatch was a major issue."
            )

with candidate_gru_tabs[1]:
    try:
        candidate_gru_test = load_candidate_gru_report(
            CANDIDATE_GRU_RESULT_ARTIFACTS["test"],
            "test",
        )
    except (json.JSONDecodeError, ValueError) as error:
        st.warning(f"Could not load candidate-aware observed-test report: {error}")
    else:
        if candidate_gru_test is None:
            st.warning("Candidate-aware GRU observed-test report not found.")
        else:
            test_mrr = candidate_gru_test.set_index("model")["MRR@20"]
            metric_columns = st.columns(2)
            metric_columns[0].metric(
                "Candidate-aware GRU observed-test MRR@20",
                f"{test_mrr['Candidate-aware GRU']:.6f}",
            )
            metric_columns[1].metric(
                "Same-pool current GRU observed-test MRR@20",
                f"{test_mrr['Current GRU on quota pool']:.6f}",
            )
            st.warning(
                "Observed test was run after the validation gate passed and should be treated "
                "as confirmatory evidence, not an unbiased final holdout."
            )

st.markdown("#### Quota candidate recall")
quota_tabs = st.tabs(("Validation quota recall", "Observed-test quota recall"))
for quota_tab, split_name in zip(quota_tabs, ("validation", "test")):
    with quota_tab:
        try:
            quota_summary = load_quota_candidate_recall(
                QUOTA_CANDIDATE_RECALL_ARTIFACTS[split_name],
                split_name,
            )
        except (json.JSONDecodeError, ValueError) as error:
            st.warning(f"Could not load {split_name} quota candidate recall: {error}")
        else:
            if quota_summary is None:
                st.warning(f"{split_name.title()} quota candidate recall report not found.")
            else:
                st.dataframe(
                    quota_summary,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        column: st.column_config.NumberColumn(format="%.6f")
                        for column in ("quota_combined", "old combined", "improvement")
                    },
                )

try:
    candidate_gru_diagnostics = load_candidate_gru_diagnostics(
        CANDIDATE_GRU_DIAGNOSTICS_ARTIFACT
    )
except ValueError as error:
    st.warning(f"Could not load candidate-aware diagnostics: {error}")
else:
    if candidate_gru_diagnostics is None:
        st.caption("Candidate-aware GRU diagnostics report not found.")
    else:
        with st.expander("Candidate-aware GRU diagnostics"):
            st.markdown(candidate_gru_diagnostics)

st.markdown(
    "**Limitations**\n\n"
    "- Candidate-aware GRU improves strongly over current GRU but does not dominate every "
    "classical baseline.\n"
    "- Candidate recall still limits all rerankers.\n"
    "- MMR at alpha 0.8 reduced ranking metrics and should be treated as a diversity tradeoff."
)

with st.sidebar:
    st.header("Session input")
    input_mode = st.radio("Choose input", ("Real test session", "Manual item IDs"))
    top_k = st.slider("Recommendations per model", min_value=5, max_value=20, value=10)
    diversity_alpha = st.slider(
        "Diversity relevance weight (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Higher values favor GRU relevance; lower values penalize similar items more.",
    )
    st.caption(
        "Top-N controls only displayed recommendations. Alpha controls only displayed MMR "
        "recommendations; neither slider changes recorded offline metrics."
    )
    if abs(diversity_alpha - RECORDED_OFFLINE_MMR_ALPHA) > 1e-9:
        st.warning(
            f"Interactive MMR uses alpha={diversity_alpha:.2f}, while recorded offline "
            f"MMR metrics use alpha={RECORDED_OFFLINE_MMR_ALPHA:.2f}."
        )

prefix: list[int] = []
if input_mode == "Real test session":
    session_ids = list(resources.sessions)
    selected_session = st.selectbox(
        "Test session",
        session_ids,
        format_func=lambda session_id: (
            f"Session {session_id} · {len(resources.sessions[session_id])} interactions"
        ),
    )
    full_session = resources.sessions[selected_session]
    maximum_prefix = max(1, len(full_session) - 1)
    if maximum_prefix == 1:
        prefix_length = 1
        st.caption("Observed prefix length: 1")
    else:
        prefix_length = st.slider(
            "Observed prefix length",
            min_value=1,
            max_value=maximum_prefix,
            value=min(3, maximum_prefix),
        )
    prefix = full_session[:prefix_length]
else:
    id_namespace = st.selectbox(
        "ID format",
        ("Original RetailRocket IDs", "Model vocabulary IDs"),
    )
    manual_value = st.text_input(
        "Item IDs",
        placeholder="Example: 257040, 298351, 123458",
    )
    try:
        prefix, input_warnings = _manual_prefix(manual_value, id_namespace, resources)
    except ValueError as error:
        st.error(str(error))
        prefix, input_warnings = [], []
    for warning in input_warnings:
        st.warning(warning)

st.subheader("Current session prefix")
if not prefix:
    st.info("Enter at least one item ID to generate recommendations.")
    st.stop()
st.dataframe(_prefix_table(prefix, resources), hide_index=True, width="stretch")

all_tables = {
    model_name: _recommendation_table(
        _recommendations_for_model(
            resources,
            prefix,
            model_name,
            top_k,
            diversity_alpha,
        ),
        model_name,
        resources,
    )
    for model_name in MODEL_NAMES
}

st.subheader("Baseline vs model comparison")
comparison = pd.DataFrame(
    {
        model_name: table["item_id"].tolist()
        for model_name, table in all_tables.items()
    },
    index=range(1, top_k + 1),
)
comparison.index.name = "rank"
st.dataframe(comparison, width="stretch")

st.subheader("Recommendation details")
tabs = st.tabs(MODEL_NAMES)
for tab, model_name in zip(tabs, MODEL_NAMES):
    with tab:
        if model_name == "GRU4Rec + diversity":
            st.caption(
                f"MMR alpha={diversity_alpha:.2f}; reranking trades GRU relevance against "
                "Item-KNN co-visitation similarity and may reduce ranking metrics."
            )
        st.dataframe(
            all_tables[model_name],
            hide_index=True,
            width="stretch",
            column_config={"score": st.column_config.NumberColumn(format="%.5f")},
        )

if resources.categories:
    st.caption(
        "Categories are optional display metadata and are not used by the current MMR reranker."
    )
else:
    st.caption(
        "Category mapping is not available in the processed artifacts; category values are "
        "shown as Unavailable."
    )
st.caption(
    "Scores are model-specific and are not directly comparable across columns. "
    "GRU recommendations are scored over the shared classical candidate pool; diversity is "
    "measured from item co-visitation similarity."
)
