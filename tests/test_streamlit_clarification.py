"""Runtime checks for the text-only Streamlit clarification."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_keeps_existing_models_and_warns_on_alpha_mismatch() -> None:
    app = AppTest.from_file("app/streamlit_app.py").run(timeout=60)

    assert not app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert "Candidate-aware GRU" not in tab_labels
    assert all(
        model_name in tab_labels
        for model_name in (
            "Popularity",
            "Markov",
            "Item-KNN",
            "GRU4Rec",
            "GRU4Rec + diversity",
        )
    )
    app_source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "models.candidate_gru" not in app_source
    assert any(
        subheader.value == "Candidate-aware GRU experiment" for subheader in app.subheader
    )
    candidate_frames = [
        dataframe.value
        for dataframe in app.dataframe
        if "model" in dataframe.value.columns
        and "Candidate-aware GRU" in dataframe.value["model"].astype(str).tolist()
    ]
    assert len(candidate_frames) == 1
    candidate_validation = candidate_frames[0].set_index("model")
    assert candidate_validation.loc["Current GRU on quota pool", "MRR@20"] == pytest.approx(
        0.038378,
        abs=1e-6,
    )
    assert candidate_validation.loc["Candidate-aware GRU", "MRR@20"] == pytest.approx(
        0.090700,
        abs=1e-6,
    )
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Candidate-aware GRU observed-test MRR@20"] == "0.073671"
    assert metrics["Same-pool current GRU observed-test MRR@20"] == "0.030761"
    assert any(
        "confirmatory evidence, not an unbiased final holdout" in warning.value
        for warning in app.warning
    )
    assert any(
        "Candidate-aware GRU improves strongly over current GRU" in markdown.value
        for markdown in app.markdown
    )
    assert any(
        "Top-N controls only displayed recommendations" in caption.value
        for caption in app.caption
    )

    alpha_slider = next(
        slider for slider in app.slider if slider.label == "Diversity relevance weight (alpha)"
    )
    alpha_slider.set_value(0.9)
    app.run(timeout=60)

    assert any(
        "Interactive MMR uses alpha=0.90" in warning.value for warning in app.warning
    )
