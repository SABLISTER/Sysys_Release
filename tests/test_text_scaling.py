from __future__ import annotations

import math

import pytest

from core.models import Article
from process.text_scaling import (
    TextScalingCancelled,
    build_stylometric_profile,
    run_text_scaling,
)


def _article(article_id: str, title: str, sentence: str, repeats: int = 8) -> Article:
    return Article(
        id=article_id,
        title=title,
        abstract=" ".join([sentence] * repeats),
    )


def test_stylometric_profile_reports_interpretable_rates():
    article = Article(
        id="a1",
        title="Measured language",
        abstract=(
            "The result may possibly suggest an effect. "
            "The effect was clearly demonstrated and confirmed."
        ),
    )

    profile = build_stylometric_profile("a1", article)

    assert profile["word_count"] > 10
    assert profile["hedge_count"] >= 3
    assert profile["certainty_count"] >= 3
    assert profile["hedge_density"] > 0
    assert profile["certainty_density"] > 0
    assert profile["type_token_ratio"] > 0


def test_text_scaling_returns_positions_profiles_and_statistics():
    articles = {
        "clinical-1": _article(
            "clinical-1",
            "Clinical response",
            "Patients showed treatment response in the clinical trial and symptom assessment.",
        ),
        "clinical-2": _article(
            "clinical-2",
            "Treatment outcomes",
            "Clinical patients demonstrated symptom improvement after treatment and assessment.",
        ),
        "computational-1": _article(
            "computational-1",
            "Network model",
            "The computational network model estimated nodes, connectivity, and simulation parameters.",
        ),
        "computational-2": _article(
            "computational-2",
            "Connectivity simulation",
            "Network connectivity was estimated with a computational model and simulation parameters.",
        ),
        "too-short": Article(id="too-short", title="Short", abstract="Not enough text."),
    }

    model, profiles, statistics = run_text_scaling(
        articles,
        min_doc_words=20,
        max_features=100,
        max_iterations=6,
    )

    assert model["algorithm"] == "Wordfish Poisson text scaling"
    assert model["n_documents"] == 4
    assert model["n_features"] >= 2
    assert set(model["positions"]) == {
        "clinical-1", "clinical-2", "computational-1", "computational-2"
    }
    assert set(profiles) == set(model["positions"])
    assert statistics["excluded_documents"] == 1
    assert statistics["n_documents"] == 4
    assert model["discriminating_terms"]
    assert all(math.isfinite(position) for position in model["positions"].values())
    assert max(model["positions"].values()) - min(model["positions"].values()) > 0.1


def test_text_scaling_honors_cancellation():
    articles = {
        "a": _article("a", "One", "Shared scientific text contains enough repeated words."),
        "b": _article("b", "Two", "Shared scientific text contains enough repeated words."),
    }

    with pytest.raises(TextScalingCancelled):
        run_text_scaling(
            articles,
            min_doc_words=5,
            cancel_cb=lambda: True,
        )
