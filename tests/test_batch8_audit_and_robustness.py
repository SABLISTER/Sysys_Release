"""Tests for the eighth fix batch.

Covers:
  - Stage 5 TF-IDF equivalence, model caching, and paper_id normalization.
  - Stage 9 Spearman-Brown split-half rewrite and the scipy-free Kendall tau-b.
  - Synthesizer deterministic embedding/clustering fallbacks.
  - Query builder no longer leaking its mutable module defaults.
"""
from __future__ import annotations

import math
from typing import SupportsFloat, cast

import numpy as np
import pytest

from process import queries
from process.synthesizer import _hashing_embeddings, _simple_kmeans
from stages import s5_audit
from stages.s5_audit import (
    _derive_paper_id,
    _normalize_paper_ids,
    _tfidf_counts,
    _tfidf_similarity,
    _tfidf_similarity_counts,
)
from stages.s9_robustness import (
    _kendall_tau_b_numpy,
    _spearman_brown,
    kendall_tau,
    split_half,
)


# ── Stage 5: TF-IDF ───────────────────────────────────────────────────────

def _sklearn_tfidf_cosine(a: str, b: str) -> float:
    import sklearn.feature_extraction.text as sklearn_text
    import sklearn.metrics.pairwise as sklearn_pair
    matrix = sklearn_text.TfidfVectorizer().fit_transform([a, b])
    return float(sklearn_pair.cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


@pytest.mark.parametrize(
    "span,abstract",
    [
        ("cortical thickness increased", "We measured cortical thickness in adults."),
        ("no shared words here", "completely different vocabulary entirely"),
        ("repeat repeat repeat", "repeat once and then some other words"),
        ("Mixed CASE Tokens", "mixed case tokens appear in the abstract"),
    ],
)
def test_tfidf_matches_sklearn(span, abstract):
    """The pure-Python TF-IDF must reproduce sklearn's cosine exactly."""
    assert _tfidf_similarity(span, abstract) == pytest.approx(
        _sklearn_tfidf_cosine(span, abstract), abs=1e-12
    )


def test_tfidf_identical_and_disjoint_documents():
    assert _tfidf_similarity("alpha beta", "alpha beta") == pytest.approx(1.0)
    assert _tfidf_similarity("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_tfidf_empty_inputs_are_zero_not_error():
    assert _tfidf_similarity("", "") == 0.0
    assert _tfidf_similarity("something", "") == 0.0
    assert _tfidf_similarity_counts(_tfidf_counts(""), _tfidf_counts("x y")) == 0.0


def test_tfidf_precomputed_counts_match_string_api():
    """Precomputing the abstract side must not change the score."""
    abstract = "the cortex shows increased activation during the task"
    span = "increased activation in cortex"
    assert _tfidf_similarity_counts(
        _tfidf_counts(span), _tfidf_counts(abstract)
    ) == pytest.approx(_tfidf_similarity(span, abstract))


def test_tfidf_does_not_depend_on_sklearn_presence(monkeypatch):
    """Audit scores must be identical whether or not sklearn is installed."""
    baseline = _tfidf_similarity("shared token here", "a shared token appears here")
    monkeypatch.setattr(s5_audit, "_HAS_SKLEARN", False)
    assert _tfidf_similarity(
        "shared token here", "a shared token appears here"
    ) == pytest.approx(baseline)


# ── Stage 5: model caching ────────────────────────────────────────────────

def test_span_model_is_loaded_at_most_once(monkeypatch):
    calls = []

    def fake_ctor(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(s5_audit, "_HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(s5_audit, "SentenceTransformer", fake_ctor, raising=False)
    monkeypatch.setattr(s5_audit, "_span_model_cache", [None, False])

    first = s5_audit._get_span_model()
    second = s5_audit._get_span_model()

    assert first is second
    assert len(calls) == 1


def test_failed_model_load_is_not_retried(monkeypatch):
    calls = []

    def exploding_ctor(name):
        calls.append(name)
        raise RuntimeError("no weights")

    monkeypatch.setattr(s5_audit, "_HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(s5_audit, "SentenceTransformer", exploding_ctor, raising=False)
    monkeypatch.setattr(s5_audit, "_span_model_cache", [None, False])

    assert s5_audit._get_span_model() is None
    assert s5_audit._get_span_model() is None
    assert len(calls) == 1


# ── Stage 5: paper_id normalization ───────────────────────────────────────

def test_normalize_paper_ids_makes_blank_ids_unique():
    claims = [{"title": "Alpha", "year": 2020}, {"title": "Beta", "year": 2021}]
    _normalize_paper_ids(claims, [])
    ids = [c["paper_id"] for c in claims]
    assert all(ids)
    assert len(set(ids)) == 2


def test_normalize_paper_ids_disambiguates_duplicates():
    claims = [{"paper_id": "dup"}, {"paper_id": "dup"}, {"paper_id": "dup"}]
    repaired = _normalize_paper_ids(claims, [])
    ids = [c["paper_id"] for c in claims]
    assert ids == ["dup", "dup#2", "dup#3"]
    assert repaired == 2


def test_normalize_paper_ids_preserves_existing_unique_ids():
    claims = [{"paper_id": "keep-me"}, {"id": "legacy-id"}]
    _normalize_paper_ids(claims, [])
    assert claims[0]["paper_id"] == "keep-me"
    assert claims[1]["paper_id"] == "legacy-id"


def test_derive_paper_id_prefers_doi_and_is_stable():
    paper = {"doi": "10.1234/ABC", "pmid": "999", "title": "T"}
    assert _derive_paper_id(paper, 0) == "doi:10.1234/abc"
    # Same title/year always yields the same synthetic id across runs.
    a = _derive_paper_id({"title": "Same Title", "year": 2020}, 0)
    b = _derive_paper_id({"title": "Same Title", "year": 2020}, 7)
    assert a == b


def test_normalize_paper_ids_fills_blank_profile_ids():
    profiles = [{"title": "P one"}, {"paper_id": "existing"}]
    _normalize_paper_ids([], profiles)
    assert profiles[0]["paper_id"]
    assert profiles[1]["paper_id"] == "existing"


# ── Stage 9: split-half ───────────────────────────────────────────────────

_RANKING = [
    "Fragile X", "Rett syndrome", "Epilepsy", "Intellectual disability",
    "Schizophrenia", "Anxiety", "Depression", "ADHD",
]
_PATTERNS = {c.lower(): c for c in _RANKING}


def _condition_corpus(rates: dict[str, float], n_per: int = 60, seed: int = 0):
    """Build a corpus whose per-condition support rates are known."""
    rng = np.random.default_rng(seed)
    papers = []
    for condition, rate in rates.items():
        for _ in range(n_per):
            papers.append({
                "title": f"Study of {condition.lower()}",
                "abstract": "placeholder",
                "supports_primary_question": bool(rng.random() < rate),
                "relevance_score": 5,
            })
    rng.shuffle(papers)
    return papers


def _split_half(papers, **kw):
    kw.setdefault("condition_patterns", _PATTERNS)
    kw.setdefault("condition_ranking", _RANKING)
    return split_half(papers, **kw)


def test_split_half_is_deterministic():
    papers = _condition_corpus({c: 0.5 for c in _RANKING}, n_per=20)
    assert _split_half(papers, n_splits=100) == _split_half(papers, n_splits=100)


def test_split_half_recovers_a_strong_condition_gradient():
    """A corpus with genuinely different per-condition rates is reliable."""
    rates = dict(zip(_RANKING, [0.95, 0.85, 0.75, 0.6, 0.4, 0.25, 0.15, 0.05]))
    result = _split_half(_condition_corpus(rates), n_splits=200)
    assert result["estimable"] is True
    assert result["mean_correlation"] > 0.8
    assert result["interpretation"] == "excellent"


def test_split_half_reports_no_reliability_when_there_is_no_gradient():
    """Identical rates across conditions leave nothing to replicate."""
    rates = {c: 0.5 for c in _RANKING}
    result = _split_half(_condition_corpus(rates, seed=1), n_splits=200)
    assert result["estimable"] is True
    assert result["mean_correlation"] < 0.4
    assert result["interpretation"] == "poor"


def test_split_half_is_not_pinned_by_complementary_halves():
    """Regression: the old statistic was forced to -1.0 for even n.

    ``mean_a + mean_b == 2 * grand_mean`` made the reported correlation and
    ICC mathematical constants rather than measurements. The rate profile must
    not reproduce that behaviour at any corpus size or parity.
    """
    rates = dict(zip(_RANKING, [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]))
    for n_per in (7, 8, 25, 26):
        result = _split_half(_condition_corpus(rates, n_per=n_per), n_splits=120)
        assert result["mean_correlation"] > 0.0
        assert result["icc"] != pytest.approx(-1.0, abs=1e-6)


def test_split_half_applies_spearman_brown_correction():
    """The headline figure must exceed the uncorrected half-length value."""
    rates = dict(zip(_RANKING, [0.9, 0.75, 0.65, 0.55, 0.45, 0.3, 0.2, 0.1]))
    result = _split_half(_condition_corpus(rates, seed=4), n_splits=200)
    assert result["spearman_brown_applied"] is True
    assert result["mean_correlation"] > result["mean_half_correlation"] > 0
    assert result["mean_correlation"] == pytest.approx(
        _spearman_brown(result["mean_half_correlation"]), abs=0.06
    )


@pytest.mark.parametrize(
    "r_half,expected",
    [(0.0, 0.0), (0.5, 2 / 3), (0.8, 8 / 9), (1.0, 1.0), (-1.0, -1.0)],
)
def test_spearman_brown_formula(r_half, expected):
    assert _spearman_brown(r_half) == pytest.approx(expected)


def test_split_half_not_estimable_without_conditions():
    papers = _condition_corpus({c: 0.5 for c in _RANKING}, n_per=10)
    result = split_half(papers, n_splits=10)
    assert result["estimable"] is False
    assert "no conditions configured" in result["note"]
    assert result["interpretation"] == "not estimable"


def test_split_half_not_estimable_for_tiny_corpora():
    result = split_half([{"relevance_score": 1.0}], n_splits=10)
    assert result["estimable"] is False
    assert result["n_splits"] == 0


def test_split_half_not_estimable_when_no_paper_matches_a_condition():
    papers = [{"title": "unrelated topic", "abstract": ""} for _ in range(50)]
    result = _split_half(papers, n_splits=10)
    assert result["estimable"] is False
    assert "no paper matched" in result["note"]


def test_split_half_result_carries_keys_downstream_consumers_read():
    """_write_report, the GUI, and s10_verify all index these keys."""
    rates = dict(zip(_RANKING, [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))
    result = _split_half(_condition_corpus(rates, seed=6), n_splits=50)
    for key in (
        "mean_correlation", "std_correlation", "min_correlation",
        "max_correlation", "kendall_tau", "kendall_tau_p", "icc",
        "icc_ci_lower", "icc_ci_upper", "n_splits", "interpretation",
    ):
        assert key in result


# ── Stage 9: Kendall tau without scipy ────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kendall_fallback_matches_scipy(seed):
    import scipy.stats as scipy_stats
    rng = np.random.default_rng(seed)
    x = list(rng.normal(size=50))
    y = list(rng.normal(size=50))

    expected_tau, expected_p = scipy_stats.kendalltau(x, y)
    tau, p = _kendall_tau_b_numpy(x, y)

    assert tau == pytest.approx(float(cast(SupportsFloat, expected_tau)), abs=1e-12)
    # Normal approximation: close but not exact.
    assert p == pytest.approx(float(cast(SupportsFloat, expected_p)), abs=0.02)


def test_kendall_fallback_handles_ties():
    x = [1.0, 1.0, 2.0, 2.0, 3.0]
    y = [1.0, 2.0, 2.0, 3.0, 3.0]
    tau, p = _kendall_tau_b_numpy(x, y)
    assert -1.0 <= tau <= 1.0
    assert 0.0 <= p <= 1.0
    assert not math.isnan(tau)


def test_kendall_tau_reports_whether_p_is_approximate(monkeypatch):
    monkeypatch.setattr("stages.s9_robustness._HAS_SCIPY", False)
    result = kendall_tau([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0])
    assert result["p_value_approximate"] is True


def test_kendall_tau_degenerate_input():
    assert kendall_tau([], [])["tau"] == 0.0
    assert kendall_tau([1.0], [1.0])["p_value"] == 1.0


# ── Synthesizer fallbacks ─────────────────────────────────────────────────

def test_hashing_embeddings_are_deterministic():
    texts = ["cortical thickness autism", "monetary policy inflation"]
    assert np.array_equal(_hashing_embeddings(texts), _hashing_embeddings(texts))


def test_hashing_embeddings_separate_topics():
    """The fallback must carry real signal, unlike the old random embeddings."""
    texts = [
        "brain imaging autism cortical thickness",
        "cortical thickness autism brain imaging",
        "monetary policy inflation interest rates",
        "inflation interest rates monetary policy",
    ]
    embeddings = _hashing_embeddings(texts)
    sims = embeddings @ embeddings.T
    assert sims[0, 1] > 0.5
    assert sims[2, 3] > 0.5
    assert sims[0, 2] < 0.1


def test_hashing_embeddings_are_unit_norm_and_handle_empty_text():
    embeddings = _hashing_embeddings(["some real words here", ""])
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0)
    assert np.linalg.norm(embeddings[1]) == pytest.approx(0.0)


def test_simple_kmeans_is_reproducible_and_ignores_global_seed():
    embeddings = _hashing_embeddings(
        [
            "brain imaging autism cortex",
            "autism cortex brain imaging",
            "inflation monetary policy rates",
            "rates policy monetary inflation",
        ]
    )
    np.random.seed(1)
    first = _simple_kmeans(embeddings, 2)
    np.random.seed(999)
    second = _simple_kmeans(embeddings, 2)
    assert np.array_equal(first, second)
    # Topic pairs must land together.
    assert first[0] == first[1]
    assert first[2] == first[3]
    assert first[0] != first[2]


# ── Query builder default aliasing ────────────────────────────────────────

def test_get_base_terms_does_not_leak_module_default():
    original = list(queries._DEFAULT_BASE_TERMS)
    returned = queries._get_base_terms(None)
    returned.append("contaminant")
    assert queries._DEFAULT_BASE_TERMS == original


def test_get_domains_does_not_leak_module_default():
    original = list(queries._DEFAULT_DOMAINS)
    returned = queries._get_domains(None)
    returned.append({"name": "contaminant"})
    assert queries._DEFAULT_DOMAINS == original


def test_get_base_terms_does_not_leak_caller_config():
    config = {"base_terms": ["alpha"]}
    returned = queries._get_base_terms(config)
    returned.append("beta")
    assert config["base_terms"] == ["alpha"]
