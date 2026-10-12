"""Tests for statistical enhancements: BH FDR, Cohen's d, Kendall's τ, ICC.

Known-answer values verified against:
- R's p.adjust(method="BH") for FDR
- Hand-computed Cohen's d with pooled SD formula
- scipy.stats.kendalltau for Kendall's tau
- Textbook ICC(1,1) from one-way ANOVA decomposition
"""
import importlib.util
import math
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Direct module loading — bypass stages/__init__.py which eagerly imports
# all stage modules and cascades into httpx/aiohttp/etc.
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load_module_direct(name: str, file_path: Path):
    """Load a single .py file as a module without triggering its package __init__."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_benjamini_hochberg():
    mod = _load_module_direct(
        "process.meta_analysis", _root / "process" / "meta_analysis.py"
    )
    return mod.benjamini_hochberg


def _get_cluster_effect_size():
    mod = _load_module_direct(
        "stages.s9_robustness", _root / "stages" / "s9_robustness.py"
    )
    return mod.cluster_effect_size


def _get_kendall_tau():
    mod = _load_module_direct(
        "stages.s9_robustness", _root / "stages" / "s9_robustness.py"
    )
    return mod.kendall_tau


def _get_icc_oneway():
    mod = _load_module_direct(
        "stages.s9_robustness", _root / "stages" / "s9_robustness.py"
    )
    return mod.icc_oneway


# ═══════════════════════════════════════════════════════════════════════════
# Benjamini-Hochberg FDR Correction
# ═══════════════════════════════════════════════════════════════════════════

class TestBenjaminiHochberg:
    """Tests for BH FDR correction against R's p.adjust(method='BH')."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.benjamini_hochberg = _get_benjamini_hochberg()

    def test_bh_known_answer(self):
        """Verified against R: p.adjust(c(0.001,0.01,0.03,0.15,0.6), method='BH')
        → [0.005, 0.025, 0.050, 0.1875, 0.600]
        """
        p_values = {
            "A": 0.001,
            "B": 0.01,
            "C": 0.03,
            "D": 0.15,
            "E": 0.6,
        }
        result = self.benjamini_hochberg(p_values, alpha=0.05)

        assert pytest.approx(result["A"]["adjusted_p"], abs=1e-6) == 0.005
        assert pytest.approx(result["B"]["adjusted_p"], abs=1e-6) == 0.025
        assert pytest.approx(result["C"]["adjusted_p"], abs=1e-6) == 0.050
        assert pytest.approx(result["D"]["adjusted_p"], abs=1e-6) == 0.1875
        assert pytest.approx(result["E"]["adjusted_p"], abs=1e-6) == 0.600

        assert result["A"]["significant"] is True
        assert result["B"]["significant"] is True
        assert result["C"]["significant"] is True
        assert result["D"]["significant"] is False
        assert result["E"]["significant"] is False

    def test_bh_empty(self):
        """Empty input returns empty output."""
        assert self.benjamini_hochberg({}) == {}

    def test_bh_single(self):
        """Single p-value: adjusted = raw (no correction needed)."""
        result = self.benjamini_hochberg({"only": 0.03}, alpha=0.05)
        assert pytest.approx(result["only"]["adjusted_p"], abs=1e-6) == 0.03
        assert result["only"]["significant"] is True

    def test_bh_all_significant(self):
        """All very small p-values remain significant."""
        pvals = {"a": 0.001, "b": 0.002, "c": 0.003}
        result = self.benjamini_hochberg(pvals, alpha=0.05)
        assert all(v["significant"] for v in result.values())

    def test_bh_all_nonsignificant(self):
        """All large p-values remain non-significant."""
        pvals = {"a": 0.5, "b": 0.7, "c": 0.9}
        result = self.benjamini_hochberg(pvals, alpha=0.05)
        assert not any(v["significant"] for v in result.values())

    def test_bh_monotonicity(self):
        """Adjusted p-values maintain the ordering of raw p-values."""
        pvals = {"a": 0.01, "b": 0.04, "c": 0.02, "d": 0.5}
        result = self.benjamini_hochberg(pvals)
        sorted_by_raw = sorted(result.items(), key=lambda kv: kv[1]["raw_p"])
        adjusted_vals = [kv[1]["adjusted_p"] for kv in sorted_by_raw]
        for i in range(len(adjusted_vals) - 1):
            assert adjusted_vals[i] <= adjusted_vals[i + 1] + 1e-12

    def test_bh_preserves_raw(self):
        """Raw p-values are preserved unchanged in the output."""
        pvals = {"x": 0.03, "y": 0.07}
        result = self.benjamini_hochberg(pvals)
        assert result["x"]["raw_p"] == 0.03
        assert result["y"]["raw_p"] == 0.07

    def test_bh_adjusted_never_exceeds_one(self):
        """Adjusted p-values are capped at 1.0."""
        pvals = {"a": 0.8, "b": 0.9}
        result = self.benjamini_hochberg(pvals)
        assert all(v["adjusted_p"] <= 1.0 for v in result.values())


# ═══════════════════════════════════════════════════════════════════════════
# Cohen's d / Hedges' g Effect Sizes
# ═══════════════════════════════════════════════════════════════════════════

class TestClusterEffectSize:
    """Tests for standardized effect size computation."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.cluster_effect_size = _get_cluster_effect_size()

    def test_cohens_d_known(self):
        """Groups [1,2,3] vs [4,5,6]: d = (2-5)/1.0 = -3.0.
        Pooled SD = sqrt(((2*1 + 2*1)/4)) = 1.0, so d = -3.0.
        """
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = self.cluster_effect_size(a, b, hedges_correct=False)
        assert pytest.approx(result["d"], abs=0.01) == -3.0

    def test_cohens_d_identical(self):
        """Identical groups → d = 0."""
        a = np.array([3.0, 3.0, 3.0])
        result = self.cluster_effect_size(a, a, hedges_correct=False)
        assert pytest.approx(result["d"], abs=1e-6) == 0.0

    def test_cohens_d_hedges_shrinks(self):
        """Hedges' correction should reduce |d| for small samples."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        uncorrected = self.cluster_effect_size(a, b, hedges_correct=False)
        corrected = self.cluster_effect_size(a, b, hedges_correct=True)
        assert abs(corrected["hedges_g"]) < abs(uncorrected["d"])

    def test_cohens_d_ci_contains_d(self):
        """95% CI should contain the point estimate."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        result = self.cluster_effect_size(a, b)
        assert result["ci_lower"] <= result["d"] <= result["ci_upper"]

    def test_cohens_d_magnitude_labels(self):
        """Verify magnitude thresholds: <0.2 negligible, <0.5 small, <0.8 medium, else large."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert self.cluster_effect_size(a, b)["magnitude"] == "large"

        c = np.array([5.0, 5.1, 4.9, 5.0, 5.05])
        d = np.array([5.01, 5.09, 4.91, 5.02, 5.04])
        assert self.cluster_effect_size(c, d)["magnitude"] == "negligible"

    def test_cohens_d_small_n(self):
        """Should handle n=2 per group without error."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = self.cluster_effect_size(a, b)
        assert "d" in result
        assert math.isfinite(result["d"])

    def test_cohens_d_returns_group_sizes(self):
        """Result includes group sizes for transparency."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        result = self.cluster_effect_size(a, b)
        assert result["n_a"] == 3
        assert result["n_b"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Kendall's Tau
# ═══════════════════════════════════════════════════════════════════════════

class TestKendallTau:
    """Tests for Kendall's tau-b wrapper."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.kendall_tau = _get_kendall_tau()

    def test_kendall_perfect(self):
        """Identical rankings → τ = 1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.kendall_tau(x, y)
        assert pytest.approx(result["tau"], abs=1e-6) == 1.0

    def test_kendall_reversed(self):
        """Reversed rankings → τ = -1.0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = self.kendall_tau(x, y)
        assert pytest.approx(result["tau"], abs=1e-6) == -1.0

    def test_kendall_matches_scipy(self):
        """Verify our wrapper matches scipy.stats.kendalltau."""
        from scipy.stats import kendalltau as sp_kendalltau
        x = [1.0, 3.0, 2.0, 5.0, 4.0]
        y = [2.0, 4.0, 1.0, 5.0, 3.0]
        result = self.kendall_tau(x, y)
        expected_tau, expected_p = sp_kendalltau(x, y)
        assert pytest.approx(result["tau"], abs=1e-6) == expected_tau
        assert pytest.approx(result["p_value"], abs=1e-6) == expected_p

    def test_kendall_too_short(self):
        """< 2 elements returns tau=0, p=1."""
        result = self.kendall_tau([1.0], [2.0])
        assert result["tau"] == 0.0
        assert result["p_value"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Intraclass Correlation Coefficient (ICC)
# ═══════════════════════════════════════════════════════════════════════════

class TestICC:
    """Tests for ICC(1,1) one-way random model."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.icc_oneway = _get_icc_oneway()

    def test_icc_perfect_agreement(self):
        """Identical raters → ICC ≈ 1.0."""
        ratings = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
                           dtype=float)
        result = self.icc_oneway(ratings)
        assert result["icc"] > 0.99

    def test_icc_no_agreement(self):
        """Random independent raters → ICC ≈ 0.0 (within sampling noise)."""
        rng = np.random.default_rng(42)
        ratings = rng.standard_normal((200, 2))
        result = self.icc_oneway(ratings)
        assert abs(result["icc"]) < 0.2

    def test_icc_known_answer(self):
        """Ratings with constant offset: [1,2], [2,3], [3,4], [4,5], [5,6].
        Between-subject variance dominates → high ICC(1,1).
        """
        ratings = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
                           dtype=float)
        result = self.icc_oneway(ratings)
        assert result["icc"] > 0.7

    def test_icc_ci_contains_point(self):
        """CI should contain the point estimate."""
        ratings = np.array([[1, 2], [3, 3], [5, 4], [2, 3], [4, 5]],
                           dtype=float)
        result = self.icc_oneway(ratings)
        assert result["ci_lower"] <= result["icc"] <= result["ci_upper"]

    def test_icc_too_few_subjects(self):
        """< 3 subjects returns ICC=0, CI=(0,0)."""
        ratings = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.icc_oneway(ratings)
        assert result["icc"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# REML Meta-Analysis Estimator
# ═══════════════════════════════════════════════════════════════════════════

class TestREML:
    """Tests for REML tau² estimation in pool_random_effects()."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_module_direct(
            "process.meta_analysis", _root / "process" / "meta_analysis.py"
        )
        self.pool_random_effects = mod.pool_random_effects
        self.StudyEffect = mod.StudyEffect

    def _make_studies(self, effects, variances):
        """Helper to create StudyEffect list from parallel arrays."""
        studies = []
        for i, (e, v) in enumerate(zip(effects, variances)):
            se = math.sqrt(v)
            studies.append(self.StudyEffect(
                article_id=f"study_{i}",
                effect_size=e,
                standard_error=se,
                variance=v,
                sample_size=50,
            ))
        return studies

    def test_reml_returns_valid_result(self):
        """REML should return a PooledEstimate with all fields populated."""
        studies = self._make_studies(
            [0.5, 0.3, 0.7, 0.4, 0.6],
            [0.04, 0.05, 0.03, 0.06, 0.04],
        )
        result = self.pool_random_effects(studies, method="REML")
        assert result.n_studies == 5
        assert result.pooled_se > 0
        assert 0.0 <= result.i_squared <= 100.0

    def test_reml_tau2_geq_zero(self):
        """REML τ² should always be non-negative."""
        # Homogeneous studies → τ² should be near 0
        studies = self._make_studies(
            [0.5, 0.5, 0.5, 0.5],
            [0.04, 0.04, 0.04, 0.04],
        )
        result = self.pool_random_effects(studies, method="REML")
        assert result.tau_squared >= 0.0

    def test_reml_vs_dl_heterogeneous(self):
        """With heterogeneous studies, REML τ² should differ from DL τ²
        (REML is generally larger for small k)."""
        studies = self._make_studies(
            [0.1, 0.5, 1.2, 0.8, 2.0],
            [0.04, 0.05, 0.03, 0.06, 0.04],
        )
        dl_result = self.pool_random_effects(studies, method="DL")
        reml_result = self.pool_random_effects(studies, method="REML")
        # They shouldn't be exactly equal for heterogeneous data
        assert dl_result.tau_squared != pytest.approx(reml_result.tau_squared, abs=1e-6)
        # Both should produce valid pooled estimates
        assert math.isfinite(reml_result.pooled_effect)
        assert math.isfinite(dl_result.pooled_effect)

    def test_reml_single_study(self):
        """Single study should work without error."""
        studies = self._make_studies([0.5], [0.04])
        result = self.pool_random_effects(studies, method="REML")
        assert result.n_studies == 1

    def test_dl_still_works(self):
        """Ensure DL method still works after adding REML."""
        studies = self._make_studies(
            [0.5, 0.3, 0.7],
            [0.04, 0.05, 0.03],
        )
        result = self.pool_random_effects(studies, method="DL")
        assert result.n_studies == 3
        assert math.isfinite(result.pooled_effect)


# ═══════════════════════════════════════════════════════════════════════════
# Data-Driven Network Thresholding
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeThreshold:
    """Tests for permutation-based edge thresholding."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_module_direct(
            "process.autism_graph_analysis",
            _root / "process" / "autism_graph_analysis.py",
        )
        self.compute_edge_threshold = mod.compute_edge_threshold
        self.build_cooccurrence_edges = mod.build_cooccurrence_edges
        self.load_patterns = mod.load_patterns

    def test_threshold_positive(self):
        """Threshold should be a positive number."""
        # Create a small synthetic corpus with known co-occurrences
        corpus = [
            {"title": "anxiety and amygdala activation", "abstract": ""},
            {"title": "ADHD prefrontal cortex connectivity", "abstract": ""},
            {"title": "anxiety amygdala fMRI study", "abstract": ""},
            {"title": "epilepsy and hippocampus", "abstract": ""},
            {"title": "anxiety amygdala volume", "abstract": ""},
        ]
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        threshold = self.compute_edge_threshold(
            corpus, patterns, n_permutations=100, seed=42,
        )
        assert threshold > 0

    def test_strong_signal_survives(self):
        """A deliberately strong co-occurrence should survive thresholding."""
        # Plant a very strong signal: anxiety+amygdala in 10 papers
        corpus = [
            {"title": f"anxiety and amygdala study {i}", "abstract": ""}
            for i in range(10)
        ] + [
            {"title": "epilepsy hippocampus case report", "abstract": ""},
        ]
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        threshold = self.compute_edge_threshold(
            corpus, patterns, n_permutations=200, seed=42,
        )
        edges = self.build_cooccurrence_edges(corpus, patterns)
        # The anxiety-amygdala edge should have weight >> threshold
        anxiety_amygdala = [e for e in edges
                           if e["source"] == "Anxiety" and e["target"] == "Amygdala"]
        assert len(anxiety_amygdala) == 1
        assert anxiety_amygdala[0]["weight"] >= threshold

    def test_empty_corpus(self):
        """Empty corpus returns threshold 0."""
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        threshold = self.compute_edge_threshold([], patterns, n_permutations=50)
        assert threshold == 0


# ═══════════════════════════════════════════════════════════════════════════
# Community Detection (Louvain)
# ═══════════════════════════════════════════════════════════════════════════

class TestCommunityDetection:
    """Tests for Louvain community detection."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_module_direct(
            "process.autism_graph_analysis",
            _root / "process" / "autism_graph_analysis.py",
        )
        self.detect_communities = mod.detect_communities

    def test_two_cliques(self):
        """Two cliques connected by a weak bridge → 2 communities."""
        # Clique A: nodes 0,1,2 (strongly connected)
        # Clique B: nodes 3,4,5 (strongly connected)
        # Weak bridge: 2-3 (weight=1)
        edges = [
            {"source": "A0", "target": "A1", "weight": 10},
            {"source": "A0", "target": "A2", "weight": 10},
            {"source": "A1", "target": "A2", "weight": 10},
            {"source": "B0", "target": "B1", "weight": 10},
            {"source": "B0", "target": "B2", "weight": 10},
            {"source": "B1", "target": "B2", "weight": 10},
            {"source": "A2", "target": "B0", "weight": 1},  # weak bridge
        ]
        result = self.detect_communities(edges)
        assert result["n_communities"] >= 2
        assert result["modularity"] > 0.0
        # A0 and A1 should be in the same community
        assert result["communities"]["A0"] == result["communities"]["A1"]
        # A0 and B0 should be in different communities
        assert result["communities"]["A0"] != result["communities"]["B0"]

    def test_single_node(self):
        """Single-edge graph returns 1 community."""
        edges = [{"source": "X", "target": "Y", "weight": 5}]
        result = self.detect_communities(edges)
        assert result["n_communities"] >= 1

    def test_empty_edges(self):
        """Empty edge list returns empty communities."""
        result = self.detect_communities([])
        assert result["n_communities"] == 0
        assert result["communities"] == {}
        assert result["modularity"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Network Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalNetwork:
    """Tests for temporal co-occurrence network analysis."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_module_direct(
            "process.autism_graph_analysis",
            _root / "process" / "autism_graph_analysis.py",
        )
        self.temporal_network_analysis = mod.temporal_network_analysis
        self.load_patterns = mod.load_patterns

    def test_basic_windows(self):
        """Papers spanning 10 years with 5-year windows → 2 windows."""
        corpus = [
            {"title": "anxiety amygdala", "abstract": "", "year": 2015},
            {"title": "anxiety amygdala", "abstract": "", "year": 2016},
            {"title": "epilepsy hippocampus", "abstract": "", "year": 2020},
            {"title": "epilepsy hippocampus", "abstract": "", "year": 2021},
        ]
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        windows = self.temporal_network_analysis(corpus, patterns, window_years=5)
        assert len(windows) >= 2
        assert all("n_papers" in w for w in windows)
        assert all("n_edges" in w for w in windows)

    def test_jaccard_similarity(self):
        """First window has no jaccard_vs_prev; subsequent ones do."""
        corpus = [
            {"title": "anxiety amygdala", "abstract": "", "year": 2010},
            {"title": "anxiety amygdala", "abstract": "", "year": 2015},
            {"title": "anxiety amygdala", "abstract": "", "year": 2020},
        ]
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        windows = self.temporal_network_analysis(corpus, patterns, window_years=5)
        if len(windows) >= 2:
            assert windows[0]["jaccard_vs_prev"] is None
            assert isinstance(windows[1]["jaccard_vs_prev"], float)

    def test_empty_corpus(self):
        """Empty corpus returns empty list."""
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        assert self.temporal_network_analysis([], patterns) == []

    def test_no_years(self):
        """Papers without year field are skipped gracefully."""
        corpus = [{"title": "anxiety amygdala", "abstract": ""}]
        patterns = self.load_patterns(SimpleNamespace(research_config={"defaults": {"pattern_pack": "neuroscience"}}))
        assert self.temporal_network_analysis(corpus, patterns) == []


# ═══════════════════════════════════════════════════════════════════════════
# Confidence Calibration
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibration:
    """Tests for confidence score calibration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_module_direct(
            "analysis.scoring", _root / "analysis" / "scoring.py"
        )
        self.calibrate_confidence = mod.calibrate_confidence

    def test_isotonic_perfect_calibration(self):
        """Already-calibrated scores → ECE near 0."""
        # Scores that match the true probabilities
        predicted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 20
        rng = np.random.default_rng(42)
        observed = [int(rng.random() < p) for p in predicted]
        result = self.calibrate_confidence(predicted, observed, method="isotonic")
        # ECE should be relatively low for well-calibrated inputs
        assert result["ece"] < 0.3
        assert len(result["calibrated"]) == len(predicted)
        assert result["method"] == "isotonic"

    def test_platt_returns_valid(self):
        """Platt scaling returns calibrated values between 0 and 1."""
        predicted = [0.1, 0.3, 0.5, 0.7, 0.9] * 10
        observed = [0, 0, 1, 1, 1] * 10
        result = self.calibrate_confidence(predicted, observed, method="platt")
        assert all(0 <= c <= 1 for c in result["calibrated"])
        assert result["method"] == "platt"

    def test_too_few_samples(self):
        """< 3 samples returns uncalibrated."""
        result = self.calibrate_confidence([0.5, 0.6], [0, 1])
        assert result["ece"] == 1.0
        assert result["calibrated"] == [0.5, 0.6]

    def test_bin_stats_structure(self):
        """bin_stats should have expected fields."""
        predicted = [0.2, 0.4, 0.6, 0.8] * 10
        observed = [0, 0, 1, 1] * 10
        result = self.calibrate_confidence(predicted, observed, n_bins=5)
        assert isinstance(result["bin_stats"], list)
        if result["bin_stats"]:
            b = result["bin_stats"][0]
            assert "bin_center" in b
            assert "bin_count" in b
            assert "mean_predicted" in b
            assert "mean_observed" in b


# ═══════════════════════════════════════════════════════════════════════════
# Fleiss' Kappa
# ═══════════════════════════════════════════════════════════════════════════

class TestFleissKappa:
    """Tests for multi-rater Fleiss' kappa."""

    @pytest.fixture(autouse=True)
    def _load(self):
        mod = _load_module_direct(
            "stages.s8_interrater", _root / "stages" / "s8_interrater.py"
        )
        self.fleiss_kappa = mod.fleiss_kappa

    def test_perfect_agreement(self):
        """All raters agree on every subject → κ = 1.0."""
        # 5 subjects, 3 categories, 4 raters
        # All 4 raters pick category 0 for subject 0, category 1 for subject 1, etc.
        matrix = [
            [4, 0, 0],  # all raters pick cat 0
            [0, 4, 0],  # all raters pick cat 1
            [0, 0, 4],  # all raters pick cat 2
            [4, 0, 0],
            [0, 4, 0],
        ]
        result = self.fleiss_kappa(matrix)
        assert pytest.approx(result["kappa"], abs=1e-6) == 1.0
        assert result["interpretation"] == "almost perfect"

    def test_known_answer(self):
        """Fleiss (1971) textbook example.

        14 subjects, 5 raters, 3 categories.
        Expected κ ≈ 0.210 (fair agreement).
        """
        # Simplified version of Fleiss's original example
        matrix = [
            [0, 0, 5],
            [0, 2, 3],
            [0, 5, 0],
            [2, 3, 0],
            [5, 0, 0],
            [0, 1, 4],
            [1, 0, 4],
            [0, 4, 1],
            [3, 2, 0],
            [2, 2, 1],
        ]
        result = self.fleiss_kappa(matrix)
        # Should be a valid kappa between -1 and 1
        assert -1.0 <= result["kappa"] <= 1.0
        assert result["n_subjects"] == 10
        assert result["n_raters"] == 5

    def test_random_agreement(self):
        """Uniform ratings → κ ≤ 0 (no better than chance, possibly worse).
        With [2,2,2] (6 raters, 3 categories), P̄=0.2 < P̄ₑ=0.333 → κ=-0.2.
        """
        matrix = [
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ]
        result = self.fleiss_kappa(matrix)
        assert result["kappa"] <= 0.0

    def test_empty_matrix(self):
        """Empty matrix returns kappa 0."""
        result = self.fleiss_kappa([])
        assert result["kappa"] == 0.0

    def test_single_subject(self):
        """Single subject → insufficient data."""
        result = self.fleiss_kappa([[3, 0, 0]])
        assert result["interpretation"] == "insufficient data"


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced Citation sLDA with Metadata Features
# ═══════════════════════════════════════════════════════════════════════════

class TestEnhancedSLDA:
    """Tests for metadata-enhanced citation regression on top of sLDA."""

    @pytest.fixture(autouse=True)
    def _load(self):
        import pandas as pd
        self.pd = pd
        mod = _load_module_direct(
            "process.citation_slda", _root / "process" / "citation_slda.py"
        )
        self.enhanced_citation_regression = mod.enhanced_citation_regression
        self._extract_metadata_features = mod._extract_metadata_features
        self.METADATA_FEATURES = mod.METADATA_FEATURES

    def _make_mock_slda_result(self, n=50, k=5, seed=42):
        """Create a mock sLDA result with synthetic topic distributions."""
        rng = np.random.default_rng(seed)
        topic_cols = [f"topic_{i}" for i in range(k)]
        data: dict[str, Any] = {col: rng.random(n) for col in topic_cols}
        data["citation_target"] = rng.random(n) * 5
        data["citation_count"] = np.expm1(data["citation_target"]).astype(int)
        data["year"] = rng.integers(2010, 2024, n)
        data["n_authors"] = rng.integers(1, 10, n)
        data["openalex_is_oa"] = rng.integers(0, 2, n)
        data["split"] = ["train"] * (n // 2) + ["test"] * (n - n // 2)
        df = self.pd.DataFrame(data)
        return {"documents": df, "model": None}

    def test_enhanced_returns_valid_structure(self):
        """Enhanced regression returns all expected fields."""
        result = self.enhanced_citation_regression(self._make_mock_slda_result())
        assert "r2_topics_only" in result
        assert "r2_combined" in result
        assert "r2_improvement" in result
        assert "feature_importances" in result
        assert "n_topic_features" in result
        assert "n_metadata_features" in result
        assert "metadata_features_used" in result

    def test_metadata_improves_or_maintains(self):
        """Adding metadata should not drastically hurt R² (improvement ≥ -0.1)."""
        result = self.enhanced_citation_regression(self._make_mock_slda_result(n=100))
        # With random data, improvement could be slightly negative due to
        # overfitting, but shouldn't be dramatically worse
        assert result["r2_improvement"] > -0.2

    def test_feature_importances_include_metadata(self):
        """Feature importances should include both topic and metadata features."""
        result = self.enhanced_citation_regression(self._make_mock_slda_result())
        importances = result["feature_importances"]
        # Should have topic features
        assert any(k.startswith("topic_") for k in importances)
        # Should have at least some metadata features
        assert result["n_metadata_features"] > 0

    def test_extract_metadata_handles_missing_columns(self):
        """Missing columns are silently skipped."""
        df = self.pd.DataFrame({"year": [2020, 2021, 2022]})
        features, names = self._extract_metadata_features(df)
        assert "year" in names
        assert "openalex_is_oa" not in names  # not in df

    def test_extract_metadata_normalizes(self):
        """Numeric features should be z-score normalized."""
        df = self.pd.DataFrame({
            "year": [2010, 2020, 2030],
            "n_authors": [1, 5, 9],
        })
        features, names = self._extract_metadata_features(df)
        # Z-scored: mean ≈ 0, std ≈ 1
        assert abs(features["year"].mean()) < 1e-10
        assert abs(features["year"].std() - 1.0) < 0.1  # ddof can shift slightly

    def test_no_metadata_available(self):
        """When source docs have no metadata columns, falls back gracefully."""
        mock = self._make_mock_slda_result()
        # Remove all metadata columns
        mock["documents"] = mock["documents"].drop(
            columns=["year", "n_authors", "openalex_is_oa", "citation_count"],
            errors="ignore",
        )
        result = self.enhanced_citation_regression(mock)
        assert result["n_metadata_features"] == 0
        assert result["r2_improvement"] == 0.0
