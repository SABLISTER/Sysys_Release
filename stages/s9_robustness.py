"""Stage 9: Robustness analyses.

Six statistical tests to confirm that findings are not artifacts:
1. Bootstrap confidence intervals for mean relevance score
2. Split-half reliability (random splits + correlation)
3. Sensitivity analysis at different relevance thresholds
4. Permutation test on the condition–support gradient
5. Condition–entity co-occurrence network analysis
6. Co-occurring condition sensitivity (each configured condition's papers
   included vs excluded)

No LLM calls — pure statistics with numpy.

scipy is optional. Spearman rank correlation is always computed manually.
When scipy is present it supplies the exact Kendall tau-b p-value and the
ICC confidence interval; without it, Kendall tau-b falls back to a normal
approximation and the ICC is reported without a confidence interval.
"""

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from core.claims import PRIMARY_OUTCOME_FIELD, claim_flag, claim_list, claim_value
from core.io import atomic_write_json, atomic_write_text
from process.domain_patterns import contains_term, research_config, terms

logger = logging.getLogger(__name__)

# ── Optional scipy ────────────────────────────────────────────────────────
# Probed once at import so the stage degrades gracefully instead of raising
# ImportError deep inside an analysis. Catches Exception rather than
# ImportError because a broken install can fail with other error types.
_HAS_SCIPY = False
try:
    from scipy.stats import kendalltau as _sp_kendalltau, f as _sp_f_dist
    _HAS_SCIPY = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    _sp_kendalltau = None
    _sp_f_dist = None
    logger.info(
        "scipy unavailable (%s: %s) — Kendall tau p-values use a normal "
        "approximation and ICC confidence intervals are omitted",
        type(_exc).__name__, _exc,
    )

# ── Condition configuration ───────────────────────────────────────────────
# The permutation test, split-half profile, and condition sensitivity need a
# ranked list of co-occurring conditions: ``co_occurring_conditions`` in the
# research config, or the condition names of an explicitly selected pattern
# pack. Nothing is assumed when neither is configured.

def _load_condition_config(config) -> tuple[dict[str, str], list[str]]:
    """Derive condition patterns and ranking from the research config."""
    conditions = getattr(config, "co_occurring_conditions", None) or []
    if conditions:
        ranking = [str(c).strip() for c in conditions if str(c).strip()]
        patterns = {name.lower(): name for name in ranking}
        return patterns, ranking

    from process.domain_patterns import load_pack
    pack_conditions = load_pack(config).get("graph_analysis", {}).get("conditions") or {}
    ranking = [str(name).strip() for name in pack_conditions if str(name).strip()]
    return {name.lower(): name for name in ranking}, ranking


# ── Paper field helpers ───────────────────────────────────────────────────

def _get_relevance_score(paper: dict) -> int:
    """Get relevance score from a paper dict, defaulting to 0."""
    score = paper.get("relevance_score", 0)
    if score is None:
        return 0
    try:
        return int(score)
    except (ValueError, TypeError):
        return 0


@dataclass(frozen=True)
class RobustnessOutcome:
    """The extracted claim flag the robustness analyses count as "support".

    ``field`` names a boolean claim field (nested under ``claims`` or at the
    top level). When a paper has no value for it, ``fallback_terms`` are
    matched against its findings; with no terms there is no keyword fallback.
    """

    field: str = PRIMARY_OUTCOME_FIELD
    fallback_terms: tuple[str, ...] = ()

    def supports(self, paper: dict) -> bool:
        value = claim_value(paper, self.field)
        if isinstance(value, (bool, str)):
            return claim_flag(paper, self.field)
        if not self.fallback_terms:
            return False
        findings = " ".join(claim_list(paper, "findings"))
        return any(contains_term(findings, term) for term in self.fallback_terms)


DEFAULT_OUTCOME = RobustnessOutcome()


def _load_outcome(config) -> RobustnessOutcome:
    """Resolve the robustness outcome from ``robustness`` in the research config.

    ``robustness.outcome_field`` picks the claim flag (default
    ``supports_primary_question``) and ``robustness.outcome_terms`` the
    optional keyword fallback. For the legacy ``supports_criticality`` field
    the active pattern pack's ``criticality_terms`` serve as the fallback.
    """
    if config is None:
        return DEFAULT_OUTCOME
    section = research_config(config).get("robustness") or {}
    if not isinstance(section, dict):
        raise ValueError("robustness must be a mapping")
    field = str(section.get("outcome_field") or PRIMARY_OUTCOME_FIELD).strip()
    if "outcome_terms" in section:
        fallback = terms(section["outcome_terms"], "robustness.outcome_terms") \
            if section["outcome_terms"] else []
    elif field == "supports_criticality":
        from process.domain_patterns import extraction_patterns
        fallback = extraction_patterns(config).get("criticality_terms", [])
    else:
        fallback = []
    return RobustnessOutcome(field=field, fallback_terms=tuple(fallback))


def _normalize_condition(raw: str, patterns: dict[str, str]) -> Optional[str]:
    """Map raw text to a canonical condition name from the given patterns.

    Patterns match whole terms only, so "rett" never matches "Barrett".
    """
    for pattern, canonical in patterns.items():
        if contains_term(raw, pattern):
            return canonical
    return None


def _extract_conditions(paper: dict, patterns: dict[str, str]) -> list[str]:
    """Extract normalized conditions from a paper.

    Checks the ``conditions`` field (list of strings), ``title``,
    and ``abstract`` for condition pattern matches. ``patterns`` is required.
    """
    found: set[str] = set()

    # From the extracted conditions field (nested under claims or top level)
    for c in claim_list(paper, "conditions"):
        norm = _normalize_condition(c, patterns)
        if norm:
            found.add(norm)

    # From title + abstract text
    text = " ".join(filter(None, [
        paper.get("title", ""),
        paper.get("abstract", ""),
    ]))
    for pattern, canonical in patterns.items():
        if contains_term(text, pattern):
            found.add(canonical)

    return list(found)


# ── Spearman rank correlation (no scipy) ──────────────────────────────────

def _rank_array(values: list[float]) -> list[float]:
    """Assign fractional ranks to values (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find all tied values
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        # Assign average rank to ties (ranks are 1-based)
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation without scipy.

    Ranks both arrays, then computes Pearson correlation on the ranks.
    Returns 0.0 if the data is degenerate (constant values, too few points).
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    rx = _rank_array(x)
    ry = _rank_array(y)

    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return cov / denom


# ── Standardised effect size ─────────────────────────────────────────────

def cluster_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
    hedges_correct: bool = True,
) -> dict:
    """Cohen's d (and optionally Hedges' g) between two groups.

    Returns dict with d, hedges_g, ci_lower, ci_upper, magnitude, n_a, n_b.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    df = n_a + n_b - 2

    if df < 1 or n_a < 1 or n_b < 1:
        return {"d": 0.0, "hedges_g": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "magnitude": "negligible", "n_a": n_a, "n_b": n_b}

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0

    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / df)
    if pooled_sd < 1e-12:
        return {"d": 0.0, "hedges_g": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "magnitude": "negligible", "n_a": n_a, "n_b": n_b}

    d = (mean_a - mean_b) / pooled_sd

    # Hedges' correction factor J
    j = 1.0 - 3.0 / (4.0 * df - 1.0) if hedges_correct else 1.0
    g = d * j

    # SE of d (Borenstein et al., 2009)
    se = math.sqrt((n_a + n_b) / (n_a * n_b) + d * d / (2.0 * (n_a + n_b)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "d": round(d, 6),
        "hedges_g": round(g, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "magnitude": magnitude,
        "n_a": n_a,
        "n_b": n_b,
    }


# ── Kendall's τ and ICC ─────────────────────────────────────────────────

def _kendall_tau_b_numpy(x: list[float], y: list[float]) -> tuple[float, float]:
    """Kendall's tau-b and an approximate p-value, without scipy.

    Counts concordant/discordant pairs with a vectorised sign-product over the
    upper triangle, applies the standard tie corrections for tau-b, and derives
    a two-sided p-value from the normal approximation to the null distribution
    of the pair-count statistic. The p-value ignores the tie correction to the
    variance, so it is approximate; scipy is used instead whenever available.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    n = a.size

    sign_a = np.sign(a[:, None] - a[None, :])
    sign_b = np.sign(b[:, None] - b[None, :])
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    products = (sign_a * sign_b)[upper]
    concordant_minus_discordant = float(products.sum())

    n0 = n * (n - 1) / 2.0
    # Tie corrections: n1 over ties in x, n2 over ties in y.
    def _tie_term(values: np.ndarray) -> float:
        _, counts = np.unique(values, return_counts=True)
        return float((counts * (counts - 1) / 2.0).sum())

    n1 = _tie_term(a)
    n2 = _tie_term(b)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    tau = concordant_minus_discordant / denom if denom > 0 else 0.0

    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0:
        return tau, 1.0
    z = concordant_minus_discordant / math.sqrt(var)
    # Two-sided p from the standard normal survival function.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return tau, min(1.0, max(0.0, p))


def kendall_tau(x: list[float], y: list[float]) -> dict:
    """Kendall's tau-b with p-value (scipy when available, else approximate)."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return {"tau": 0.0, "p_value": 1.0}

    if _HAS_SCIPY:
        assert _sp_kendalltau is not None
        # scipy's kendalltau result is built via _make_tuple_bunch, which
        # has no static return type, so pyright sees an opaque element type.
        result = _sp_kendalltau(x, y)
        tau = cast(float, result[0])
        p = cast(float, result[1])
        approximate = False
    else:
        tau, p = _kendall_tau_b_numpy(x, y)
        approximate = True

    if tau is None or math.isnan(tau):
        tau = 0.0
    if p is None or math.isnan(p):
        p = 1.0
    return {
        "tau": float(tau),
        "p_value": float(p),
        "p_value_approximate": approximate,
    }


def icc_oneway(ratings: np.ndarray) -> dict:
    """ICC(1,1) — one-way random, single measures.

    Parameters
    ----------
    ratings : (n_subjects, k_raters) array

    Returns
    -------
    dict with icc, ci_lower, ci_upper
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n, k = ratings.shape
    if n < 3:
        return {"icc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    # One-way ANOVA decomposition
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)

    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((ratings - row_means[:, np.newaxis]) ** 2)

    df_between = n - 1
    df_within = n * (k - 1)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # ICC(1,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    denom = ms_between + (k - 1) * ms_within
    if denom < 1e-12:
        return {"icc": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    icc = (ms_between - ms_within) / denom

    # Confidence interval via F-distribution — requires scipy. The ICC point
    # estimate is pure ANOVA and always available, so report it either way
    # rather than failing the whole analysis.
    if not _HAS_SCIPY:
        return {
            "icc": float(icc),
            "ci_lower": None,
            "ci_upper": None,
            "ci_unavailable_reason": "scipy not installed",
        }

    assert _sp_f_dist is not None  # guaranteed by the _HAS_SCIPY check above
    f_dist = _sp_f_dist
    f_val = ms_between / ms_within if ms_within > 1e-12 else 1e6
    alpha = 0.05

    fl = f_val / f_dist.ppf(1 - alpha / 2, df_between, df_within)
    fu = f_val / f_dist.ppf(alpha / 2, df_between, df_within)

    ci_lower = (fl - 1) / (fl + k - 1)
    ci_upper = (fu - 1) / (fu + k - 1)

    return {
        "icc": round(float(icc), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
    }


# ── Analysis 1: Bootstrap CI ─────────────────────────────────────────────

def bootstrap_ci(papers: list[dict], n_iter: int = 10000,
                 seed: int = 42) -> dict:
    """Bootstrap confidence intervals for the mean relevance score.

    Resamples papers with replacement ``n_iter`` times and computes the
    mean relevance score for each resample.  Returns the overall mean,
    2.5th/97.5th percentile CI, standard deviation, and metadata.
    """
    logger.info("bootstrap_ci: START — %d papers, %d iterations", len(papers), n_iter)

    n = len(papers)
    if n < 3:
        logger.warning("bootstrap_ci: < 3 papers, returning defaults")
        return {
            "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "std": 0.0, "n_papers": n, "n_iterations": 0,
        }

    scores = np.array([_get_relevance_score(p) for p in papers], dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Vectorised bootstrap: generate all resample indices at once
    indices = rng.integers(0, n, size=(n_iter, n))
    means = scores[indices].mean(axis=1)

    result = {
        "mean": float(np.mean(means)),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "std": float(np.std(means)),
        "n_papers": n,
        "n_iterations": n_iter,
    }
    logger.info(
        "bootstrap_ci: DONE — mean=%.3f, 95%% CI=[%.3f, %.3f]",
        result["mean"], result["ci_lower"], result["ci_upper"],
    )
    return result


# ── Analysis 2: Split-half reliability ────────────────────────────────────

def _spearman_brown(r_half: float) -> float:
    """Project a half-length reliability to full length.

    The Spearman-Brown prophecy formula for doubling test length is
    ``r_full = 2 * r_half / (1 + r_half)``. Returns -1.0 at the singularity
    ``r_half == -1`` and clamps the result to [-1, 1].
    """
    denominator = 1.0 + r_half
    if abs(denominator) < 1e-12:
        return -1.0
    return max(-1.0, min(1.0, 2.0 * r_half / denominator))


def _bounded_pairs(
    values_a: list[float], values_b: list[float], max_pairs: int
) -> tuple[list[float], list[float]]:
    """Deterministically thin two paired sequences to at most ``max_pairs``.

    Kendall tau-b is computed with an O(n^2) sign-product matrix, so the paired
    profile values accumulated across every split must be capped before it is
    called. Stride sampling keeps the sample spread evenly over all splits
    without introducing randomness.
    """
    total = len(values_a)
    if total <= max_pairs:
        return values_a, values_b
    stride = math.ceil(total / max_pairs)
    return values_a[::stride], values_b[::stride]


def _not_estimable(note: str, categories: int = 0) -> dict:
    """Uniform 'reliability could not be estimated' result.

    Returning an explicit note beats emitting a plausible-looking number that
    does not measure anything, which is exactly how the previous
    implementation misled readers.
    """
    return {
        "mean_correlation": 0.0, "std_correlation": 0.0,
        "min_correlation": 0.0, "max_correlation": 0.0,
        "mean_half_correlation": 0.0,
        "kendall_tau": 0.0, "kendall_tau_p": 1.0,
        "icc": 0.0, "icc_ci_lower": None, "icc_ci_upper": None,
        "n_splits": 0, "profile_categories": categories,
        "spearman_brown_applied": True,
        "estimable": False,
        "interpretation": "not estimable",
        "note": note,
    }


def split_half(papers: list[dict], n_splits: int = 1000,
               seed: int = 42,
               condition_patterns: dict[str, str] | None = None,
               condition_ranking: list[str] | None = None,
               outcome: RobustnessOutcome = DEFAULT_OUTCOME,
               min_category_n: int = 3,
               min_categories: int = 3,
               max_agreement_pairs: int = 1000) -> dict:
    """Split-half reliability of the per-condition support profile.

    Randomly partitions the papers into two disjoint halves ``n_splits`` times.
    For each split it computes, independently in each half, the support
    rate for every configured condition. Those two rate vectors are correlated
    across conditions, and the resulting half-length correlation is projected
    to full corpus length with the Spearman-Brown formula
    ``r_full = 2*r_half / (1 + r_half)``.

    Why the profile is per-condition rates
    --------------------------------------
    Two earlier formulations were degenerate because they compared quantities
    that are *constrained to sum* across the two halves:

    * Correlating the halves' **mean scores** fails because the halves
      partition the corpus, so ``mean_a + mean_b == 2 * grand_mean`` is an
      identity. That pins the correlation — and the ICC built on it — to
      exactly -1.0 for even ``n``, whatever the data says.
    * Correlating the halves' **score distributions** fails for the same
      reason in compositional form: the per-level counts of the two halves sum
      to the corpus counts, so the two profiles are anti-correlated by
      construction.

    Per-condition rates avoid this. ``rate_a[c]`` and ``rate_b[c]`` are
    computed on disjoint paper sets and each has its own denominator, so they
    are two *independent estimates of the same underlying quantity* — the
    textbook precondition for a reliability coefficient. When conditions truly
    differ in support, both halves recover that pattern and the
    correlation is high; when the apparent gradient is sampling noise, it is
    low. That is precisely the question Stage 9 exists to answer.

    Conditions must be configured (``co_occurring_conditions``); without them
    there is no profile to replicate and the result is reported as not
    estimable rather than fabricated.
    """
    logger.info("split_half: START — %d papers, %d splits", len(papers), n_splits)

    n = len(papers)
    if n < 4:
        logger.warning("split_half: < 4 papers, reliability not estimable")
        return _not_estimable("fewer than 4 papers — reliability not estimable")

    ranking = list(condition_ranking or [])
    if not ranking:
        logger.warning(
            "split_half: no conditions configured — split-half reliability "
            "needs a per-condition profile to replicate"
        )
        return _not_estimable(
            "no conditions configured (`co_occurring_conditions`) — "
            "split-half reliability is not defined without a profile to replicate"
        )

    patterns = condition_patterns or {}

    # Extract conditions and the outcome flag once per paper. These are
    # regex scans over title+abstract, so doing them inside the split loop
    # would repeat the same work n_splits times.
    paper_conditions: list[set[str]] = []
    paper_supports: list[bool] = []
    for paper in papers:
        conditions = set(_extract_conditions(paper, patterns)) & set(ranking)
        paper_conditions.append(conditions)
        paper_supports.append(outcome.supports(paper))

    if not any(paper_conditions):
        return _not_estimable(
            "no paper matched any configured condition — "
            "reliability not estimable", categories=len(ranking),
        )

    def _rates(indices: np.ndarray) -> dict[str, float]:
        totals: dict[str, int] = {c: 0 for c in ranking}
        supporting: dict[str, int] = {c: 0 for c in ranking}
        for i in indices:
            supports = paper_supports[i]
            for condition in paper_conditions[i]:
                totals[condition] += 1
                if supports:
                    supporting[condition] += 1
        return {
            c: supporting[c] / totals[c]
            for c in ranking
            if totals[c] >= min_category_n
        }

    half = n // 2
    rng = np.random.default_rng(seed)

    corrected: list[float] = []
    half_corrs: list[float] = []
    paired_a: list[float] = []
    paired_b: list[float] = []
    categories_used: list[int] = []

    for _ in range(n_splits):
        perm = rng.permutation(n)
        rates_a = _rates(perm[:half])
        rates_b = _rates(perm[half:half + half])

        # Only conditions estimable in *both* halves can be compared.
        shared = [c for c in ranking if c in rates_a and c in rates_b]
        if len(shared) < min_categories:
            continue

        vec_a = np.array([rates_a[c] for c in shared], dtype=np.float64)
        vec_b = np.array([rates_b[c] for c in shared], dtype=np.float64)
        categories_used.append(len(shared))
        paired_a.extend(vec_a.tolist())
        paired_b.extend(vec_b.tolist())

        # A half with a flat profile carries no variance to correlate.
        if np.std(vec_a) < 1e-12 or np.std(vec_b) < 1e-12:
            continue
        r = float(np.corrcoef(vec_a, vec_b)[0, 1])
        if np.isnan(r):
            continue
        half_corrs.append(r)
        corrected.append(_spearman_brown(r))

    if not corrected:
        return _not_estimable(
            f"fewer than {min_categories} conditions had at least "
            f"{min_category_n} papers in both halves — reliability not estimable",
            categories=len(ranking),
        )

    mean_r = float(np.mean(corrected))
    std_r = float(np.std(corrected))
    min_r = float(np.min(corrected))
    max_r = float(np.max(corrected))
    mean_half = float(np.mean(half_corrs))

    # Agreement between paired half-profile values pooled across splits.
    sample_a, sample_b = _bounded_pairs(paired_a, paired_b, max_agreement_pairs)
    tau_result = kendall_tau(sample_a, sample_b)
    icc_result = icc_oneway(np.column_stack([sample_a, sample_b]))

    if mean_r > 0.8:
        interpretation = "excellent"
    elif mean_r > 0.6:
        interpretation = "good"
    elif mean_r > 0.4:
        interpretation = "fair"
    else:
        interpretation = "poor"

    result = {
        "mean_correlation": mean_r,
        "std_correlation": std_r,
        "min_correlation": min_r,
        "max_correlation": max_r,
        "mean_half_correlation": mean_half,
        "kendall_tau": tau_result["tau"],
        "kendall_tau_p": tau_result["p_value"],
        "icc": icc_result["icc"],
        "icc_ci_lower": icc_result["ci_lower"],
        "icc_ci_upper": icc_result["ci_upper"],
        "n_splits": len(corrected),
        "profile_categories": round(float(np.mean(categories_used))),
        "spearman_brown_applied": True,
        "estimable": True,
        "interpretation": interpretation,
    }
    logger.info(
        "split_half: DONE — Spearman-Brown r=%.3f (half r=%.3f), tau=%.3f, "
        "icc=%.3f (%s), std=%.3f",
        mean_r, mean_half, tau_result["tau"], icc_result["icc"],
        interpretation, std_r,
    )
    return result


# ── Analysis 3: Sensitivity analysis ──────────────────────────────────────

def sensitivity_analysis(papers: list[dict],
                         outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Test how results change at different relevance thresholds (1–5).

    For each threshold, counts papers passing, computes the percentage
    supporting the configured outcome, and the mean relevance score of passing papers.
    """
    logger.info("sensitivity_analysis: START — %d papers", len(papers))

    thresholds: dict[int, dict] = {}
    prev_direction: Optional[bool] = None
    direction_flipped = False

    for thresh in range(1, 6):
        passing = [p for p in papers if _get_relevance_score(p) >= thresh]
        n_pass = len(passing)

        if n_pass > 0:
            n_support = sum(1 for p in passing if outcome.supports(p))
            pct_support = (n_support / n_pass) * 100.0
            mean_score = sum(_get_relevance_score(p) for p in passing) / n_pass
        else:
            pct_support = 0.0
            mean_score = 0.0

        row = {
            "n_pass": n_pass,
            "pct_supporting": round(pct_support, 2),
            "mean_score": round(mean_score, 3),
        }

        # Effect size: passing vs failing papers on relevance scores
        failing = [p for p in papers if _get_relevance_score(p) < thresh]
        if n_pass >= 2 and len(failing) >= 2:
            pass_scores = np.array([_get_relevance_score(p) for p in passing],
                                   dtype=np.float64)
            fail_scores = np.array([_get_relevance_score(p) for p in failing],
                                   dtype=np.float64)
            row["effect_size"] = cluster_effect_size(pass_scores, fail_scores)

        thresholds[thresh] = row

        # Track whether the majority direction flips
        current_above_50 = pct_support > 50.0
        if prev_direction is not None and current_above_50 != prev_direction:
            direction_flipped = True
        if n_pass > 0:
            prev_direction = current_above_50

    stable = not direction_flipped

    result = {
        "thresholds": thresholds,
        "stable": stable,
    }
    logger.info(
        "sensitivity_analysis: DONE — stable=%s, thresholds tested=5",
        stable,
    )
    return result


# ── Analysis 4: Permutation test ──────────────────────────────────────────

def _condition_gradient(papers: list[dict],
                        patterns: dict[str, str],
                        ranking: list[str],
                        min_n: int = 3,
                        outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Compute per-condition support rates.

    Returns a dict mapping condition name to
    ``{"n_papers": int, "support_rate": float}``.
    Conditions with fewer than ``min_n`` papers are excluded.
    """
    counts: dict[str, dict] = {}
    for cond in ranking:
        counts[cond] = {"total": 0, "supporting": 0}

    for paper in papers:
        conds = _extract_conditions(paper, patterns)
        supports = outcome.supports(paper)
        for c in conds:
            if c in counts:
                counts[c]["total"] += 1
                if supports:
                    counts[c]["supporting"] += 1

    gradient: dict[str, dict] = {}
    for cond in ranking:
        total = counts[cond]["total"]
        if total >= min_n:
            rate = counts[cond]["supporting"] / total
            gradient[cond] = {
                "n_papers": total,
                "support_rate": round(rate, 4),
            }
        elif total > 0:
            logger.info(
                "permutation_test: skipping %s (n=%d < min_n=%d)",
                cond, total, min_n,
            )

    return gradient


def permutation_test(papers: list[dict],
                     patterns: dict[str, str],
                     ranking: list[str],
                     n_perm: int = 10000,
                     seed: int = 42,
                     outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Permutation test on the condition–support gradient.

    Tests whether the observed Spearman correlation between the configured
    condition ranking and each condition's support rate is
    significant by permuting condition labels.
    """
    logger.info("permutation_test: START — %d papers, %d permutations", len(papers), n_perm)

    gradient = _condition_gradient(papers, patterns, ranking,
                                   outcome=outcome)

    if len(gradient) < 3:
        logger.warning(
            "permutation_test: only %d conditions with enough papers, "
            "returning defaults", len(gradient),
        )
        return {
            "observed_correlation": 0.0,
            "p_value": 1.0,
            "n_permutations": 0,
            "significant": False,
            "condition_gradient": gradient,
        }

    # Build ordered vectors: condition rank (1-based) vs support rate.
    # Only include conditions present in the gradient.
    ranks = []
    rates = []
    for i, cond in enumerate(ranking):
        if cond in gradient:
            ranks.append(float(i + 1))
            rates.append(gradient[cond]["support_rate"])

    observed_r = _spearman_rank_correlation(ranks, rates)

    # Permutation: shuffle rates, recompute correlation
    rng = random.Random(seed)
    n_extreme = 0
    for _ in range(n_perm):
        shuffled = list(rates)
        rng.shuffle(shuffled)
        perm_r = _spearman_rank_correlation(ranks, shuffled)
        if abs(perm_r) >= abs(observed_r):
            n_extreme += 1

    p_value = n_extreme / n_perm

    result = {
        "observed_correlation": round(observed_r, 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perm,
        "significant": p_value < 0.05,
        "condition_gradient": gradient,
    }
    logger.info(
        "permutation_test: DONE — r=%.4f, p=%.4f, significant=%s",
        observed_r, p_value, result["significant"],
    )
    return result


# ── Analysis 6: Co-occurring condition sensitivity ────────────────────────

# Support-rate change (absolute) at which a condition is reported as
# materially shifting the pooled result.
MATERIAL_RATE_SHIFT = 0.05


def _support_rate(subset: list[dict],
                  outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> float:
    if not subset:
        return 0.0
    return sum(1 for p in subset if outcome.supports(p)) / len(subset)


def _single_condition_sensitivity(condition: str, with_cond: list[dict],
                                  without_cond: list[dict],
                                  outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Compare papers that mention ``condition`` against the rest of the corpus."""
    if not with_cond:
        return {"skipped": True, "n_with": 0, "n_without": len(without_cond),
                "reason": f"no papers mention {condition!r}"}

    rate_with = _support_rate(with_cond, outcome)
    rate_without = _support_rate(without_cond, outcome)
    diff = rate_with - rate_without
    es = cluster_effect_size(
        np.array([_get_relevance_score(p) for p in with_cond], dtype=np.float64),
        np.array([_get_relevance_score(p) for p in without_cond], dtype=np.float64),
    )
    shifts = abs(diff) >= MATERIAL_RATE_SHIFT
    return {
        "skipped": False,
        "n_with": len(with_cond),
        "n_without": len(without_cond),
        "support_rate_with": round(rate_with, 4),
        "support_rate_without": round(rate_without, 4),
        "rate_difference": round(diff, 4),
        "effect_size": es,
        "hedges_g": round(es.get("hedges_g", 0.0), 4),
        "interpretation": (
            f"Papers mentioning {condition} {'substantially' if shifts else 'do not substantially'}"
            f"{' shift' if shifts else ' alter'} the observed support rate"
        ),
    }


def condition_sensitivity_summary(sens: dict) -> str:
    """One-line summary for progress and GUI logs."""
    if not sens:
        return "Condition sensitivity: not run"
    if sens.get("skipped"):
        return f"Condition sensitivity skipped ({sens.get('reason', '')})"
    largest = sens.get("largest_shift")
    if not largest:
        return "Condition sensitivity: no configured condition appears in the corpus"
    res = sens["conditions"][largest]
    return (
        f"Condition sensitivity: {sens['n_conditions_tested']} tested; largest shift "
        f"{largest} ({res['rate_difference']:+.1%} support, g={res['hedges_g']:.3f})"
    )


def condition_sensitivity_analysis(papers: list[dict],
                                   patterns: dict[str, str],
                                   ranking: list[str],
                                   outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Test whether any configured co-occurring condition drives the result.

    For each condition in ``ranking``, splits the corpus into papers that
    mention it (via the same pattern matching as the other condition
    analyses) and papers that do not, then reports both support rates and
    Hedges' g on relevance scores. A large shift means the pooled finding
    depends on that condition's literature.
    """
    if not ranking:
        return {"skipped": True,
                "reason": "no co_occurring_conditions configured"}

    paper_conditions = [set(_extract_conditions(p, patterns)) for p in papers]
    per_condition: dict[str, dict] = {}
    for condition in ranking:
        with_cond = [p for p, conds in zip(papers, paper_conditions) if condition in conds]
        without_cond = [p for p, conds in zip(papers, paper_conditions) if condition not in conds]
        per_condition[condition] = _single_condition_sensitivity(
            condition, with_cond, without_cond, outcome
        )

    tested = {name: res for name, res in per_condition.items() if not res["skipped"]}
    largest = max(tested, key=lambda name: abs(tested[name]["rate_difference"]), default=None)
    logger.info("condition_sensitivity: tested %d/%d conditions, largest shift=%s",
                len(tested), len(ranking), largest)
    return {
        "skipped": False,
        "conditions": per_condition,
        "n_conditions_tested": len(tested),
        "largest_shift": largest,
    }


# ── Report writer ─────────────────────────────────────────────────────────

def _condition_sensitivity_report_lines(sens: dict) -> list[str]:
    """Markdown for analysis 6 (empty when it was skipped)."""
    if not sens or sens.get("skipped"):
        return []
    lines = [
        "## 6. Co-occurring Condition Sensitivity",
        "",
        "Each configured condition's papers are compared with the rest of the corpus.",
        "",
        "| Condition | With | Without | Support (with) | Support (without) "
        "| Difference | Hedges' g |",
        "|-----------|------|---------|----------------|-------------------"
        "|------------|-----------|",
    ]
    for name, res in sens.get("conditions", {}).items():
        if res.get("skipped"):
            lines.append(f"| {name} | 0 | {res.get('n_without', 0)} | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {res['n_with']} | {res['n_without']} | "
            f"{res['support_rate_with']:.1%} | {res['support_rate_without']:.1%} | "
            f"{res['rate_difference']:+.1%} | {res['hedges_g']:.2f} |"
        )
    largest = sens.get("largest_shift")
    if largest:
        lines += ["", f"- **Largest shift:** {largest} — "
                      f"{sens['conditions'][largest]['interpretation']}"]
    return lines + [""]


def _write_report(results: dict, path: Path, n_papers: int,
                  ranking: list[str] | None = None) -> None:
    """Write a markdown robustness report."""
    ranking = ranking or []
    lines = [
        "# Robustness Analysis Report",
        "",
        f"**Papers analysed:** {n_papers}",
        f"**Outcome measured:** `{results.get('outcome_field', PRIMARY_OUTCOME_FIELD)}` "
        "(share of papers whose extracted claims support it)",
        "",
    ]

    # Bootstrap CI
    bs = results.get("bootstrap", {})
    lines += [
        "## 1. Bootstrap Confidence Intervals",
        "",
        f"- **Mean relevance score:** {bs.get('mean', 0):.3f}",
        f"- **95% CI:** [{bs.get('ci_lower', 0):.3f}, {bs.get('ci_upper', 0):.3f}]",
        f"- **Std:** {bs.get('std', 0):.3f}",
        f"- **Iterations:** {bs.get('n_iterations', 0):,}",
        "",
    ]

    # Split-half
    sh = results.get("split_half", {})
    # ICC bounds are None when scipy is unavailable, so they cannot be
    # formatted with a float format spec.
    icc_lo = sh.get("icc_ci_lower")
    icc_hi = sh.get("icc_ci_upper")
    icc_ci = (
        f"[{icc_lo:.3f}, {icc_hi:.3f}]"
        if isinstance(icc_lo, (int, float)) and isinstance(icc_hi, (int, float))
        else "[CI unavailable — scipy not installed]"
    )
    lines += [
        "## 2. Split-Half Reliability",
        "",
    ]
    if not sh.get("estimable", False):
        lines += [
            f"> **Not estimable.** {sh.get('note', 'No result was produced.')}",
            "",
        ]
    else:
        lines += [
            "Random halves of the corpus are compared on their per-condition "
            "support rates; the half-length correlation is projected to "
            "full corpus length with the Spearman-Brown formula.",
            "",
            f"- **Reliability (Spearman-Brown corrected):** {sh.get('mean_correlation', 0):.3f}",
            f"- **Half-length correlation (uncorrected):** {sh.get('mean_half_correlation', 0):.3f}",
            f"- **Kendall τ:** {sh.get('kendall_tau', 0):.3f} (p = {sh.get('kendall_tau_p', 1):.4f})",
            f"- **ICC(1,1):** {sh.get('icc', 0):.3f} {icc_ci}",
            f"- **Std:** {sh.get('std_correlation', 0):.3f}",
            f"- **Range:** [{sh.get('min_correlation', 0):.3f}, {sh.get('max_correlation', 0):.3f}]",
            f"- **Interpretation:** {sh.get('interpretation', 'N/A')}",
            f"- **Splits contributing:** {sh.get('n_splits', 0):,}",
            f"- **Conditions in profile (mean):** {sh.get('profile_categories', 0)}",
            "",
        ]

    # Sensitivity
    sa = results.get("sensitivity", {})
    lines += [
        "## 3. Sensitivity Analysis",
        "",
        f"- **Stable across thresholds:** {sa.get('stable', 'N/A')}",
        "",
        "| Threshold | Papers | Supporting % | Mean Score | Cohen's d | Magnitude |",
        "|-----------|--------|--------------|------------|-----------|-----------|",
    ]
    for t in range(1, 6):
        info = sa.get("thresholds", {}).get(str(t), sa.get("thresholds", {}).get(t, {}))
        es = info.get("effect_size", {})
        d_str = f"{es.get('d', 0):.2f}" if es else "—"
        mag_str = es.get("magnitude", "—") if es else "—"
        lines.append(
            f"| {t} | {info.get('n_pass', 0)} | "
            f"{info.get('pct_supporting', 0):.1f}% | "
            f"{info.get('mean_score', 0):.3f} | "
            f"{d_str} | {mag_str} |"
        )
    lines.append("")

    # Permutation test
    pt = results.get("permutation", {})
    if ranking:
        lines += [
            "## 4. Permutation Test (Condition–Support Gradient)",
            "",
            f"- **Observed Spearman r:** {pt.get('observed_correlation', 0):.4f}",
            f"- **p-value:** {pt.get('p_value', 1):.4f}",
            f"- **Significant (p < 0.05):** {pt.get('significant', False)}",
            f"- **Permutations:** {pt.get('n_permutations', 0):,}",
            "",
            "### Condition Gradient",
            "",
            "| Condition | Papers | Support Rate |",
            "|-----------|--------|--------------|",
        ]
        gradient = pt.get("condition_gradient", {})
        for cond in ranking:
            if cond in gradient:
                info = gradient[cond]
                lines.append(
                    f"| {cond} | {info['n_papers']} | "
                    f"{info['support_rate']:.2%} |"
                )
        lines.append("")

    # Network analysis
    net = results.get("network_analysis", {})
    if net:
        comm = net.get("communities", {})
        lines += [
            "## 5. Condition–Entity Co-occurrence Networks",
            "",
            f"- **Edges:** {net.get('n_edges', 0)} "
            f"({net.get('n_significant_edges', '?')} significant)",
            f"- **Edge threshold:** {net.get('edge_threshold', '—')} "
            f"(permutation null, α=0.05)",
            f"- **Conditions found:** {net.get('n_conditions', 0)}",
            f"- **Target entities found:** {net.get('n_regions', 0)}",
            f"- **Communities:** {comm.get('n_communities', 0)} "
            f"(modularity = {comm.get('modularity', 0):.3f})",
            "",
        ]
        if net.get("skipped"):
            lines += [f"Network analysis skipped: {net.get('reason', '')}", ""]

    lines += _condition_sensitivity_report_lines(results.get("condition_sensitivity", {}))

    lines += [
        "---",
        "*Generated by SystS Stage 9: Robustness Analyses*",
    ]

    atomic_write_text(path, "\n".join(lines))
    logger.info("Robustness report written: %s", path)


def _write_network_report(edges: list[dict], path: Path,
                          n_papers: int, note: str = "") -> None:
    """Write a markdown report for the condition–entity networks."""
    lines = [
        "# Condition–Entity Co-occurrence Networks",
        "",
        f"**Papers analysed:** {n_papers}",
        f"**Edges detected:** {len(edges)}",
        "",
        "## Top Co-occurrences",
        "",
        "| Condition | Entity | Weight | Entity Type |",
        "|-----------|--------|--------|-------------|",
    ]

    if note:
        lines += [f"Network analysis skipped: {note}", ""]
    for edge in edges[:30]:
        lines.append(
            f"| {edge['source']} | {edge['target']} | "
            f"{edge['weight']} | {edge.get('target_type', 'entity')} |"
        )

    lines += [
        "",
        "---",
        "*Generated by SystS Stage 9: Network Analysis*",
    ]

    atomic_write_text(path, "\n".join(lines))
    logger.info("Network report written: %s", path)


# ── Main entry point ──────────────────────────────────────────────────────

def run(
    claims_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config=None,
    seed: int = 42,
    bootstrap_iterations: int = 10000,
    split_iterations: int = 1000,
    permutation_iterations: int = 10000,
    cancel_event=None,
    progress_callback=None,
    **kw,
) -> dict:
    """Run all 6 robustness analyses.

    Parameters
    ----------
    claims_path : str, optional
        Explicit path to claims JSON.  If not provided, standard locations
        under ``config.data_dir`` are searched.
    output_dir : str, optional
        Override for output directory.
    config : Config, optional
        Pipeline configuration object.
    seed : int
        Random seed for reproducibility.
    bootstrap_iterations, split_iterations, permutation_iterations : int
        Iteration counts for each analysis.
    cancel_event : threading.Event or similar, optional
        Set to cancel the run.
    progress_callback : callable, optional
        Called with progress message strings.
    """

    def _cancelled() -> bool:
        if cancel_event is None:
            return False
        if hasattr(cancel_event, "is_set"):
            return cancel_event.is_set()
        return bool(cancel_event)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    _progress("═" * 60)
    _progress("STAGE 9: ROBUSTNESS ANALYSES")
    _progress("═" * 60)

    # ── Resolve paths ─────────────────────────────────────────────────
    project_root = config.project_root if config else Path.cwd()
    data_dir = project_root / "data"
    validation_dir = data_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # ── Load claims corpus ────────────────────────────────────────────
    corpus_path = None
    if claims_path:
        corpus_path = Path(claims_path)
    else:
        candidates = [
            data_dir / "claims" / "claims_filtered.json",
            data_dir / "claims" / "claims.json",
            data_dir / "corpus" / "relevant.json",
        ]
        if config:
            candidates = [
                config.claims_dir / "claims_filtered.json",
                config.claims_dir / "claims.json",
                config.corpus_dir / "relevant.json",
            ] + candidates
        for cand in candidates:
            if cand.exists():
                corpus_path = cand
                break

    if corpus_path is None or not corpus_path.exists():
        _progress("ERROR: No claims corpus found. Run earlier stages first.")
        return {"error": "No claims corpus found"}

    with open(corpus_path, encoding="utf-8") as f:
        papers = json.load(f)
    if not isinstance(papers, list):
        papers = list(papers.values()) if isinstance(papers, dict) else []

    n_papers = len(papers)
    _progress(f"Loaded {n_papers} papers from {corpus_path}")

    # Load condition configuration for the permutation and sensitivity analyses.
    condition_patterns, condition_ranking = _load_condition_config(config)
    if condition_ranking:
        _progress(f"Condition ranking: {condition_ranking}")
    else:
        _progress("No co_occurring_conditions configured; condition-gradient analyses will be skipped.")

    outcome = _load_outcome(config)
    _progress(f"Outcome measured: {outcome.field}")

    if n_papers == 0:
        _progress("WARNING: Empty corpus — returning default results")
        empty_result = {
            "n_papers": 0,
            "bootstrap": {},
            "split_half": {},
            "sensitivity": {},
            "permutation": {},
            "network_analysis": {},
        }
        atomic_write_json(validation_dir / "robustness_results.json", empty_result)
        return empty_result

    results: dict[str, Any] = {"n_papers": n_papers, "outcome_field": outcome.field}

    # ── Analysis 1: Bootstrap CI ──────────────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 1/6: Bootstrap confidence intervals...")
    results["bootstrap"] = bootstrap_ci(
        papers, n_iter=bootstrap_iterations, seed=seed,
    )
    _progress(
        f"  Bootstrap: mean={results['bootstrap']['mean']:.3f}, "
        f"95% CI=[{results['bootstrap']['ci_lower']:.3f}, "
        f"{results['bootstrap']['ci_upper']:.3f}]"
    )

    # ── Analysis 2: Split-half reliability ────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 2/6: Split-half reliability...")
    results["split_half"] = split_half(
        papers, n_splits=split_iterations, seed=seed,
        condition_patterns=condition_patterns,
        condition_ranking=condition_ranking,
        outcome=outcome,
    )
    _sh = results["split_half"]
    if _sh.get("estimable", False):
        _progress(
            f"  Split-half: Spearman-Brown r={_sh['mean_correlation']:.3f}, "
            f"τ={_sh.get('kendall_tau', 0):.3f}, "
            f"ICC={_sh.get('icc', 0):.3f} "
            f"({_sh['interpretation']})"
        )
    else:
        _progress(f"  Split-half: not estimable — {_sh.get('note', '')}")

    # ── Analysis 3: Sensitivity analysis ──────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 3/6: Sensitivity analysis (thresholds 1-5)...")
    results["sensitivity"] = sensitivity_analysis(
        papers, outcome=outcome)
    _progress(
        f"  Sensitivity: stable={results['sensitivity']['stable']}"
    )

    # ── Analysis 4: Permutation test ──────────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 4/6: Permutation test (condition gradient)...")
    results["permutation"] = permutation_test(
        papers,
        condition_patterns,
        condition_ranking,
        n_perm=permutation_iterations,
        seed=seed,
        outcome=outcome,
    )
    _progress(
        f"  Permutation: r={results['permutation']['observed_correlation']:.4f}, "
        f"p={results['permutation']['p_value']:.4f}, "
        f"significant={results['permutation']['significant']}"
    )

    # ── Analysis 5: Network analysis ──────────────────────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 5/6: Condition–entity co-occurrence networks...")
    try:
        from process.graph_analysis import (
            load_patterns,
            build_cooccurrence_edges,
            compute_edge_threshold,
            detect_communities,
        )

        patterns = load_patterns(config)
        edges = build_cooccurrence_edges(papers, patterns)

        # Data-driven thresholding
        threshold = compute_edge_threshold(
            papers, patterns, n_permutations=500, seed=seed,
        )
        for e in edges:
            e["significant"] = e["weight"] >= threshold
        n_significant = sum(1 for e in edges if e["significant"])
        _progress(f"  Edge threshold: {threshold} ({n_significant}/{len(edges)} significant)")

        # Community detection
        community_result = detect_communities(
            [e for e in edges if e["significant"]]
        )
        _progress(
            f"  Communities: {community_result['n_communities']} "
            f"(modularity={community_result['modularity']:.3f})"
        )

        # Summarise network
        conditions_found = set()
        regions_found = set()
        for e in edges:
            conditions_found.add(e["source"])
            regions_found.add(e["target"])

        network_result = {
            "n_edges": len(edges),
            "n_significant_edges": n_significant,
            "edge_threshold": threshold,
            "n_conditions": len(conditions_found),
            "n_regions": len(regions_found),
            "top_edges": edges[:20],
            "edges": edges,
            "communities": community_result,
        }
        if not patterns["conditions"] or not any(
            values for category, values in patterns.items() if category != "conditions"
        ):
            network_result["skipped"] = True
            network_result["reason"] = "Configure graph_analysis conditions and target entities to enable network analysis."
            _progress(network_result["reason"])
        results["network_analysis"] = network_result

        # Save network JSON (with threshold + communities)
        network_path = validation_dir / "condition_networks.json"
        results["network_path"] = str(network_path)
        atomic_write_json(network_path, {
            "n_papers": n_papers,
            "n_edges": len(edges),
            "n_significant_edges": n_significant,
            "edge_threshold": threshold,
            "conditions": sorted(conditions_found),
            "regions": sorted(regions_found),
            "edges": edges,
            "communities": community_result,
            "skipped": network_result.get("skipped", False),
            "reason": network_result.get("reason", ""),
        })
        _progress(f"  Networks: {len(edges)} edges, {len(conditions_found)} conditions, "
                  f"{len(regions_found)} target entities")
        _progress(f"  Saved: {network_path}")

        # Write network report
        net_report_path = validation_dir / "condition_networks_report.md"
        _write_network_report(edges, net_report_path, n_papers, network_result.get("reason", ""))
        results["network_report_path"] = str(net_report_path)

    except Exception as exc:
        logger.warning("Network analysis failed: %s", exc, exc_info=True)
        _progress(f"  Network analysis error: {exc}")
        results["network_analysis"] = {"error": str(exc)}

    # ── Analysis 6: Co-occurring condition sensitivity ────────────────
    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    _progress("Analysis 6/6: Co-occurring condition sensitivity...")
    results["condition_sensitivity"] = condition_sensitivity_analysis(
        papers, condition_patterns, condition_ranking,
        outcome=outcome,
    )
    _progress("  " + condition_sensitivity_summary(results["condition_sensitivity"]))

    # ── Save combined results ─────────────────────────────────────────
    results_path = validation_dir / "robustness_results.json"
    # Make a serialisable copy (strip full edge list from top-level)
    save_results = dict(results)
    if "network_analysis" in save_results:
        net = dict(save_results["network_analysis"])
        net.pop("edges", None)  # Too large for the summary JSON
        save_results["network_analysis"] = net
    atomic_write_json(results_path, save_results)
    _progress(f"Results saved: {results_path}")
    results["results_path"] = str(results_path)

    # ── Write robustness report ───────────────────────────────────────
    report_path = validation_dir / "robustness_report.md"
    _write_report(results, report_path, n_papers, condition_ranking)
    results["report_path"] = str(report_path)

    _progress("═" * 60)
    _progress("STAGE 9 COMPLETE")
    _progress("═" * 60)

    return results


# ── Output checker ────────────────────────────────────────────────────────

def check_output(config=None) -> dict:
    """Check whether Stage 9 output already exists."""
    if config is None:
        return {"exists": False}
    path = config.data_dir / "validation" / "robustness_results.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "exists": True,
                "n_papers": data.get("n_papers", 0),
                "path": str(path),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return {"exists": False}
