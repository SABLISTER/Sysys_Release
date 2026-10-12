"""
Meta-analytic effect size pooling for Sysys.

Implements DerSimonian-Laird random-effects meta-analysis,
heterogeneity statistics, and publication bias tests.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

STAGE2C_META_ANALYSIS_INPUT = "statistical_meta_analysis_input.json"
STAGE7_FULLTEXT_METRICS = "auto_extracted_metrics.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StudyEffect:
    """A single study's effect size data."""
    article_id: str
    title: str = ""
    effect_size: float = 0.0       # Hedges' g or Cohen's d
    standard_error: float = 0.0
    variance: float = 0.0
    sample_size: int = 0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    weight: float = 0.0            # inverse-variance weight (computed)
    component_id: str = ""
    condition: str = ""


@dataclass
class PooledEstimate:
    """Result of a random-effects meta-analysis."""
    pooled_effect: float = 0.0
    pooled_se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    z_value: float = 0.0
    p_value: float = 0.0
    tau_squared: float = 0.0       # between-study variance
    i_squared: float = 0.0         # heterogeneity %
    q_statistic: float = 0.0       # Cochran's Q
    q_p_value: float = 0.0
    n_studies: int = 0
    prediction_interval: tuple = (0.0, 0.0)
    studies: list = field(default_factory=list)  # list of StudyEffect


@dataclass
class BiasTestResult:
    """Results from publication bias testing."""
    # Egger's regression test
    eggers_intercept: float = 0.0
    eggers_se: float = 0.0
    eggers_t: float = 0.0
    eggers_p: float = 0.0
    eggers_interpretation: str = ""

    # Trim-and-fill
    trimfill_n_imputed: int = 0
    trimfill_adjusted_effect: float = 0.0
    trimfill_adjusted_ci_lower: float = 0.0
    trimfill_adjusted_ci_upper: float = 0.0
    trimfill_original_effect: float = 0.0

    # Fail-safe N (Rosenthal)
    failsafe_n: int = 0
    failsafe_interpretation: str = ""

    n_studies: int = 0
    overall_interpretation: str = ""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_se_from_ci(ci_lower: float, ci_upper: float, n: int = 0) -> float:
    """Compute standard error from a 95% CI.

    SE = (upper - lower) / (2 * 1.96)
    """
    width = ci_upper - ci_lower
    if width <= 0:
        return 0.0
    return width / (2.0 * 1.96)


def compute_se_from_n(effect_size: float, n: int) -> float:
    """Approximate SE from effect size and sample size.

    Assumes equal groups: SE ~ sqrt(4/n + d^2 / (2*n))
    """
    if n <= 0:
        return 0.0
    return math.sqrt(4.0 / n + effect_size ** 2 / (2.0 * n))


def hedges_correction(d: float, n: int) -> float:
    """Apply Hedges' small-sample correction factor J.

    J = 1 - 3 / (4*df - 1) where df = n - 2
    """
    df = n - 2
    if df <= 0:
        return d
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * j


# ---------------------------------------------------------------------------
# Core meta-analysis
# ---------------------------------------------------------------------------

def pool_random_effects(studies: list[StudyEffect], method: str = "DL") -> PooledEstimate:
    """Random-effects meta-analysis with DerSimonian-Laird or REML estimator.

    Parameters
    ----------
    method : str
        "DL" (DerSimonian-Laird, default) or "REML" (restricted maximum
        likelihood via Fisher scoring, using DL as starting value).

    Steps:
    1. Compute fixed-effect weights: w_i = 1/v_i (inverse variance)
    2. Compute Cochran's Q = sum(w_i * (y_i - y_fixed)^2)
    3. Estimate tau^2 via DL or REML
    4. Compute random-effects weights: w_i* = 1/(v_i + tau^2)
    5. Pool: y_RE = sum(w_i* * y_i) / sum(w_i*)
    6. SE_RE = sqrt(1/sum(w_i*))
    7. I^2 = max(0, (Q - (k-1)) / Q * 100)
    8. Prediction interval: y_RE +/- t(df=k-2, 0.975) * sqrt(SE^2 + tau^2)
    """
    k = len(studies)
    if k == 0:
        return PooledEstimate()
    if k == 1:
        s = studies[0]
        return PooledEstimate(
            pooled_effect=s.effect_size,
            pooled_se=s.standard_error,
            ci_lower=s.effect_size - 1.96 * s.standard_error,
            ci_upper=s.effect_size + 1.96 * s.standard_error,
            z_value=s.effect_size / s.standard_error if s.standard_error > 0 else 0.0,
            p_value=float(2.0 * (1.0 - sp_stats.norm.cdf(abs(s.effect_size / s.standard_error)))) if s.standard_error > 0 else 1.0,
            n_studies=1,
            studies=list(studies),
        )

    effects = np.array([s.effect_size for s in studies])
    variances = np.array([s.variance for s in studies])

    # Guard against zero variances
    variances = np.where(variances > 0, variances, 1e-10)

    # Step 1: Fixed-effect weights
    w_fe = 1.0 / variances
    sum_w = np.sum(w_fe)

    # Fixed-effect pooled estimate
    y_fe = np.sum(w_fe * effects) / sum_w

    # Step 2: Cochran's Q
    q_stat = float(np.sum(w_fe * (effects - y_fe) ** 2))
    q_df = k - 1
    q_p = float(1.0 - sp_stats.chi2.cdf(q_stat, q_df)) if q_df > 0 else 1.0

    # Step 3: Estimate tau^2
    c = sum_w - np.sum(w_fe ** 2) / sum_w
    tau2 = max(0.0, (q_stat - q_df) / c) if c > 0 else 0.0

    # Step 3b: REML refinement via Fisher scoring (if requested)
    if method.upper() == "REML" and k >= 3:
        tau2_reml = tau2  # start from DL estimate
        max_iter = 50
        tol = 1e-8
        for _iter in range(max_iter):
            w_r = 1.0 / (variances + tau2_reml)
            sw = np.sum(w_r)
            # REML gradient: 0.5 * (sum(w_r^2*(y_i - y_hat)^2) - sum(w_r) + sum(w_r^2)/sum(w_r))
            y_hat = np.sum(w_r * effects) / sw
            resid2 = (effects - y_hat) ** 2
            # First derivative of restricted log-likelihood w.r.t. tau2
            dl_dtau2 = 0.5 * (np.sum(w_r ** 2 * resid2) - np.sum(w_r) + np.sum(w_r ** 2) / sw)
            # Expected information (Fisher information)
            # Simplified: info ≈ 0.5 * sum(w_r^2)  for the one-step approximation
            info = 0.5 * np.sum(w_r ** 2)
            if info < 1e-15:
                break
            step = dl_dtau2 / info
            tau2_new = max(0.0, tau2_reml + step)
            if abs(tau2_new - tau2_reml) < tol:
                tau2_reml = tau2_new
                break
            tau2_reml = tau2_new
        tau2 = tau2_reml

    # Step 4: Random-effects weights
    w_re = 1.0 / (variances + tau2)
    sum_w_re = np.sum(w_re)

    # Step 5: Pooled random-effects estimate
    y_re = float(np.sum(w_re * effects) / sum_w_re)

    # Step 6: Standard error
    se_re = float(np.sqrt(1.0 / sum_w_re))

    # Step 7: I^2
    i2 = max(0.0, (q_stat - q_df) / q_stat * 100.0) if q_stat > 0 else 0.0

    # Z-test
    z_val = y_re / se_re if se_re > 0 else 0.0
    p_val = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z_val))))

    # 95% CI
    ci_lo = y_re - 1.96 * se_re
    ci_hi = y_re + 1.96 * se_re

    # Step 8: Prediction interval
    pred_lo, pred_hi = 0.0, 0.0
    if k > 2:
        t_crit = float(sp_stats.t.ppf(0.975, k - 2))
        pred_se = math.sqrt(se_re ** 2 + tau2)
        pred_lo = y_re - t_crit * pred_se
        pred_hi = y_re + t_crit * pred_se

    # Update study weights
    w_re_list = w_re.tolist()
    max_w = max(w_re_list) if w_re_list else 1.0
    for i, s in enumerate(studies):
        s.weight = w_re_list[i] / max_w if max_w > 0 else 0.0

    return PooledEstimate(
        pooled_effect=round(y_re, 6),
        pooled_se=round(se_re, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        z_value=round(z_val, 4),
        p_value=round(p_val, 6),
        tau_squared=round(tau2, 6),
        i_squared=round(i2, 2),
        q_statistic=round(q_stat, 4),
        q_p_value=round(q_p, 6),
        n_studies=k,
        prediction_interval=(round(pred_lo, 4), round(pred_hi, 4)),
        studies=list(studies),
    )


def resolve_meta_analysis_input(
    config,
    *,
    input_source: str = "stage2c",
    metrics_path: Optional[str | Path] = None,
) -> tuple[Path, str]:
    """Resolve the explicitly selected meta-analysis input artifact."""
    source = str(input_source or "stage2c").strip().lower()
    if metrics_path is not None:
        path = Path(metrics_path)
        expected_paths = {
            "stage2c": config.claims_dir / STAGE2C_META_ANALYSIS_INPUT,
            "stage7": config.data_dir / "fulltext" / STAGE7_FULLTEXT_METRICS,
            "legacy": config.output_dir / STAGE7_FULLTEXT_METRICS,
        }
        if source in expected_paths and path == expected_paths[source]:
            return path, source
        return path, "custom"

    if source == "stage2c":
        return config.claims_dir / STAGE2C_META_ANALYSIS_INPUT, source
    if source == "stage7":
        return config.data_dir / "fulltext" / STAGE7_FULLTEXT_METRICS, source
    if source == "legacy":
        return config.output_dir / STAGE7_FULLTEXT_METRICS, source
    raise ValueError(
        "input_source must be one of: 'stage2c', 'stage7', 'legacy', "
        "or pass metrics_path for an explicit file"
    )


def _first_numeric(values: list) -> float | None:
    for value in values:
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_effect_sizes(
    config,
    *,
    metrics_path: Optional[str | Path] = None,
    input_source: str = "stage2c",
) -> list[StudyEffect]:
    """Load effect sizes from the explicitly selected meta-analysis input.

    The file contains a list of dicts, each with:
    - "article_id" or "doi"
    - "effect_sizes": list of floats
    - "sample_sizes": list of ints
    - "confidence_intervals": list of [lower, upper] pairs

    For papers with multiple effect sizes, use the first one.
    Compute SE from CI when available, or from sample size as fallback.
    Skip papers with no extractable effect size.
    """
    metrics_path, resolved_source = resolve_meta_analysis_input(
        config,
        input_source=input_source,
        metrics_path=metrics_path,
    )
    if not metrics_path.exists():
        logger.warning("No %s meta-analysis input found at %s", resolved_source, metrics_path)
        return []

    try:
        with open(metrics_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load metrics: %s", exc)
        return []

    if not isinstance(raw, list):
        logger.warning("Meta-analysis input is not a list: %s", metrics_path)
        return []

    logger.info("Loading meta-analysis input from %s (%s)", metrics_path, resolved_source)
    studies: list[StudyEffect] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue

        article_id = entry.get("article_id", entry.get("doi", ""))
        if not article_id:
            continue

        metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
        effect_sizes = entry.get("effect_sizes", metrics.get("effect_sizes", []))
        if not effect_sizes or not isinstance(effect_sizes, list):
            continue

        es = _first_numeric(effect_sizes)
        if es is None:
            continue

        sample_sizes = entry.get("sample_sizes", metrics.get("sample_sizes", []))
        cis = entry.get("confidence_intervals", metrics.get("confidence_intervals", []))
        title = entry.get("title", "")
        component_id = entry.get("component_id", "")
        condition = entry.get("condition", "")

        n = 0
        if sample_sizes and isinstance(sample_sizes, list):
            try:
                first_sample = sample_sizes[0]
                if isinstance(first_sample, dict):
                    first_sample = first_sample.get("value", first_sample.get("n"))
                if first_sample is not None:
                    n = int(first_sample)
            except (TypeError, ValueError, IndexError):
                pass

        # Apply Hedges' correction if sample size available
        if n > 2:
            es = hedges_correction(es, n)

        # Compute SE: prefer CI, then sample size
        se = 0.0
        if cis and isinstance(cis, list) and len(cis) > 0:
            ci = cis[0]
            if isinstance(ci, dict):
                ci = [ci.get("lower"), ci.get("upper")]
            if (
                isinstance(ci, (list, tuple))
                and len(ci) == 2
                and ci[0] is not None
                and ci[1] is not None
            ):
                try:
                    se = compute_se_from_ci(float(ci[0]), float(ci[1]))
                except (TypeError, ValueError):
                    pass

        if se <= 0 and n > 0:
            se = compute_se_from_n(es, n)

        if se <= 0:
            logger.warning(
                "Skipping study %s: cannot compute standard error from CI or sample size",
                article_id,
            )
            continue

        variance = se ** 2
        ci_lo = es - 1.96 * se
        ci_hi = es + 1.96 * se

        studies.append(StudyEffect(
            article_id=article_id,
            title=title,
            effect_size=round(es, 6),
            standard_error=round(se, 6),
            variance=round(variance, 8),
            sample_size=n,
            ci_lower=round(ci_lo, 6),
            ci_upper=round(ci_hi, 6),
            component_id=component_id,
            condition=condition,
        ))

    logger.info("Loaded %d studies with effect sizes from %s.", len(studies), metrics_path)
    return studies


def pool_by_condition(studies: list[StudyEffect]) -> dict[str, PooledEstimate]:
    """Group studies by condition and pool each group."""
    groups: dict[str, list[StudyEffect]] = {}
    for s in studies:
        key = s.condition if s.condition else "overall"
        groups.setdefault(key, []).append(s)
    return {k: pool_random_effects(v) for k, v in groups.items() if len(v) >= 2}


def pool_by_component(studies: list[StudyEffect]) -> dict[str, PooledEstimate]:
    """Group studies by component_id and pool each group."""
    groups: dict[str, list[StudyEffect]] = {}
    for s in studies:
        key = s.component_id if s.component_id else "overall"
        groups.setdefault(key, []).append(s)
    return {k: pool_random_effects(v) for k, v in groups.items() if len(v) >= 2}


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    p_values : dict mapping test name → raw p-value
    alpha : significance threshold (default 0.05)

    Returns
    -------
    dict mapping test name → {"raw_p", "adjusted_p", "significant"}
    """
    if not p_values:
        return {}

    # Sort by p-value ascending
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)

    # Compute adjusted p-values: p_adj[i] = p[i] * m / rank
    adjusted = []
    for rank_1based, (name, raw_p) in enumerate(sorted_items, start=1):
        adj = raw_p * m / rank_1based
        adjusted.append((name, raw_p, adj))

    # Enforce monotonicity (step-up): walk backwards, each must be ≤ successor
    for i in range(len(adjusted) - 2, -1, -1):
        name, raw_p, adj = adjusted[i]
        _, _, adj_next = adjusted[i + 1]
        adjusted[i] = (name, raw_p, min(adj, adj_next))

    # Cap at 1.0 and build result
    result = {}
    for name, raw_p, adj in adjusted:
        adj_capped = min(adj, 1.0)
        result[name] = {
            "raw_p": raw_p,
            "adjusted_p": round(adj_capped, 10),
            "significant": adj_capped < alpha,
        }

    return result


# ---------------------------------------------------------------------------
# Publication bias tests
# ---------------------------------------------------------------------------

def eggers_test(studies: list[StudyEffect]) -> dict:
    """Egger's regression test for funnel plot asymmetry.

    Regress (effect / SE) on (1/SE).
    Test if the intercept is significantly different from 0.
    Significant intercept (p < 0.1 conventionally) suggests asymmetry.

    Needs >= 10 studies to be meaningful.
    Uses simple OLS via numpy.
    """
    k = len(studies)
    if k < 3:
        return {
            "intercept": 0.0, "se": 0.0, "t_value": 0.0, "p_value": 1.0,
            "interpretation": "Too few studies for Egger's test.",
            "n_studies": k,
        }

    precision = np.array([1.0 / s.standard_error for s in studies if s.standard_error > 0])
    std_effect = np.array([s.effect_size / s.standard_error for s in studies if s.standard_error > 0])

    if len(precision) < 3:
        return {
            "intercept": 0.0, "se": 0.0, "t_value": 0.0, "p_value": 1.0,
            "interpretation": "Too few studies with valid SE for Egger's test.",
            "n_studies": len(precision),
        }

    # OLS: std_effect = intercept + slope * precision
    n = len(precision)
    X = np.column_stack([np.ones(n), precision])
    beta = np.linalg.lstsq(X, std_effect, rcond=None)[0]
    intercept = float(beta[0])
    residuals = std_effect - X @ beta
    mse = float(np.sum(residuals ** 2) / (n - 2))
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_intercept = float(np.sqrt(var_beta[0, 0]))

    t_val = intercept / se_intercept if se_intercept > 0 else 0.0
    p_val = float(2.0 * (1.0 - sp_stats.t.cdf(abs(t_val), n - 2)))

    if k < 10:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"Note: only {k} studies; test has low power with < 10 studies."
        )
    elif p_val < 0.05:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"Significant asymmetry detected — publication bias likely."
        )
    elif p_val < 0.10:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"Borderline asymmetry — possible publication bias."
        )
    else:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"No significant asymmetry detected."
        )

    return {
        "intercept": round(intercept, 4),
        "se": round(se_intercept, 4),
        "t_value": round(t_val, 4),
        "p_value": round(p_val, 4),
        "interpretation": interpretation,
        "n_studies": k,
    }


def trim_and_fill(studies: list[StudyEffect]) -> dict:
    """Duval & Tweedie trim-and-fill procedure.

    1. Pool original studies to get initial estimate
    2. Compute residuals (distance from pooled effect)
    3. Rank by absolute residual
    4. Estimate number of asymmetric studies via the R0 estimator
    5. Trim the k0 most extreme studies on the positive side
    6. Re-pool with trimmed set, then fill mirror-image studies
    7. Re-pool again with original + imputed studies
    """
    k = len(studies)
    if k < 3:
        return {
            "n_imputed": 0,
            "adjusted_effect": 0.0,
            "adjusted_ci_lower": 0.0,
            "adjusted_ci_upper": 0.0,
            "original_effect": 0.0,
            "interpretation": "Too few studies for trim-and-fill.",
        }

    # Initial pooled estimate
    initial = pool_random_effects(studies)
    theta0 = initial.pooled_effect

    effects = np.array([s.effect_size for s in studies])

    # Residuals from pooled estimate
    residuals = effects - theta0

    # R0 estimator: number of studies to impute
    # Sort by residual value (not abs)
    sorted_idx = np.argsort(residuals)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, k + 1)

    # Count studies on the side with more extreme positive residuals
    # (assumes positive bias — larger effects overrepresented)
    n_positive = int(np.sum(residuals > 0))
    n_negative = int(np.sum(residuals < 0))

    # R0 estimator: k0 = max(0, round((4*T - k) / 2))
    # where T = sum of ranks of positive residuals
    if n_positive > n_negative:
        # Bias is on the positive side
        t_sum = float(np.sum(ranks[residuals > 0]))
        k0 = max(0, round((4.0 * t_sum - k * (k + 1)) / (2.0 * k)))
    elif n_negative > n_positive:
        # Bias is on the negative side
        t_sum = float(np.sum(ranks[residuals < 0]))
        k0 = max(0, round((4.0 * t_sum - k * (k + 1)) / (2.0 * k)))
    else:
        k0 = 0

    if k0 == 0:
        return {
            "n_imputed": 0,
            "adjusted_effect": round(theta0, 6),
            "adjusted_ci_lower": round(initial.ci_lower, 6),
            "adjusted_ci_upper": round(initial.ci_upper, 6),
            "original_effect": round(theta0, 6),
            "interpretation": "No asymmetry detected; no studies imputed.",
        }

    # Trim the k0 most extreme positive-side studies and re-estimate
    abs_residuals = np.abs(residuals)
    trim_idx = np.argsort(-abs_residuals)[:k0]
    trimmed = [s for i, s in enumerate(studies) if i not in set(trim_idx)]

    if len(trimmed) < 2:
        trimmed = list(studies)

    trimmed_pool = pool_random_effects(trimmed)
    theta_trim = trimmed_pool.pooled_effect

    # Create mirror-image (imputed) studies
    imputed: list[StudyEffect] = []
    for idx in trim_idx:
        s = studies[idx]
        mirror_effect = 2.0 * theta_trim - s.effect_size
        imputed.append(StudyEffect(
            article_id=f"imputed_{s.article_id}",
            title=f"[Imputed] {s.title}",
            effect_size=round(mirror_effect, 6),
            standard_error=s.standard_error,
            variance=s.variance,
            sample_size=s.sample_size,
        ))

    # Re-pool with original + imputed
    all_studies = list(studies) + imputed
    adjusted = pool_random_effects(all_studies)

    return {
        "n_imputed": k0,
        "adjusted_effect": round(adjusted.pooled_effect, 6),
        "adjusted_ci_lower": round(adjusted.ci_lower, 6),
        "adjusted_ci_upper": round(adjusted.ci_upper, 6),
        "original_effect": round(theta0, 6),
        "imputed_studies": imputed,
        "interpretation": (
            f"{k0} studies imputed. Adjusted effect: {adjusted.pooled_effect:.3f} "
            f"[{adjusted.ci_lower:.3f}, {adjusted.ci_upper:.3f}] "
            f"(original: {theta0:.3f})."
        ),
    }


def fail_safe_n(studies: list[StudyEffect], alpha: float = 0.05) -> dict:
    """Rosenthal's fail-safe N.

    Number of studies with null results needed to bring the pooled p-value
    above alpha.

    N_fs = (sum(z_i) / z_alpha)^2 - k
    """
    k = len(studies)
    if k == 0:
        return {
            "failsafe_n": 0,
            "interpretation": "No studies available.",
        }

    z_alpha = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))

    # Compute per-study z-values
    z_values = []
    for s in studies:
        if s.standard_error > 0:
            z_values.append(s.effect_size / s.standard_error)
        else:
            z_values.append(0.0)

    sum_z = sum(z_values)
    if z_alpha <= 0:
        return {"failsafe_n": 0, "interpretation": "Invalid alpha."}

    nfs = max(0, int((sum_z / z_alpha) ** 2 - k))

    # Interpretation thresholds (Rosenthal's 5k + 10 rule)
    threshold = 5 * k + 10
    if nfs > threshold:
        interpretation = (
            f"Fail-safe N = {nfs} (threshold: {threshold}). "
            f"Result is robust to unpublished null studies."
        )
    else:
        interpretation = (
            f"Fail-safe N = {nfs} (threshold: {threshold}). "
            f"Result may be vulnerable to publication bias."
        )

    return {
        "failsafe_n": nfs,
        "threshold": threshold,
        "interpretation": interpretation,
    }


def run_bias_tests(studies: list[StudyEffect]) -> BiasTestResult:
    """Run all three bias tests and return combined result."""
    k = len(studies)
    result = BiasTestResult(n_studies=k)

    if k < 3:
        result.overall_interpretation = (
            "Too few studies to assess publication bias."
        )
        return result

    # Egger's test
    egger = eggers_test(studies)
    result.eggers_intercept = egger["intercept"]
    result.eggers_se = egger["se"]
    result.eggers_t = egger["t_value"]
    result.eggers_p = egger["p_value"]
    result.eggers_interpretation = egger["interpretation"]

    # Trim-and-fill
    tf = trim_and_fill(studies)
    result.trimfill_n_imputed = tf["n_imputed"]
    result.trimfill_adjusted_effect = tf["adjusted_effect"]
    result.trimfill_adjusted_ci_lower = tf["adjusted_ci_lower"]
    result.trimfill_adjusted_ci_upper = tf["adjusted_ci_upper"]
    result.trimfill_original_effect = tf["original_effect"]

    # Fail-safe N
    fsn = fail_safe_n(studies)
    result.failsafe_n = fsn["failsafe_n"]
    result.failsafe_interpretation = fsn["interpretation"]

    # Overall interpretation
    concerns = []
    if result.eggers_p < 0.05:
        concerns.append("Egger's test significant (publication bias likely)")
    elif result.eggers_p < 0.10:
        concerns.append("Egger's test borderline")

    if result.trimfill_n_imputed > 0:
        concerns.append(f"{result.trimfill_n_imputed} studies imputed by trim-and-fill")

    threshold = 5 * k + 10
    if result.failsafe_n < threshold:
        concerns.append("Fail-safe N below Rosenthal threshold")

    if concerns:
        result.overall_interpretation = (
            "Publication bias concerns: " + "; ".join(concerns) + "."
        )
    else:
        result.overall_interpretation = (
            "No strong evidence of publication bias detected."
        )

    return result


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def create_forest_plot_figure(
    pooled: PooledEstimate,
    title: str = "Forest Plot",
) -> "Figure":
    """Create a matplotlib forest plot figure.

    Layout:
    - Left column: study labels
    - Center: horizontal lines (CI) with squares (effect size, sized by weight)
    - Bottom: diamond for pooled estimate
    - Right column: effect size [CI] text
    - Bottom annotation: heterogeneity stats (I^2, Q, tau^2, p)

    Returns a matplotlib Figure (not shown, for embedding in GUI).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    studies = pooled.studies
    k = len(studies)
    if k == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No studies to display", ha="center", va="center",
                fontsize=11, color="#808080")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # Figure sizing
    row_height = 0.45
    fig_height = max(4.0, (k + 3) * row_height + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Y positions: studies top-down, pooled at bottom
    y_positions = list(range(k - 1, -1, -1))  # top to bottom
    y_pooled = -1.5

    # Determine x-axis range
    all_ci_lo = [s.ci_lower for s in studies]
    all_ci_hi = [s.ci_upper for s in studies]
    x_min = min(min(all_ci_lo), pooled.ci_lower, 0) - 0.3
    x_max = max(max(all_ci_hi), pooled.ci_upper, 0) + 0.3

    # Plot each study
    max_weight = max(s.weight for s in studies) if studies else 1.0
    for i, (s, y) in enumerate(zip(studies, y_positions)):
        # CI line
        ax.plot([s.ci_lower, s.ci_upper], [y, y], color="#2c3e50",
                linewidth=1.0, solid_capstyle="butt", zorder=2)

        # Square sized by weight
        sq_size = max(4.0, 12.0 * (s.weight / max_weight if max_weight > 0 else 0.5))
        ax.plot(s.effect_size, y, marker="s", color="#2980b9",
                markersize=sq_size, zorder=3)

        # Study label (left)
        label = s.title[:40] + "..." if len(s.title) > 40 else s.title
        if not label:
            label = s.article_id[:30]
        ax.text(x_min - 0.05, y, label, ha="right", va="center",
                fontsize=8, fontfamily="sans-serif")

        # Effect + CI text (right)
        ci_text = f"{s.effect_size:.2f} [{s.ci_lower:.2f}, {s.ci_upper:.2f}]"
        ax.text(x_max + 0.05, y, ci_text, ha="left", va="center",
                fontsize=8, fontfamily="monospace")

    # Pooled estimate diamond
    diamond_half_h = 0.35
    diamond = Polygon([
        (pooled.ci_lower, y_pooled),
        (pooled.pooled_effect, y_pooled + diamond_half_h),
        (pooled.ci_upper, y_pooled),
        (pooled.pooled_effect, y_pooled - diamond_half_h),
    ], closed=True, facecolor="#e74c3c", edgecolor="#c0392b", linewidth=1.2, zorder=3)
    ax.add_patch(diamond)

    # Pooled label
    ax.text(x_min - 0.05, y_pooled, "Pooled RE", ha="right", va="center",
            fontsize=9, fontweight="bold", fontfamily="sans-serif")
    ci_text = f"{pooled.pooled_effect:.2f} [{pooled.ci_lower:.2f}, {pooled.ci_upper:.2f}]"
    ax.text(x_max + 0.05, y_pooled, ci_text, ha="left", va="center",
            fontsize=8, fontweight="bold", fontfamily="monospace")

    # Vertical null-effect line
    ax.axvline(x=0, color="#7f8c8d", linestyle="--", linewidth=0.8, zorder=1)

    # Dashed vertical line at pooled effect
    ax.axvline(x=pooled.pooled_effect, color="#e74c3c", linestyle=":",
               linewidth=0.6, alpha=0.5, zorder=1)

    # Separator line between studies and pooled
    sep_y = -0.75
    ax.axhline(y=sep_y, color="#bdc3c7", linewidth=0.5)

    # Heterogeneity annotation
    het_text = (
        f"Heterogeneity: I\u00b2 = {pooled.i_squared:.1f}%, "
        f"\u03c4\u00b2 = {pooled.tau_squared:.4f}, "
        f"Q = {pooled.q_statistic:.2f} (p = {pooled.q_p_value:.3f})"
    )
    test_text = (
        f"Test for overall effect: Z = {pooled.z_value:.2f} "
        f"(p = {pooled.p_value:.4f})"
    )
    ax.text(0.5, -0.06, het_text, ha="center", va="top",
            fontsize=8, color="#555555", transform=ax.transAxes)
    ax.text(0.5, -0.09, test_text, ha="center", va="top",
            fontsize=8, color="#555555", transform=ax.transAxes)

    # Axis formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_pooled - 1.0, k - 0.5)
    ax.set_xlabel("Effect Size (Hedges' g)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.subplots_adjust(left=0.30, right=0.70, bottom=0.15, top=0.93)
    return fig


def create_funnel_plot_figure(
    studies: list[StudyEffect],
    pooled_effect: float,
    bias_result: Optional[BiasTestResult] = None,
    title: str = "Funnel Plot",
) -> "Figure":
    """Create a contour-enhanced funnel plot.

    Layout:
    - X-axis: effect size
    - Y-axis: standard error (inverted - larger studies on top)
    - Vertical line at pooled estimate
    - Funnel outline (95% pseudo-CI)
    - Contour regions for significance (p < .05, .01, .001) in light gray shades
    - Points for each study (filled circles)
    - If bias_result has trim-and-fill data: show imputed studies as open circles
    - Bottom annotation: Egger's test result, trim-and-fill adjustment

    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not studies:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "No studies to display", ha="center", va="center",
                fontsize=11, color="#808080")
        ax.axis("off")
        fig.tight_layout()
        return fig

    effects = [s.effect_size for s in studies]
    ses = [s.standard_error for s in studies]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Determine axis ranges
    se_max = max(ses) * 1.15
    se_range = np.linspace(0.001, se_max, 200)

    # Contour regions: significance levels
    for z_crit, alpha_label, shade in [
        (sp_stats.norm.ppf(0.9995), "p < 0.001", "#f0f0f0"),
        (sp_stats.norm.ppf(0.995), "p < 0.01", "#e0e0e0"),
        (sp_stats.norm.ppf(0.975), "p < 0.05", "#d0d0d0"),
    ]:
        left = pooled_effect - z_crit * se_range
        right = pooled_effect + z_crit * se_range
        ax.fill_betweenx(se_range, left, right, alpha=0.4, color=shade,
                         label=alpha_label, zorder=0)

    # 95% pseudo-CI funnel outline
    z95 = 1.96
    funnel_lo = pooled_effect - z95 * se_range
    funnel_hi = pooled_effect + z95 * se_range
    ax.plot(funnel_lo, se_range, "--", color="#7f8c8d", linewidth=0.8, zorder=1)
    ax.plot(funnel_hi, se_range, "--", color="#7f8c8d", linewidth=0.8, zorder=1)

    # Vertical line at pooled effect
    ax.axvline(x=pooled_effect, color="#2c3e50", linestyle="-", linewidth=1.0,
               zorder=1, label=f"Pooled = {pooled_effect:.3f}")

    # Plot observed studies
    ax.scatter(effects, ses, s=40, c="#2980b9", edgecolors="#1a5276",
               linewidths=0.5, zorder=4, label=f"Studies (k={len(studies)})")

    # Plot imputed studies from trim-and-fill
    if bias_result and bias_result.trimfill_n_imputed > 0:
        # Reconstruct imputed positions
        tf_data = trim_and_fill(studies)
        imputed_studies = tf_data.get("imputed_studies", [])
        if imputed_studies:
            imp_effects = [s.effect_size for s in imputed_studies]
            imp_ses = [s.standard_error for s in imputed_studies]
            ax.scatter(imp_effects, imp_ses, s=40, facecolors="none",
                       edgecolors="#e74c3c", linewidths=1.2, zorder=4,
                       label=f"Imputed (k={len(imputed_studies)})")

    # Egger's regression line
    if bias_result and bias_result.eggers_p < 1.0 and len(studies) >= 3:
        # Regression line: standardized_effect = intercept + slope * precision
        # On the funnel plot: effect = intercept*SE + slope
        # This is approximate — draw from se=0 to se=se_max
        # Reconstruct slope from OLS
        precision = np.array([1.0 / s.standard_error for s in studies if s.standard_error > 0])
        std_effect = np.array([s.effect_size / s.standard_error for s in studies if s.standard_error > 0])
        if len(precision) >= 3:
            X = np.column_stack([np.ones(len(precision)), precision])
            beta = np.linalg.lstsq(X, std_effect, rcond=None)[0]
            # On funnel plot coords: effect = beta[0]*SE + beta[1]
            line_effects = beta[0] * se_range + beta[1]
            ax.plot(line_effects, se_range, "-", color="#e74c3c", linewidth=1.0,
                    alpha=0.7, zorder=2, label="Egger's regression")

    # Annotations
    annotations = []
    if bias_result:
        annotations.append(
            f"Egger's: intercept = {bias_result.eggers_intercept:.3f}, "
            f"p = {bias_result.eggers_p:.3f}"
        )
        if bias_result.trimfill_n_imputed > 0:
            annotations.append(
                f"Trim-and-fill: {bias_result.trimfill_n_imputed} imputed, "
                f"adj. effect = {bias_result.trimfill_adjusted_effect:.3f}"
            )
        annotations.append(
            f"Fail-safe N = {bias_result.failsafe_n}"
        )

    if annotations:
        ann_text = " | ".join(annotations)
        ax.text(0.5, -0.10, ann_text, ha="center", va="top",
                fontsize=7.5, color="#555555", transform=ax.transAxes)

    # Axis formatting
    ax.set_xlabel("Effect Size (Hedges' g)", fontsize=10)
    ax.set_ylabel("Standard Error", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()  # Larger studies (smaller SE) at top
    ax.set_ylim(se_max, 0)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def run_full_analysis(
    config,
    *,
    input_source: str = "stage2c",
    metrics_path: Optional[str | Path] = None,
) -> dict:
    """Run the complete meta-analytic workflow.

    1. Load effect sizes from the explicitly selected input
    2. Pool overall random-effects estimate
    3. Pool by condition and component
    4. Run publication bias tests (if >= 3 studies)
    5. Generate forest + funnel plot figures
    6. Save results JSON

    Returns dict with all results, figures, and file paths.
    """
    resolved_metrics_path, resolved_source = resolve_meta_analysis_input(
        config,
        input_source=input_source,
        metrics_path=metrics_path,
    )
    results: dict = {
        "n_studies": 0,
        "pooled": None,
        "bias": None,
        "forest_fig": None,
        "funnel_fig": None,
        "by_condition": {},
        "by_component": {},
        "skipped": False,
        "reason": "",
        "input_source": resolved_source,
        "input_path": str(resolved_metrics_path),
    }

    # 1. Load effect sizes
    studies = load_effect_sizes(
        config,
        metrics_path=resolved_metrics_path,
        input_source=resolved_source,
    )
    if not studies:
        results["skipped"] = True
        results["reason"] = (
            "No effect sizes available. Run Stage 2c Statistical Extraction "
            f"or choose another meta-analysis input. Checked: {resolved_metrics_path}"
        )
        logger.warning("Meta-analysis skipped: %s", results["reason"])
        return results

    results["n_studies"] = len(studies)

    # 2. Pool overall
    pooled = pool_random_effects(studies)
    results["pooled"] = pooled

    # 3. Pool by condition and component
    results["by_condition"] = pool_by_condition(studies)
    results["by_component"] = pool_by_component(studies)

    # 3b. FDR correction across sub-group p-values
    condition_pvals = {
        k: v.p_value for k, v in results["by_condition"].items()
    }
    component_pvals = {
        k: v.p_value for k, v in results["by_component"].items()
    }
    results["fdr_by_condition"] = benjamini_hochberg(condition_pvals)
    results["fdr_by_component"] = benjamini_hochberg(component_pvals)

    # 4. Publication bias tests
    bias = run_bias_tests(studies)
    results["bias"] = bias

    # 5. Generate figures
    results["forest_fig"] = create_forest_plot_figure(pooled)
    results["funnel_fig"] = create_funnel_plot_figure(
        studies, pooled.pooled_effect, bias
    )

    # 6. Save results JSON
    try:
        save_path = config.output_dir / "meta_analysis_results.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "input_source": resolved_source,
            "input_path": str(resolved_metrics_path),
            "n_studies": len(studies),
            "pooled_effect": pooled.pooled_effect,
            "pooled_se": pooled.pooled_se,
            "ci_lower": pooled.ci_lower,
            "ci_upper": pooled.ci_upper,
            "z_value": pooled.z_value,
            "p_value": pooled.p_value,
            "tau_squared": pooled.tau_squared,
            "i_squared": pooled.i_squared,
            "q_statistic": pooled.q_statistic,
            "q_p_value": pooled.q_p_value,
            "prediction_interval": list(pooled.prediction_interval),
            "eggers_intercept": bias.eggers_intercept,
            "eggers_p": bias.eggers_p,
            "trimfill_n_imputed": bias.trimfill_n_imputed,
            "trimfill_adjusted_effect": bias.trimfill_adjusted_effect,
            "failsafe_n": bias.failsafe_n,
            "overall_bias_interpretation": bias.overall_interpretation,
            "fdr_by_condition": results["fdr_by_condition"],
            "fdr_by_component": results["fdr_by_component"],
            "studies": [
                {
                    "article_id": s.article_id,
                    "title": s.title,
                    "effect_size": s.effect_size,
                    "standard_error": s.standard_error,
                    "sample_size": s.sample_size,
                    "ci_lower": s.ci_lower,
                    "ci_upper": s.ci_upper,
                    "weight": s.weight,
                }
                for s in studies
            ],
        }
        save_path.write_text(json.dumps(save_data, indent=2, default=str))
        results["save_path"] = str(save_path)
        logger.info("Meta-analysis results saved to %s", save_path)
    except Exception as exc:
        logger.warning("Could not save meta-analysis results: %s", exc)

    return results
