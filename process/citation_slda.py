from __future__ import annotations

import math
import re
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _import_tomotopy():
    try:
        import tomotopy as tp
    except Exception as exc:  # pragma: no cover - depends on local notebook env
        raise ImportError(
            "`tomotopy` is required for citation-supervised LDA. "
            "Install it in the notebook kernel that will run this analysis."
        ) from exc
    return tp


def tokenize_topic_text(
    text: str,
    *,
    min_token_len: int = 3,
    max_tokens: int | None = None,
    extra_stop_words: set[str] | None = None,
) -> list[str]:
    stop_words = set(ENGLISH_STOP_WORDS)
    if extra_stop_words:
        stop_words.update(word.lower() for word in extra_stop_words)

    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(str(text or "").lower()):
        token = raw_token.strip("_-")
        if len(token) < min_token_len or token in stop_words:
            continue
        tokens.append(token)
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return tokens



def _coerce_estimate(value: Any) -> float:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return math.nan
        value = value[0]
    return float(value)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    return values


def _regression_metrics(
    actual_target: np.ndarray,
    predicted_target: np.ndarray,
    *,
    transform: str,
    prefix: str,
) -> dict[str, float]:
    actual_target = np.asarray(actual_target, dtype=float)
    predicted_target = np.asarray(predicted_target, dtype=float)
    actual_citations = _inverse_target(actual_target, transform)
    predicted_citations = _inverse_target(predicted_target, transform)

    metrics = {
        f"{prefix}_mae_target": float(mean_absolute_error(actual_target, predicted_target)),
        f"{prefix}_rmse_target": float(
            root_mean_squared_error(actual_target, predicted_target)
        ),
        f"{prefix}_r2_target": float(r2_score(actual_target, predicted_target))
        if len(actual_target) >= 2
        else math.nan,
        f"{prefix}_corr_target": _safe_corr(actual_target, predicted_target),
        f"{prefix}_mae_citations": float(
            mean_absolute_error(actual_citations, predicted_citations)
        ),
        f"{prefix}_rmse_citations": float(
            root_mean_squared_error(actual_citations, predicted_citations)
        ),
        f"{prefix}_corr_citations": _safe_corr(actual_citations, predicted_citations),
    }
    return metrics


def _term_weight_from_name(tp, name: str):
    lookup = {
        "one": tp.TermWeight.ONE,
        "idf": tp.TermWeight.IDF,
        "pmi": tp.TermWeight.PMI,
    }
    normalized = str(name or "idf").strip().lower()
    if normalized not in lookup:
        raise ValueError(f"Unsupported term weight: {name}")
    return lookup[normalized]


def _build_prediction_frame(
    base_frame: pd.DataFrame,
    *,
    split: str,
    topic_matrix: np.ndarray,
    predicted_target: np.ndarray,
    transform: str,
) -> pd.DataFrame:
    result = base_frame.copy().reset_index(drop=True)
    result["split"] = split
    result["predicted_citation_target"] = predicted_target
    result["predicted_citation_count"] = _inverse_target(predicted_target, transform)
    result["residual_target"] = result["citation_target"] - result["predicted_citation_target"]
    result["residual_citation_count"] = (
        result["citation_count"] - result["predicted_citation_count"]
    )
    result["dominant_topic"] = np.argmax(topic_matrix, axis=1).astype(int)

    for topic_id in range(topic_matrix.shape[1]):
        result[f"topic_{topic_id:02d}"] = topic_matrix[:, topic_id]

    return result


def run_citation_slda(
    prepared_docs: pd.DataFrame,
    *,
    k: int = 15,
    term_weight: str = "idf",
    min_cf: int = 10,
    min_df: int = 5,
    rm_top: int = 20,
    alpha: float = 0.1,
    eta: float = 0.01,
    iterations: int = 400,
    burn_in: int = 100,
    infer_iter: int = 120,
    train_fraction: float = 0.8,
    workers: int = 0,
    seed: int = 42,
    optim_interval: int = 20,
    progress_every: int = 25,
    topic_top_n: int = 12,
) -> dict[str, Any]:
    tp = _import_tomotopy()

    if "tokens" not in prepared_docs.columns or "citation_target" not in prepared_docs.columns:
        raise KeyError("prepared_docs must include `tokens` and `citation_target` columns.")
    if len(prepared_docs) < 10:
        raise ValueError("At least 10 prepared documents are required for citation sLDA.")
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in the range [0.5, 1.0).")

    frame = prepared_docs.reset_index(drop=True).copy()
    transform = (
        "log1p"
        if np.allclose(np.log1p(frame["citation_count"]), frame["citation_target"])
        else "identity"
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)

    train_size = max(8, int(round(len(frame) * train_fraction)))
    train_size = min(train_size, len(frame) - 1)
    train_idx = np.sort(indices[:train_size])
    test_idx = np.sort(indices[train_size:])

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)

    model = tp.SLDAModel(
        k=k,
        vars="l",
        # tomotopy's own docs say `tw: Union[int, tomotopy.TermWeight]`, but its
        # unannotated `tw='one'` default narrows the inferred type to `str`.
        tw=cast(str, _term_weight_from_name(tp, term_weight)),
        min_cf=min_cf,
        min_df=min_df,
        rm_top=rm_top,
        alpha=alpha,
        eta=eta,
        seed=seed,
    )
    model.burn_in = burn_in
    model.optim_interval = optim_interval

    for row in train_frame.itertuples(index=False):
        model.add_doc(row.tokens, y=[_coerce_estimate(row.citation_target)])

    history: list[dict[str, float]] = []
    trained = 0
    while trained < iterations:
        step = min(progress_every, iterations - trained)
        model.train(step, workers=workers)
        trained += step
        history.append(
            {
                "iteration": trained,
                "ll_per_word": float(model.ll_per_word),
                "perplexity": float(model.perplexity),
            }
        )

    train_topic_matrix = np.asarray([doc.get_topic_dist() for doc in model.docs], dtype=float)
    train_predicted_target = np.asarray(
        [_coerce_estimate(model.estimate(doc)) for doc in model.docs],
        dtype=float,
    )

    test_topic_rows: list[np.ndarray] = []
    test_predicted_rows: list[float] = []
    for row in test_frame.itertuples(index=False):
        doc = model.make_doc(row.tokens)
        topic_dist, _ = model.infer(doc, iterations=infer_iter, workers=workers)
        test_topic_rows.append(np.asarray(topic_dist, dtype=float))
        test_predicted_rows.append(_coerce_estimate(model.estimate(doc)))

    test_topic_matrix = np.vstack(test_topic_rows)
    test_predicted_target = np.asarray(test_predicted_rows, dtype=float)

    all_predictions = pd.concat(
        [
            _build_prediction_frame(
                train_frame,
                split="train",
                topic_matrix=train_topic_matrix,
                predicted_target=train_predicted_target,
                transform=transform,
            ),
            _build_prediction_frame(
                test_frame,
                split="test",
                topic_matrix=test_topic_matrix,
                predicted_target=test_predicted_target,
                transform=transform,
            ),
        ],
        ignore_index=True,
    )

    regression_coef = np.asarray(model.get_regression_coef(0), dtype=float)
    topic_summary_rows = []
    full_topic_matrix = all_predictions.filter(regex=r"^topic_\d+$").to_numpy(dtype=float)
    for topic_id in range(k):
        # Not passing `return_id` means tomotopy always returns 2-tuples at
        # runtime, but its stub types the result as a Union over both shapes
        # since it isn't keyed to the `return_id` default via an overload.
        topic_words = cast(
            "list[tuple[str, float]]",
            model.get_topic_words(topic_id, top_n=topic_top_n),
        )
        top_words = ", ".join(word for word, _prob in topic_words)
        topic_share = full_topic_matrix[:, topic_id]
        topic_summary_rows.append(
            {
                "topic": topic_id,
                "regression_coef": float(regression_coef[topic_id]),
                "mean_topic_share": float(topic_share.mean()),
                "topic_share_vs_target_corr": _safe_corr(
                    topic_share,
                    all_predictions["citation_target"].to_numpy(dtype=float),
                ),
                "top_words": top_words,
            }
        )

    topic_summary = (
        pd.DataFrame(topic_summary_rows)
        .sort_values(["regression_coef", "mean_topic_share"], ascending=[False, False])
        .reset_index(drop=True)
    )

    metrics = {
        "k": int(k),
        "train_docs": int(len(train_frame)),
        "test_docs": int(len(test_frame)),
        "prepared_docs": int(len(frame)),
        "mean_token_count": round(float(frame["token_count"].mean()), 2),
        "median_token_count": round(float(frame["token_count"].median()), 2),
        "num_vocabs": int(model.num_vocabs),
        "num_words": int(model.num_words),
        "ll_per_word": float(model.ll_per_word),
        "perplexity": float(model.perplexity),
        "target_transform": transform,
        "iterations": int(iterations),
        "burn_in": int(burn_in),
        "infer_iter": int(infer_iter),
    }
    metrics.update(
        _regression_metrics(
            train_frame["citation_target"].to_numpy(dtype=float),
            train_predicted_target,
            transform=transform,
            prefix="train",
        )
    )
    metrics.update(
        _regression_metrics(
            test_frame["citation_target"].to_numpy(dtype=float),
            test_predicted_target,
            transform=transform,
            prefix="test",
        )
    )

    return {
        "model": model,
        "documents": all_predictions,
        "topic_summary": topic_summary,
        "train_curve": pd.DataFrame(history),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Enhanced regression: sLDA topics + metadata features
# ---------------------------------------------------------------------------

# Metadata features extracted from paper dicts during acquisition.
# Each tuple: (column_name, type, default_value_for_missing)
METADATA_FEATURES: list[tuple[str, str, Any]] = [
    ("year", "numeric", 0),
    ("n_authors", "numeric", 0),
    ("openalex_is_oa", "binary", 0),
    # NOTE: citation_count is intentionally omitted; it is the prediction target
    # and including it would leak the target into the regression features.
]


def _extract_metadata_features(
    docs_df: pd.DataFrame,
    feature_specs: list[tuple[str, str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract and clean metadata features from a documents DataFrame.

    Returns (feature_matrix_df, feature_names).
    Missing values are filled with the specified default.
    Binary features are coerced to 0/1.
    Numeric features are z-score normalized.
    """
    specs = feature_specs or METADATA_FEATURES
    features = pd.DataFrame(index=docs_df.index)
    names = []

    for col_name, feat_type, default in specs:
        if col_name not in docs_df.columns:
            continue
        series = docs_df[col_name].copy()

        if feat_type == "binary":
            series = series.fillna(default).astype(float)
            series = (series > 0).astype(float)
        else:  # numeric
            series = pd.to_numeric(series, errors="coerce").fillna(default).astype(float)
            # Z-score normalize to put on same scale as topic proportions
            std = series.std()
            if std > 1e-10:
                series = (series - series.mean()) / std

        features[col_name] = series
        names.append(col_name)

    return features, names


def enhanced_citation_regression(
    slda_result: dict[str, Any],
    source_docs: pd.DataFrame | None = None,
    extra_features: list[tuple[str, str, Any]] | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Combine sLDA topic features with metadata for richer citation prediction.

    Fits a Ridge regression on topic proportions + metadata features and
    reports cross-validated R², feature importances, and improvement over
    topics-only baseline.

    Parameters
    ----------
    slda_result : output from run_citation_slda()
    source_docs : original DataFrame with metadata columns (year, n_authors,
        openalex_is_oa, etc.). If None, uses slda_result["documents"].
    extra_features : additional (col_name, type, default) tuples beyond
        the built-in METADATA_FEATURES list
    cv_folds : number of cross-validation folds
    seed : random seed

    Returns
    -------
    dict with: r2_topics_only, r2_combined, r2_improvement,
        feature_importances, cv_scores, n_features
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    predictions_df = slda_result["documents"]
    topic_cols = [c for c in predictions_df.columns if c.startswith("topic_")]

    if not topic_cols:
        return {"error": "No topic columns found in sLDA output"}

    target = predictions_df["citation_target"].to_numpy(dtype=float)
    X_topics = predictions_df[topic_cols].to_numpy(dtype=float)

    # Topics-only baseline
    model_topics = Ridge(alpha=1.0)
    cv_topics = cross_val_score(
        model_topics, X_topics, target, cv=min(cv_folds, len(target) // 2),
        scoring="r2",
    )
    r2_topics = float(np.mean(cv_topics))

    # Extract metadata features
    docs_for_meta = source_docs if source_docs is not None else predictions_df
    feature_specs = list(METADATA_FEATURES)
    if extra_features:
        feature_specs.extend(extra_features)

    meta_features, meta_names = _extract_metadata_features(docs_for_meta, feature_specs)

    if meta_features.empty or not meta_names:
        return {
            "r2_topics_only": round(r2_topics, 6),
            "r2_combined": round(r2_topics, 6),
            "r2_improvement": 0.0,
            "feature_importances": {c: 0.0 for c in topic_cols},
            "cv_scores_topics": [round(float(s), 6) for s in cv_topics],
            "cv_scores_combined": [round(float(s), 6) for s in cv_topics],
            "n_topic_features": len(topic_cols),
            "n_metadata_features": 0,
            "metadata_features_used": [],
        }

    # Align indices
    meta_aligned = meta_features.reindex(predictions_df.index).fillna(0)
    X_combined = np.hstack([X_topics, meta_aligned.to_numpy(dtype=float)])
    all_feature_names = topic_cols + meta_names

    # Combined model
    model_combined = Ridge(alpha=1.0)
    cv_combined = cross_val_score(
        model_combined, X_combined, target,
        cv=min(cv_folds, len(target) // 2),
        scoring="r2",
    )
    r2_combined = float(np.mean(cv_combined))

    # Fit final model for feature importances (coefficients)
    model_combined.fit(X_combined, target)
    coefs = model_combined.coef_
    importances = {
        name: round(float(abs(c)), 6)
        for name, c in zip(all_feature_names, coefs)
    }

    return {
        "r2_topics_only": round(r2_topics, 6),
        "r2_combined": round(r2_combined, 6),
        "r2_improvement": round(r2_combined - r2_topics, 6),
        "feature_importances": importances,
        "cv_scores_topics": [round(float(s), 6) for s in cv_topics],
        "cv_scores_combined": [round(float(s), 6) for s in cv_combined],
        "n_topic_features": len(topic_cols),
        "n_metadata_features": len(meta_names),
        "metadata_features_used": meta_names,
    }

