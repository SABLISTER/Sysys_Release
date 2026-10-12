"""Wordfish text scaling and lightweight stylometric profiling."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+"
    r"(?:\w+\s+){0,2}\w+(?:ed|en)\b",
    re.IGNORECASE,
)

_HEDGE_WORDS = {
    "apparently", "approximately", "arguably", "could", "generally",
    "likely", "may", "might", "perhaps", "possibly", "potentially",
    "presumably", "seem", "seemed", "seems", "suggest", "suggested",
    "suggesting", "suggests", "typically", "unlikely",
}
_CERTAINTY_WORDS = {
    "certainly", "clearly", "confirm", "confirmed", "confirms",
    "conclusive", "definite", "definitely", "demonstrate", "demonstrated",
    "demonstrates", "establish", "established", "establishes", "evident",
    "indeed", "must", "show", "showed", "showing", "shown", "shows",
    "strongly", "undoubtedly",
}


class TextScalingCancelled(RuntimeError):
    """Raised when a text-scaling run is cancelled."""


def _field(article: Any, name: str, default: Any = "") -> Any:
    if isinstance(article, Mapping):
        return article.get(name, default)
    return getattr(article, name, default)


def _article_text(article: Any) -> str:
    title = str(_field(article, "title", "") or "").strip()
    body = str(
        _field(article, "full_text", "")
        or _field(article, "fulltext", "")
        or _field(article, "abstract", "")
        or ""
    ).strip()
    if title and body:
        return f"{title}. {body}"
    return body or title


def _syllable_count(word: str) -> int:
    normalized = re.sub(r"[^a-z]", "", word.lower())
    if not normalized:
        return 0
    count = len(_VOWEL_GROUP_RE.findall(normalized))
    if normalized.endswith("e") and not normalized.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def build_stylometric_profile(article_id: str, article: Any, text: str | None = None) -> dict:
    """Compute interpretable writing-style measures for one article."""
    text = text if text is not None else _article_text(article)
    words = [word.lower() for word in _WORD_RE.findall(text)]
    word_count = len(words)
    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    sentence_count = max(1, len(sentences))
    syllables = sum(_syllable_count(word) for word in words)

    hedge_count = sum(word in _HEDGE_WORDS for word in words)
    certainty_count = sum(word in _CERTAINTY_WORDS for word in words)
    passive_count = len(_PASSIVE_RE.findall(text))
    per_100_words = 100.0 / word_count if word_count else 0.0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
    syllables_per_word = syllables / word_count if word_count else 0.0
    fk_grade = 0.39 * avg_sentence_length + 11.8 * syllables_per_word - 15.59

    return {
        "article_id": str(article_id),
        "title": str(_field(article, "title", "") or article_id),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": round(avg_sentence_length, 4),
        "hedge_count": hedge_count,
        "hedge_density": round(hedge_count * per_100_words, 4),
        "certainty_count": certainty_count,
        "certainty_density": round(certainty_count * per_100_words, 4),
        "flesch_kincaid_grade": round(max(0.0, fk_grade), 4),
        "type_token_ratio": round(len(set(words)) / word_count, 4) if word_count else 0.0,
        "passive_voice_count": passive_count,
        "passive_voice_density": round(100.0 * passive_count / sentence_count, 4),
    }


def _check_cancel(cancel_cb: Callable[[], bool] | None) -> None:
    if cancel_cb and cancel_cb():
        raise TextScalingCancelled("Text scaling cancelled")


def _fit_wordfish(
    counts,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    max_iterations: int = 50,
    tolerance: float = 5e-5,
) -> dict:
    """Fit the one-dimensional Poisson Wordfish model with damped Newton steps."""
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    counts = counts.astype(np.float64).tocsr()
    n_documents, n_features = counts.shape
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    column_totals = np.asarray(counts.sum(axis=0)).ravel()
    grand_total = float(row_totals.sum())
    epsilon = 1e-12

    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    theta = TruncatedSVD(n_components=1, random_state=0).fit_transform(tfidf).ravel()
    if float(np.std(theta)) < 1e-8:
        theta = np.linspace(-1.0, 1.0, n_documents)
    theta = (theta - float(np.mean(theta))) / max(float(np.std(theta)), 1e-8)
    initial_theta = theta.copy()

    alpha = np.log(np.maximum(row_totals, epsilon))
    psi = np.log(np.maximum(column_totals / grand_total, epsilon))
    beta = np.asarray(counts.T.dot(theta)).ravel() / np.maximum(column_totals, 1.0)
    beta -= float(np.average(beta, weights=np.maximum(column_totals, 1.0)))
    beta_scale = max(float(np.std(beta)), 1e-8)
    beta = np.clip(beta / beta_scale * 0.1, -0.5, 0.5)

    converged = False
    max_delta = math.inf
    log_likelihood = -math.inf
    relative_log_likelihood_change = math.inf
    previous_log_likelihood: float | None = None
    completed_iterations = 0
    chunk_size = 256
    counts_coo = counts.tocoo()

    for iteration in range(1, max_iterations + 1):
        _check_cancel(cancel_cb)
        previous_theta = theta.copy()
        y_beta = np.asarray(counts.dot(beta)).ravel()
        y_theta = np.asarray(counts.T.dot(theta)).ravel()

        predicted_rows = np.zeros(n_documents)
        predicted_columns = np.zeros(n_features)
        theta_mu_beta = np.zeros(n_documents)
        theta_information = np.zeros(n_documents)
        beta_mu_theta = np.zeros(n_features)
        beta_information = np.zeros(n_features)
        theta_squared = theta * theta

        for start in range(0, n_features, chunk_size):
            _check_cancel(cancel_cb)
            stop = min(start + chunk_size, n_features)
            beta_block = beta[start:stop]
            eta = (
                alpha[:, None]
                + psi[start:stop][None, :]
                + theta[:, None] * beta_block[None, :]
            )
            mu = np.exp(np.clip(eta, -30.0, 30.0))
            predicted_rows += np.sum(mu, axis=1)
            predicted_columns[start:stop] = np.sum(mu, axis=0)
            theta_mu_beta += mu.dot(beta_block)
            theta_information += mu.dot(beta_block * beta_block)
            beta_mu_theta[start:stop] = theta.dot(mu)
            beta_information[start:stop] = theta_squared.dot(mu)

        log_likelihood = float(
            np.dot(
                counts_coo.data,
                alpha[counts_coo.row]
                + psi[counts_coo.col]
                + theta[counts_coo.row] * beta[counts_coo.col],
            )
            - np.sum(predicted_rows)
        )
        if previous_log_likelihood is not None:
            relative_log_likelihood_change = abs(
                (log_likelihood - previous_log_likelihood)
                / max(abs(previous_log_likelihood), 1.0)
            )
            if iteration >= 3 and relative_log_likelihood_change < tolerance:
                converged = True
                completed_iterations = iteration
                if progress_cb:
                    progress_cb(iteration, max_iterations)
                break
        previous_log_likelihood = log_likelihood

        damping = 0.5
        alpha += damping * np.clip(
            np.log(np.maximum(row_totals, epsilon) / np.maximum(predicted_rows, epsilon)),
            -1.0,
            1.0,
        )
        psi += damping * np.clip(
            np.log(
                np.maximum(column_totals, epsilon)
                / np.maximum(predicted_columns, epsilon)
            ),
            -1.0,
            1.0,
        )
        theta += damping * np.clip(
            (y_beta - theta_mu_beta) / np.maximum(theta_information, epsilon),
            -0.5,
            0.5,
        )
        beta += damping * np.clip(
            (y_theta - beta_mu_theta) / np.maximum(beta_information, epsilon),
            -0.5,
            0.5,
        )
        beta = np.clip(beta, -5.0, 5.0)

        # Enforce Wordfish identification constraints while preserving eta.
        theta_mean = float(np.mean(theta))
        theta_std = max(float(np.std(theta)), 1e-8)
        psi += theta_mean * beta
        beta *= theta_std
        theta = (theta - theta_mean) / theta_std
        alpha_shift = float(np.mean(alpha))
        alpha -= alpha_shift
        psi += alpha_shift

        if float(np.dot(theta, initial_theta)) < 0.0:
            theta *= -1.0
            beta *= -1.0

        max_delta = float(np.max(np.abs(theta - previous_theta)))
        completed_iterations = iteration
        if progress_cb:
            progress_cb(iteration, max_iterations)

    return {
        "theta": theta,
        "beta": beta,
        "alpha": alpha,
        "psi": psi,
        "converged": converged,
        "iterations": completed_iterations,
        "max_delta": max_delta,
        "log_likelihood": log_likelihood,
        "relative_log_likelihood_change": relative_log_likelihood_change,
    }


def run_text_scaling(
    articles: Mapping[str, Any],
    *,
    min_doc_words: int = 50,
    max_features: int = 5000,
    progress_cb: Callable[[int, int], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    max_iterations: int = 50,
    tolerance: float = 5e-5,
) -> tuple[dict, dict[str, dict], dict]:
    """Scale eligible articles and return a model, profiles, and corpus statistics."""
    try:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError(
            "Text scaling requires numpy and scikit-learn. Install the project "
            "requirements and try again."
        ) from exc

    if status_cb:
        status_cb("Preparing article text and stylometric profiles...")

    document_ids: list[str] = []
    texts: list[str] = []
    profiles: dict[str, dict] = {}
    excluded = 0
    article_items = list((articles or {}).items())
    for index, (article_id, article) in enumerate(article_items, start=1):
        _check_cancel(cancel_cb)
        text = _article_text(article)
        profile = build_stylometric_profile(str(article_id), article, text)
        if profile["word_count"] < max(1, int(min_doc_words)):
            excluded += 1
        else:
            document_ids.append(str(article_id))
            texts.append(text)
            profiles[str(article_id)] = profile
        if progress_cb and (index % 100 == 0 or index == len(article_items)):
            progress_cb(index, max(1, len(article_items)))

    if len(document_ids) < 2:
        raise ValueError(
            "Text scaling needs at least two articles with "
            f"{min_doc_words} or more words; found {len(document_ids)}."
        )

    if status_cb:
        status_cb(f"Building document-term matrix for {len(document_ids)} articles...")
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max(2, int(max_features)),
        min_df=2 if len(document_ids) >= 3 else 1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{1,}\b",
    )
    # CountVectorizer.fit_transform ships no type stubs; sklearn's own return-type
    # inference conflates it with an internal helper's tuple return, so pyright
    # infers nonsense (tuple[Unknown, set[Unknown]] | None) instead of the actual
    # scipy sparse matrix. Annotate explicitly rather than let that noise cascade.
    counts: Any = vectorizer.fit_transform(texts)
    nonempty_rows = np.asarray(counts.sum(axis=1)).ravel() > 0
    if not bool(np.all(nonempty_rows)):
        kept_indices = np.flatnonzero(nonempty_rows)
        dropped_indices = np.flatnonzero(~nonempty_rows)
        for dropped_index in dropped_indices:
            profiles.pop(document_ids[int(dropped_index)], None)
        document_ids = [document_ids[int(index)] for index in kept_indices]
        texts = [texts[int(index)] for index in kept_indices]
        counts = counts[kept_indices]
        excluded += int(len(dropped_indices))
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError(
            "The eligible articles do not contain enough shared vocabulary for text scaling."
        )

    if status_cb:
        status_cb(
            f"Fitting Wordfish model ({counts.shape[0]} documents, "
            f"{counts.shape[1]} features)..."
        )
    fitted = _fit_wordfish(
        counts,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    terms = vectorizer.get_feature_names_out()
    theta = fitted.pop("theta")
    beta = fitted.pop("beta")
    fitted.pop("alpha")
    fitted.pop("psi")

    positions = {article_id: float(theta[index]) for index, article_id in enumerate(document_ids)}
    for article_id, position in positions.items():
        profiles[article_id]["position"] = round(position, 6)

    top_indices = np.argsort(np.abs(beta))[::-1][: min(50, len(beta))]
    discriminating_terms = [
        {"term": str(terms[index]), "beta": round(float(beta[index]), 6)}
        for index in top_indices
    ]
    average_hedge = float(np.mean([p["hedge_density"] for p in profiles.values()]))
    average_certainty = float(
        np.mean([p["certainty_density"] for p in profiles.values()])
    )

    model = {
        "algorithm": "Wordfish Poisson text scaling",
        "n_documents": counts.shape[0],
        "n_features": counts.shape[1],
        "positions": positions,
        "discriminating_terms": discriminating_terms,
        **fitted,
    }
    statistics = {
        "n_documents": counts.shape[0],
        "n_features": counts.shape[1],
        "excluded_documents": excluded,
        "converged": bool(model["converged"]),
        "iterations": int(model["iterations"]),
        "max_delta": round(float(model["max_delta"]), 6),
        "relative_log_likelihood_change": round(
            float(model["relative_log_likelihood_change"]), 8
        ),
        "average_hedge_density": round(average_hedge, 4),
        "average_certainty_density": round(average_certainty, 4),
    }
    if status_cb:
        convergence = "converged" if model["converged"] else "reached iteration limit"
        status_cb(
            f"Text scaling {convergence}: {counts.shape[0]} profiles, "
            f"{counts.shape[1]} features"
        )
    return model, profiles, statistics
