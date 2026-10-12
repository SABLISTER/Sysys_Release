"""Tests for silhouette-based automatic cluster-count selection."""
import numpy as np

from process import synthesizer


def _papers(n: int) -> list[dict]:
    return [{"title": f"Paper {i}", "abstract": f"abstract {i}"} for i in range(n)]


def _blob_embeddings(n_per_blob: int, n_blobs: int, dim: int = 8) -> np.ndarray:
    """Well-separated unit-norm blobs — silhouette should favour n_blobs."""
    rng = np.random.default_rng(7)
    rows = []
    for b in range(n_blobs):
        center = np.zeros(dim)
        center[b % dim] = 10.0
        rows.append(center + rng.normal(scale=0.05, size=(n_per_blob, dim)))
    return np.vstack(rows)


def test_select_k_by_silhouette_finds_blob_count():
    emb = _blob_embeddings(n_per_blob=6, n_blobs=3)
    best_k, scores = synthesizer._select_k_by_silhouette(emb)
    assert best_k == 3
    assert [k for k, _ in scores] == list(range(2, min(12, len(emb) - 1) + 1))
    assert all(-1.0 <= s <= 1.0 for _, s in scores)


def test_select_k_by_silhouette_tiny_input():
    emb = np.ones((2, 4))
    best_k, scores = synthesizer._select_k_by_silhouette(emb)
    assert best_k >= 1
    assert scores == []


def test_cluster_claims_auto_mode_selects_k(monkeypatch):
    n_per_blob, n_blobs = 5, 3
    papers = _papers(n_per_blob * n_blobs)
    monkeypatch.setattr(
        synthesizer,
        "_get_embeddings",
        lambda papers, model_name="all-MiniLM-L6-v2": _blob_embeddings(
            n_per_blob, n_blobs
        ),
    )

    k_info: dict = {}
    clusters = synthesizer.cluster_claims(papers, n_clusters=0, k_info=k_info)

    assert k_info["mode"] == "auto"
    assert k_info["chosen_k"] == n_blobs
    assert len(k_info["silhouette_scores"]) > 0
    assert len(clusters) == n_blobs
    assert sum(c["paper_count"] for c in clusters) == len(papers)


def test_cluster_claims_fixed_mode_reports_k(monkeypatch):
    papers = _papers(4)
    monkeypatch.setattr(
        synthesizer,
        "_get_embeddings",
        lambda papers, model_name="all-MiniLM-L6-v2": np.ones((len(papers), 2)),
    )

    k_info: dict = {}
    clusters = synthesizer.cluster_claims(papers, n_clusters=2, k_info=k_info)

    assert k_info == {"mode": "fixed", "chosen_k": 2, "silhouette_scores": []}
    assert len(clusters) == 2


def test_cluster_claims_n1_still_means_one_cluster(monkeypatch):
    papers = _papers(3)
    monkeypatch.setattr(
        synthesizer,
        "_get_embeddings",
        lambda papers, model_name="all-MiniLM-L6-v2": np.ones((len(papers), 2)),
    )

    clusters = synthesizer.cluster_claims(papers, n_clusters=1)
    assert len(clusters) == 1
    assert clusters[0]["paper_count"] == 3


def test_select_k_by_silhouette_without_sklearn(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sklearn)
    emb = _blob_embeddings(4, 2)
    best_k, scores = synthesizer._select_k_by_silhouette(emb)
    assert best_k == min(12, len(emb) - 1)
    assert scores == []
