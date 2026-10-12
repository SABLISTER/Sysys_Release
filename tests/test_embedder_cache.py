"""Tests for cross-encoder model cache eviction (RAM reduction)."""
from __future__ import annotations

from process import relevance_embedder


def test_clear_model_cache_evicts_entries():
    relevance_embedder._model_cache[("model-a", "cpu")] = object()
    relevance_embedder._model_cache[("model-b", "cpu")] = object()

    relevance_embedder.clear_model_cache()

    assert relevance_embedder._model_cache == {}


def test_clear_model_cache_is_safe_when_empty():
    relevance_embedder._model_cache.clear()
    relevance_embedder.clear_model_cache()
    assert relevance_embedder._model_cache == {}
