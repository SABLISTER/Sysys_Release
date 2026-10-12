from __future__ import annotations

import json
from pathlib import Path

from scripts.train_dr_brt_bert import (
    Triplet,
    _json_safe,
    dataset_fingerprint,
    load_pairwise_jsonls,
    split_by_query,
)


def test_load_pairwise_jsonls_deduplicates_and_skips_incomplete_rows(tmp_path):
    source = tmp_path / "pairs.jsonl"
    records = [
        {"query": " Question A ", "positive": "Good  paper", "negative": "Bad paper"},
        {"query": "Question A", "positive": "Good paper", "negative": "Bad paper"},
        {"query": "Question B", "positive": "Positive", "negative": ""},
        {"query": "Question C", "positive": "Relevant", "negative": "Irrelevant"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    rows = load_pairwise_jsonls([source])

    assert [(row.query, row.positive, row.negative) for row in rows] == [
        ("Question A", "Good paper", "Bad paper"),
        ("Question C", "Relevant", "Irrelevant"),
    ]


def test_split_by_query_is_deterministic_and_has_no_query_leakage():
    rows = [
        Triplet(f"query-{query}", f"positive-{query}-{item}", f"negative-{query}-{item}")
        for query in range(20)
        for item in range(3)
    ]

    train_a, eval_a = split_by_query(rows, eval_fraction=0.2, seed=42)
    train_b, eval_b = split_by_query(rows, eval_fraction=0.2, seed=42)

    assert train_a == train_b
    assert eval_a == eval_b
    assert {row.query for row in train_a}.isdisjoint({row.query for row in eval_a})
    assert len({row.query for row in eval_a}) == 4


def test_dataset_fingerprint_changes_when_training_content_changes():
    first = [Triplet("q", "positive", "negative")]
    second = [Triplet("q", "different positive", "negative")]

    assert dataset_fingerprint(first) == dataset_fingerprint(first)
    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_json_safe_serializes_nested_windows_paths():
    windows_path = Path("C:/data/train.jsonl")
    payload = {"paths": [windows_path]}

    safe = _json_safe(payload)

    assert safe == {"paths": [str(windows_path)]}
    json.dumps(safe)
