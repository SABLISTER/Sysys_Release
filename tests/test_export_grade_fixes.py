"""Regression tests: export corpus source paths and GRADE tier vocabulary."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from process.export import load_corpus_for_export
from process.grade import _evidence_tier
from process.method_review import ALLOWED_METHOD_TYPES


def _config(tmp_path: Path):
    data_dir = tmp_path / "data"
    claims_dir = data_dir / "claims"
    corpus_dir = data_dir / "corpus"
    output_dir = tmp_path / "output"
    for d in (claims_dir, corpus_dir, output_dir):
        d.mkdir(parents=True)
    return SimpleNamespace(
        data_dir=data_dir,
        claims_dir=claims_dir,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
    )


def test_export_prefers_claims_dir_filtered_corpus(tmp_path):
    cfg = _config(tmp_path)
    filtered = [{"title": "filtered paper"}]
    (cfg.claims_dir / "claims_filtered.json").write_text(json.dumps(filtered))
    # An unfiltered corpus also exists; the filtered one must win.
    (cfg.corpus_dir / "corpus.json").write_text(
        json.dumps([{"title": "unfiltered paper"}])
    )
    assert load_corpus_for_export(cfg) == filtered


def test_export_falls_back_to_claims_dir_relevant(tmp_path):
    cfg = _config(tmp_path)
    relevant = [{"title": "relevant paper"}]
    (cfg.claims_dir / "relevant.json").write_text(json.dumps(relevant))
    (cfg.corpus_dir / "corpus.json").write_text(
        json.dumps([{"title": "unfiltered paper"}])
    )
    assert load_corpus_for_export(cfg) == relevant


def test_evidence_tier_covers_method_review_vocabulary():
    # Every method type the pipeline actually emits must map to a real tier,
    # not "unknown" (which risk-of-bias lumps in with the low bucket).
    for method_type in ALLOWED_METHOD_TYPES:
        if method_type == "unknown":
            continue
        assert _evidence_tier(method_type) in ("high", "moderate", "low"), method_type


def test_evidence_tier_spot_checks():
    assert _evidence_tier("randomized_controlled_trial") == "high"
    assert _evidence_tier("systematic_review_meta_analysis") == "high"
    assert _evidence_tier("cohort_observational") == "moderate"
    assert _evidence_tier("case_control") == "moderate"
    assert _evidence_tier("case_report_case_series") == "low"
    assert _evidence_tier("cross_sectional") == "low"
    assert _evidence_tier("not_a_real_design") == "unknown"
