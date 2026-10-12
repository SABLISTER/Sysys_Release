from __future__ import annotations

import json
from pathlib import Path

from core.config import Config
from process.committee_brief import (
    export_abct_committee_brief,
    export_abct_tech_ai_infosheet,
)


ROOT = Path(__file__).resolve().parent.parent


def test_export_abct_committee_brief_pack_from_partial_pipeline(tmp_path):
    cfg = Config.load_config(
        config_path=ROOT / "templates" / "abct_ai_digital_mental_health.yaml",
        project_root=tmp_path,
    )
    cfg.ensure_dirs()

    corpus = [
        {
            "title": "AI chatbot for depression symptoms",
            "year": 2025,
            "journal": "Digital Mental Health",
            "authors": ["Example A"],
            "abstract": "A randomized trial of a conversational agent.",
            "relevance_score": 5,
        },
        {
            "title": "Passive sensing for anxiety relapse",
            "year": 2024,
            "journal": "Behavioral Health Tech",
            "authors": ["Example B"],
            "abstract": "A prospective monitoring study.",
            "relevance_score": 4,
        },
    ]
    relevant = [corpus[0]]
    claims = [
        {
            **corpus[0],
            "claims": {
                "safety_privacy_equity": [
                    "Crisis escalation and consent procedures require explicit governance."
                ],
                "implementation_notes": ["Clinician oversight is needed before deployment."],
                "clinical_readiness": "Promising but not ready for unsupervised use.",
            },
        }
    ]

    (cfg.corpus_dir / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (cfg.claims_dir / "relevant.json").write_text(json.dumps(relevant), encoding="utf-8")
    (cfg.claims_dir / "claims_filtered.json").write_text(json.dumps(claims), encoding="utf-8")
    (cfg.clusters_dir / "abstract_draft.md").write_text(
        "Evidence is strongest for narrow, supervised digital mental health use cases.",
        encoding="utf-8",
    )

    out_dir = export_abct_committee_brief(cfg, tmp_path / "brief_pack")

    brief = out_dir / "abct_ai_digital_mental_health_brief.md"
    assert brief.exists()
    text = brief.read_text(encoding="utf-8")

    assert "AI in Digital Mental Health: Evidence Brief" in text
    assert "June 30, 2026" in text
    assert "PRISMA-trAIce Notes" in text
    assert "Crisis escalation" in text
    assert "AI chatbot for depression symptoms" in text

    assert (out_dir / "prisma_traice_checklist.md").exists()
    assert (out_dir / "prisma_traice_checklist.csv").exists()
    assert (out_dir / "abct_brief_corpus.bib").exists()
    assert (out_dir / "README.md").exists()


def test_export_abct_tech_ai_infosheet_from_partial_pipeline(tmp_path):
    cfg = Config.load_config(
        config_path=ROOT / "templates" / "abct_tech_ai_infosheet.yaml",
        project_root=tmp_path,
    )
    cfg.ensure_dirs()

    corpus = [
        {
            "title": "Human oversight for AI mental health chatbots",
            "year": 2026,
            "journal": "Journal of Digital Behavioral Health",
            "authors": ["Example C"],
            "abstract": "Implementation guidance for supervised conversational agents.",
            "relevance_score": 5,
        },
        {
            "title": "Equity risks in digital mental health AI",
            "year": 2025,
            "journal": "Implementation Science",
            "authors": ["Example D"],
            "abstract": "A review of access, language, and bias concerns.",
            "relevance_score": 4,
        },
    ]
    claims = [
        {
            **corpus[0],
            "claims": {
                "safety_privacy_equity": [
                    "Consent, privacy, and crisis escalation procedures should be explicit."
                ],
                "implementation_notes": ["Use supervised workflows with clinician review."],
                "clinical_readiness": "Appropriate only as an adjunct to care.",
            },
        }
    ]

    (cfg.corpus_dir / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (cfg.claims_dir / "relevant.json").write_text(json.dumps(corpus), encoding="utf-8")
    (cfg.claims_dir / "claims_filtered.json").write_text(json.dumps(claims), encoding="utf-8")

    out_dir = export_abct_tech_ai_infosheet(cfg, tmp_path / "infosheet_pack")

    handout = out_dir / "abct_tech_ai_infosheet.md"
    assert handout.exists()
    text = handout.read_text(encoding="utf-8")

    assert "AI in Digital Mental Health: Ethical and Practical Guidance" in text
    assert "June 22, 2026" in text
    assert "June 22 Takeaways" in text
    assert "Practical Guidance Checks" in text
    assert "Consent, privacy, and crisis escalation" in text
    assert "Human oversight for AI mental health chatbots" in text

    assert (out_dir / "prisma_traice_checklist.md").exists()
    assert (out_dir / "abct_infosheet_corpus.bib").exists()
    assert (out_dir / "README.md").exists()
