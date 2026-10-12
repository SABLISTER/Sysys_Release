"""Tests for the LLM request throughput optimizations.

Covers: prompt compression, the prompt-hash response cache in
``process.llm.chat``, concurrent claim extraction (order + equivalence
with sequential), the reduced stagger default, and parallel stage-4
method review.
"""
import json

import pytest

from core import checkpoint as checkpoint_module
from core.config import Config
from process import llm
from process.llm import compress_prompt_text


# ---------------------------------------------------------------------------
# Prompt compression
# ---------------------------------------------------------------------------

def test_compress_short_clean_text_unchanged():
    text = "A short abstract about drought and yield. " * 20  # ~860 chars
    assert len(text) < 2000
    assert compress_prompt_text(text) == text


def test_compress_strips_trailing_references():
    body = ("Introduction. Methods. Results. Discussion. " * 120).strip()
    refs = "\n\nReferences\n[1] Smith et al. 2020. A paper.\n[2] Jones 2021. Another.\n"
    text = body + refs
    assert len(text) > 2000
    out = compress_prompt_text(text)
    assert "Smith et al" not in out
    assert "References" not in out
    assert out.startswith("Introduction.")


def test_compress_strips_acknowledgments_and_funding():
    body = ("Results and discussion of the findings. " * 120).strip()
    tail = ("\n\nAcknowledgments\nWe thank the lab staff.\n"
            "Funding\nSupported by grant XYZ.\n"
            "\nReferences\n[1] Someone 2022.\n")
    out = compress_prompt_text(body + tail)
    assert "lab staff" not in out
    assert "grant XYZ" not in out
    assert "Someone 2022" not in out
    assert "findings." in out


def test_compress_does_not_cut_mid_text_mentions():
    # A mid-text mention of "references" must not truncate the body.
    body = ("We checked the references cited in prior work and then ran "
            "the experiment. " * 80).strip()
    assert len(body) > 2000
    assert "references" in body[: len(body) // 2]
    out = compress_prompt_text(body)
    assert out == body or "prior work" in out
    assert len(out) > len(body) * 0.9


def test_compress_collapses_redundant_whitespace():
    body = ("Sentence one.  Sentence two with  double spaces.\n\n\n\n"
            "Paragraph two. " * 60).strip()
    assert len(body) > 2000
    out = compress_prompt_text(body)
    assert "  " not in out
    assert "\n\n\n" not in out
    assert "Sentence one." in out and "Paragraph two." in out


def test_compress_empty_and_none_safe():
    assert compress_prompt_text("") == ""
    assert compress_prompt_text("tiny") == "tiny"


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

def _cache_config(tmp_path, **overrides):
    kw = dict(project_root=tmp_path, llm_provider="openai")
    kw.update(overrides)
    cfg = Config(**kw)
    llm.set_config(cfg)
    return cfg


def test_cache_miss_calls_provider_and_hit_does_not(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    calls = []

    def fake_uncached(route, model, messages, temperature, max_tokens, think):
        calls.append((model, messages))
        return '{"score": 4}'

    monkeypatch.setattr(llm, "_chat_uncached", fake_uncached)
    msgs = [{"role": "user", "content": "Score this paper"}]

    first = llm.chat("gpt-test", msgs)
    second = llm.chat("gpt-test", msgs)
    assert first == second == '{"score": 4}'
    assert len(calls) == 1  # second call served from cache

    cache_files = list((cfg.output_dir / "llm_cache").glob("*.json"))
    assert len(cache_files) == 1


def test_cache_key_varies_with_model_and_prompt(tmp_path, monkeypatch):
    _cache_config(tmp_path)
    calls = []

    def fake_uncached(route, model, messages, temperature, max_tokens, think):
        calls.append(model)
        return f"answer-from-{model}"

    monkeypatch.setattr(llm, "_chat_uncached", fake_uncached)
    msgs = [{"role": "user", "content": "hello"}]
    assert llm.chat("model-a", msgs) == "answer-from-model-a"
    assert llm.chat("model-b", msgs) == "answer-from-model-b"
    assert llm.chat("model-a", [{"role": "user", "content": "different"}]).startswith(
        "answer-from-model-a")
    assert len(calls) == 3


def test_cache_does_not_store_empty_responses(tmp_path, monkeypatch):
    _cache_config(tmp_path)
    calls = []

    def fake_uncached(route, model, messages, temperature, max_tokens, think):
        calls.append(1)
        return "   "

    monkeypatch.setattr(llm, "_chat_uncached", fake_uncached)
    msgs = [{"role": "user", "content": "hello"}]
    assert llm.chat("gpt-test", msgs) == "   "
    assert llm.chat("gpt-test", msgs) == "   "
    assert len(calls) == 2  # never cached


def test_cache_disabled_writes_nothing(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path, llm_cache_enabled=False)
    calls = []

    def fake_uncached(route, model, messages, temperature, max_tokens, think):
        calls.append(1)
        return "text"

    monkeypatch.setattr(llm, "_chat_uncached", fake_uncached)
    msgs = [{"role": "user", "content": "hello"}]
    assert llm.chat("gpt-test", msgs) == "text"
    assert llm.chat("gpt-test", msgs) == "text"
    assert len(calls) == 2
    assert not (cfg.output_dir / "llm_cache").exists()


def test_cache_key_is_stable_and_hex(tmp_path):
    _cache_config(tmp_path)
    msgs = [{"role": "user", "content": "hello"}]
    k1 = llm._llm_cache_key("openai", "m", 0, 100, None, msgs)
    k2 = llm._llm_cache_key("openai", "m", 0, 100, None, msgs)
    k3 = llm._llm_cache_key("openai", "m", 0, 100, None,
                            [{"role": "user", "content": "other"}])
    assert k1 == k2 and k1 != k3
    assert len(k1) == 64 and all(c in "0123456789abcdef" for c in k1)


# ---------------------------------------------------------------------------
# Concurrent extraction
# ---------------------------------------------------------------------------

def _papers(n):
    return [
        {
            "paper_id": f"p{i}",
            "title": f"Paper title {i}",
            "abstract": f"This is the abstract of paper {i}. " * 10,
        }
        for i in range(n)
    ]


def _fake_extract_claims(model, paper, think=None, max_tokens=4096,
                         research_config=None):
    i = paper["paper_id"]
    return {
        "findings": [f"finding-{i}"],
        "mechanisms": [],
        "brain_regions": [],
        "conditions": [],
        "error": "",
    }


def test_extraction_concurrent_matches_sequential(tmp_path, monkeypatch):
    import process.claim_extractor as ce

    monkeypatch.setattr(ce, "extract_claims", _fake_extract_claims)
    papers = _papers(12)

    seq, cancelled_seq = ce.extract_all_claims(
        papers, "m", checkpoint_dir=tmp_path / "ckpt1",
        concurrent_workers=1, stagger_delay=0)
    par, cancelled_par = ce.extract_all_claims(
        papers, "m", checkpoint_dir=tmp_path / "ckpt2",
        concurrent_workers=4, stagger_delay=0)

    assert not cancelled_seq and not cancelled_par
    assert [p["paper_id"] for p in par] == [p["paper_id"] for p in seq]
    for a, b in zip(seq, par):
        assert a["claims"] == b["claims"]
        assert a["claims"]["findings"] == [f"finding-{a['paper_id']}"]


def test_extraction_failed_paper_keeps_error_dict(tmp_path, monkeypatch):
    import process.claim_extractor as ce

    def boom(model, paper, think=None, max_tokens=4096, research_config=None):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(ce, "extract_claims", boom)
    papers = _papers(3)
    out, cancelled = ce.extract_all_claims(
        papers, "m", checkpoint_dir=tmp_path / "ckpt",
        concurrent_workers=2, stagger_delay=0)
    assert not cancelled
    assert len(out) == 3
    assert all(p["claims"]["error"] == "provider exploded" for p in out)


def test_extraction_stagger_default_reduced_and_configurable(tmp_path, monkeypatch):
    import process.claim_extractor as ce

    monkeypatch.setattr(ce, "extract_claims", _fake_extract_claims)
    captured = {}
    real_window = checkpoint_module.run_sliding_window

    def spy(**kw):
        captured.update(kw)
        return real_window(**kw)

    monkeypatch.setattr(checkpoint_module, "run_sliding_window", spy)
    papers = _papers(2)

    ce.extract_all_claims(papers, "m", checkpoint_dir=tmp_path / "c1")
    assert captured["stagger"] == 0.5
    assert captured["workers"] == 4  # function default without config

    ce.extract_all_claims(papers, "m", checkpoint_dir=tmp_path / "c2",
                          stagger_delay=2.0, concurrent_workers=1)
    assert captured["stagger"] == 2.0
    assert captured["workers"] == 1


def test_extraction_honors_config_workers(tmp_path, monkeypatch):
    import process.claim_extractor as ce

    monkeypatch.setattr(ce, "extract_claims", _fake_extract_claims)
    captured = {}
    real_window = checkpoint_module.run_sliding_window

    def spy(**kw):
        captured.update(kw)
        return real_window(**kw)

    monkeypatch.setattr(checkpoint_module, "run_sliding_window", spy)
    cfg = Config(project_root=tmp_path, llm_provider="openai")
    cfg.concurrent_workers = 6
    ce.extract_all_claims(_papers(2), "m", checkpoint_dir=tmp_path / "c3",
                          config=cfg)
    assert captured["workers"] == 6


def test_extraction_checkpoint_resume_still_works(tmp_path, monkeypatch):
    import process.claim_extractor as ce

    monkeypatch.setattr(ce, "extract_claims", _fake_extract_claims)
    papers = _papers(6)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed a checkpoint as if 2 papers were already extracted.
    seed_records = [
        {**papers[0], "claims": {"findings": ["old-0"], "mechanisms": [],
                                "brain_regions": [], "conditions": [],
                                "error": ""}},
        {**papers[1], "claims": {"findings": ["old-1"], "mechanisms": [],
                                "brain_regions": [], "conditions": [],
                                "error": ""}},
    ]
    (ckpt_dir / "extract_checkpoint.json").write_text(json.dumps({
        "processed_count": 2, "total": 6, "data": seed_records,
    }))

    resumed, cancelled = ce.extract_all_claims(
        papers, "m", checkpoint_dir=ckpt_dir,
        concurrent_workers=3, stagger_delay=0)
    assert not cancelled
    assert len(resumed) == 6
    # First two records come from the checkpoint untouched.
    assert resumed[0]["claims"]["findings"] == ["old-0"]
    assert resumed[1]["claims"]["findings"] == ["old-1"]
    # The rest were freshly extracted, in order.
    assert [p["paper_id"] for p in resumed] == [p["paper_id"] for p in papers]
    assert resumed[2]["claims"]["findings"] == ["finding-p2"]


# ---------------------------------------------------------------------------
# Stage 4 parallel validation
# ---------------------------------------------------------------------------

def _s4_config(tmp_path):
    return Config(project_root=tmp_path, llm_provider="openai")


def test_s4_parallel_matches_input_order_and_content(tmp_path, monkeypatch):
    from stages import s4_validate

    cfg = _s4_config(tmp_path)
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    papers = [
        {"paper_id": f"s4-{i}", "title": f"Trial of X {i}",
         "abstract": f"Randomized controlled trial number {i}. " * 20}
        for i in range(6)
    ]
    src = cfg.claims_dir / "relevant.json"
    src.write_text(json.dumps(papers))

    def fake_chat(model, messages, temperature=0, max_tokens=1500, think=None):
        content = messages[0]["content"]
        # Echo a per-paper allowed type so we can check record alignment.
        n = int(content.split("Trial of X ")[1][0])
        mtype = ("randomized_controlled_trial" if n % 2 == 0
                 else "cohort_observational")
        return json.dumps({
            "study_method_type": mtype,
            "confidence": 0.9,
            "sample_size": 100,
            "has_control_group": True,
            "is_longitudinal": False,
            "design_notes": "ok",
            "evidence_spans": [],
        })

    monkeypatch.setattr(llm, "chat", fake_chat)
    result = s4_validate.run(cfg, input_path=str(src), concurrent_workers=3)
    assert result["total"] == 6
    assert not result.get("cancelled")

    validated = json.loads(open(result["output_path"]).read())
    assert [r["paper_id"] for r in validated] == [p["paper_id"] for p in papers]
    for i, rec in enumerate(validated):
        mp = rec["method_profile"]
        expected = ("randomized_controlled_trial" if i % 2 == 0
                    else "cohort_observational")
        assert mp["study_method_type"] == expected
        assert mp["classification_method"] == "llm"


def test_s4_heuristic_fallback_still_works_in_parallel(tmp_path, monkeypatch):
    from stages import s4_validate

    cfg = _s4_config(tmp_path)
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    papers = [
        {"paper_id": "h0", "title": "No abstract here", "abstract": ""},
    ]
    src = cfg.claims_dir / "relevant.json"
    src.write_text(json.dumps(papers))

    def fake_chat(model, messages, temperature=0, max_tokens=1500, think=None):
        raise AssertionError("LLM must not be called without an abstract")

    monkeypatch.setattr(llm, "chat", fake_chat)
    result = s4_validate.run(cfg, input_path=str(src), concurrent_workers=2)
    validated = json.loads(open(result["output_path"]).read())
    assert validated[0]["method_profile"]["classification_method"] == \
        "heuristic_fallback"


# ---------------------------------------------------------------------------
# Stage 8 interrater parallel scoring
# ---------------------------------------------------------------------------

def _irr_papers():
    papers = []
    for i in range(5):
        papers.append({
            "paper_id": f"irr-{i}",
            "title": f"IRR Paper {i}",
            "abstract": f"Abstract of interrater paper {i}. " * 10,
            "relevance_score": 3,
            "findings": [f"orig-finding-{i}"],
            "mechanisms": [],
            "conditions": [],
            "brain_regions": [],
        })
    # Duplicate title: the old loop skipped the second occurrence.
    papers.append({
        "paper_id": "irr-dup",
        "title": "IRR Paper 0",
        "abstract": "Duplicate abstract. " * 10,
        "relevance_score": 3,
        "findings": [],
        "mechanisms": [],
        "conditions": [],
        "brain_regions": [],
    })
    return papers


def _irr_fake_chat(model, messages, temperature=0, max_tokens=4096, think=None):
    content = messages[0]["content"]
    n = int(content.split("IRR Paper ")[1][0])
    # One JSON satisfying both the relevance-score parser ("score") and
    # the claim-extraction parser (findings/mechanisms/...).
    return json.dumps({
        "score": n,
        "findings": [f"second-finding-{n}"],
        "mechanisms": [],
        "conditions": [],
        "brain_regions": [],
    })


def test_s8_parallel_scores_align_with_titles_and_skips_duplicates(
        tmp_path, monkeypatch):
    from stages import s8_interrater

    cfg = Config(project_root=tmp_path, llm_provider="openai")
    corpus_path = tmp_path / "claims.json"
    corpus_path.write_text(json.dumps(_irr_papers()))
    monkeypatch.setattr(llm, "chat", _irr_fake_chat)

    result = s8_interrater.run(
        claims_path=str(corpus_path),
        config=cfg,
        second_provider="openai",
        second_model="fake-m",
        sample_size=100,
        concurrent_workers=3,
    )
    assert "error" not in result
    assert not result.get("cancelled")

    records = json.loads(open(result["results_path"]).read())
    # 5 unique titles; the duplicate-title paper was skipped like before.
    assert len(records) == 5
    assert sorted(r["title"] for r in records) == \
        [f"IRR Paper {i}" for i in range(5)]
    for r in records:
        n = int(r["title"].split("IRR Paper ")[1])
        assert r["second_score"] == n
        assert r["second_findings"] == [f"second-finding-{n}"]
        assert r["original_findings"] == [f"orig-finding-{n}"]
