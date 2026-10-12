"""Exercise real stage entry points through a custom endpoint speaking LM Studio REST."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.config import Config
from process import llm


@pytest.fixture
def lmstudio_transport(monkeypatch, tmp_path):
    calls = []
    real_client = httpx.Client
    answer = json.dumps({
        "relevance_score": 5, "quoted_span": "Trial evidence", "relevance_quote": "Trial evidence",
        "findings": ["Trial evidence"], "mechanisms": [], "conditions": [], "brain_regions": [],
        "study_method_type": "randomized_controlled_trial", "evidence_spans": {},
        "represents_study_data": True, "usable_for_meta_analysis": True,
        "data_role": "study_result", "confidence": 0.9,
        "axis_scores": {"a": 0.8}, "overall_confidence": 0.9, "evidence_type": "empirical",
    })

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"output": [
            {"type": "reasoning", "content": "This must not replace the answer."},
            {"type": "message", "content": answer}], "stats": {}})

    class Client(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handle)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)
    def wrong_provider(*args, **kwargs):
        raise AssertionError("Stage attempted to use another provider")
    monkeypatch.setattr(llm, "_chat_ollama", wrong_provider)
    monkeypatch.setattr(llm, "_chat_openai", wrong_provider)
    cfg = Config(project_root=tmp_path, llm_provider="custom", custom_api_format="lmstudio",
                 custom_base_url="http://lmstudio.test:12345/v1/",
                 custom_api_key="contract-key", custom_model="stage-model",
                 relevance_model="", concurrent_workers=1, stagger_delay=0, score_batch_size=1)
    cfg.ensure_dirs()
    paper = {"title": "Trial evidence", "doi": "10.1234/contract", "relevance_score": 5,
             "abstract": "Trial evidence. We randomized 120 participants. N = 120. "
                         "The treatment effect was t(118) = 2.4, p = 0.02, Cohen's d = 0.45.",
             "findings": ["Trial evidence"], "claims": {"findings": ["Trial evidence"]}}
    for path in (cfg.corpus_dir / "corpus.json", cfg.claims_dir / "relevant.json",
                 cfg.claims_dir / "claims.json", cfg.claims_dir / "claims_filtered.json"):
        path.write_text(json.dumps([paper]))
    # A previous stage must never determine the next stage's provider.
    llm.set_config(Config(project_root=tmp_path, llm_provider="ollama"))
    yield cfg, calls, paper
    llm.set_config(Config(project_root=tmp_path))


@pytest.mark.parametrize("stage", ["score", "statistical_extract", "extract", "validate", "synthesize", "interrater", "hard_mode"])
@pytest.mark.parametrize("via_gui_runner", [False, True])
def test_each_llm_stage_contacts_configured_lmstudio(lmstudio_transport, monkeypatch, stage, via_gui_runner):
    cfg, calls, paper = lmstudio_transport
    from gui import stage_runner
    from stages import s2_score, s2c_statistical_extract, s3_extract, s4_validate, s6_synthesize, s8_interrater
    from process import synthesizer
    from hard_mode import scoring

    monkeypatch.setattr(synthesizer, "cluster_claims", lambda *a, **k: [
        {"cluster_id": 0, "paper_count": 1, "papers": [paper], "mechanisms": [],
         "conditions": [], "brain_regions": [], "primary_support_count": 1}])
    functions = {"score": s2_score.run, "statistical_extract": s2c_statistical_extract.run,
                 "extract": s3_extract.run, "validate": s4_validate.run,
                 "synthesize": s6_synthesize.run, "interrater": s8_interrater.run}
    kwargs = {}
    if stage == "statistical_extract":
        kwargs = {"use_llm_context_check": True, "max_llm_contexts": 1}
    if stage == "interrater":
        cfg.llm_provider = "openai"  # primary rater differs from the second
        kwargs = {"second_provider": "custom", "second_model": "second-model", "sample_size": 1}
    if stage == "hard_mode":
        # The workload can select LM Studio while the project default differs.
        cfg.llm_provider = "openai"
        cfg.abstract_llm_provider = "custom"
        result = scoring.score_corpus(cfg, {"axes": [{"id": "a"}]}, [paper])
        assert "error" not in result[0]["machine"]
    elif via_gui_runner:
        function, is_async = stage_runner._STAGE_REGISTRY[stage]
        assert is_async
        asyncio.run(function(cfg, **kwargs))
    else:
        functions[stage](config=cfg, **kwargs)

    expected_count = 2 if stage in {"synthesize", "interrater"} else 1
    assert len(calls) == expected_count
    for request in calls:
        assert str(request.url) == "http://lmstudio.test:12345/api/v1/chat"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer contract-key"
        body = json.loads(request.content)
        assert body["model"] == ("second-model" if stage == "interrater" else "stage-model")
        assert body["input"].strip()
        assert body["max_output_tokens"] > 0


@pytest.mark.parametrize("think, expected", [(None, None), (False, "off"), ("off", "off"), (True, "medium"), ("high", "high")])
def test_lmstudio_reasoning_setting_is_sent(lmstudio_transport, think, expected):
    cfg, calls, _ = lmstudio_transport
    llm.set_config(cfg)
    response = llm.chat(cfg.llm_model, [{"role": "system", "content": "Return JSON."},
                                    {"role": "user", "content": "Trial evidence"}], think=think)
    assert json.loads(response)["relevance_score"] == 5
    body = json.loads(calls[0].content)
    assert body.get("reasoning") == expected
    assert body["system_prompt"] == "Return JSON."


def test_interrater_restores_primary_config_on_failure(lmstudio_transport, monkeypatch):
    from gui import stage_runner
    from stages import s8_interrater
    cfg, _, _ = lmstudio_transport
    cfg.llm_provider = "openai"

    def fail(**kwargs):
        other = Config(project_root=cfg.project_root, llm_provider="custom", custom_api_format="lmstudio")
        llm.set_config(other)
        raise RuntimeError("Failed to read corpus")

    monkeypatch.setattr(s8_interrater, "run", fail)
    with pytest.raises(RuntimeError, match="Failed to read corpus"):
        asyncio.run(stage_runner.run_interrater(cfg, second_provider="lmstudio"))
    assert llm._config is cfg


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/api", "/api/v1/"])
def test_lmstudio_server_urls_and_optional_auth(lmstudio_transport, suffix):
    cfg, calls, _ = lmstudio_transport
    cfg.custom_base_url = "http://lmstudio.test:12345" + suffix
    cfg.custom_api_key = ""
    llm.set_config(cfg)
    llm.chat(cfg.llm_model, [{"role": "user", "content": "Test"}])
    assert str(calls[0].url) == "http://lmstudio.test:12345/api/v1/chat"
    assert "Authorization" not in calls[0].headers


def test_lmstudio_reconfigures_cached_client(lmstudio_transport):
    cfg, calls, _ = lmstudio_transport
    llm.set_config(cfg)
    llm.chat(cfg.llm_model, [{"role": "user", "content": "First stage"}])
    other = Config(project_root=cfg.project_root, llm_provider="custom", custom_api_format="lmstudio",
                   custom_base_url="http://second.test:4321/api/v1",
                   custom_api_key="second-key", custom_model="second-model")
    llm.set_config(other)
    llm.chat(other.llm_model, [{"role": "user", "content": "Next stage"}])
    assert str(calls[1].url) == "http://second.test:4321/api/v1/chat"
    assert calls[1].headers["Authorization"] == "Bearer second-key"
    assert json.loads(calls[1].content)["model"] == "second-model"


def test_statistical_full_context_mode_does_not_stop_at_eighty(lmstudio_transport):
    from stages import s2c_statistical_extract
    cfg, calls, paper = lmstudio_transport
    papers = [{**paper, 'doi': f'10.1234/context-{i}', 'title': f'Trial {i}'} for i in range(81)]
    (cfg.claims_dir / 'relevant.json').write_text(json.dumps(papers))
    s2c_statistical_extract.run(config=cfg, use_llm_context_check=True, max_llm_contexts=0)
    saved = json.loads((cfg.claims_dir / 'statistical_extractions.json').read_text())
    assert saved['llm_context_check']['max_contexts'] == 0
    assert len(calls) >= 81
    assert all(any('llm_context_review' in candidate for candidate in record['results'])
               for record in saved['records'])
