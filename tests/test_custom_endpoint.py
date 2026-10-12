"""Custom LLM endpoint: URL normalization, model discovery, routing, migration."""
from __future__ import annotations

import json

import httpx
import pytest

from core.config import Config, migrate_legacy_llm_fields
from process import llm
from process.custom_endpoint import (
    CustomEndpoint,
    ModelListError,
    anthropic_base_url,
    list_models,
    openai_base_url,
    server_root,
)


# ── URL normalization ──────────────────────────────────────────────────

@pytest.mark.parametrize("url, expected", [
    ("http://localhost:1234", "http://localhost:1234/v1"),
    ("http://localhost:1234/", "http://localhost:1234/v1"),
    ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
    ("https://api.longcat.chat/openai/v1", "https://api.longcat.chat/openai/v1"),
])
def test_openai_base_url_adds_v1_only_to_bare_hosts(url, expected):
    assert openai_base_url(url) == expected


@pytest.mark.parametrize("url", ["http://h:1234", "http://h:1234/v1", "http://h:1234/v1/"])
def test_anthropic_base_url_drops_v1_because_sdk_appends_it(url):
    assert anthropic_base_url(url) == "http://h:1234"


@pytest.mark.parametrize("suffix", ["", "/v1", "/api", "/api/v1", "/api/v0/"])
def test_server_root_strips_api_paths(suffix):
    assert server_root("http://h:1234" + suffix) == "http://h:1234"


# ── Model discovery ────────────────────────────────────────────────────

def _transport(routes: dict[str, tuple[int, dict]], seen: list[httpx.Request]):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = routes.get(request.url.path, (404, {"error": "not found"}))
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handle)


def test_list_models_openai_format_sends_bearer_and_skips_embeddings():
    seen: list[httpx.Request] = []
    transport = _transport({"/v1/models": (200, {"data": [
        {"id": "qwen3-32b"}, {"id": "nomic-embed", "type": "embeddings"}, {"id": "qwen3-32b"},
    ]})}, seen)
    models = list_models(CustomEndpoint("openai", "http://h:1234", "k1"), transport=transport)
    assert models == ["qwen3-32b"]
    assert seen[0].headers["Authorization"] == "Bearer k1"


def test_list_models_anthropic_format_paginates_with_anthropic_headers():
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "after_id" in request.url.params:
            return httpx.Response(200, json={"data": [{"id": "m2"}], "has_more": False})
        return httpx.Response(200, json={"data": [{"id": "m1"}], "has_more": True, "last_id": "m1"})

    models = list_models(
        CustomEndpoint("anthropic", "https://gw.test/v1", "sk-ant"),
        transport=httpx.MockTransport(handle),
    )
    assert models == ["m1", "m2"]
    assert seen[0].url.path == "/v1/models"
    assert seen[0].headers["x-api-key"] == "sk-ant"
    assert seen[0].headers["anthropic-version"]
    assert seen[1].url.params["after_id"] == "m1"


def test_list_models_lmstudio_rest_reads_v1_keys():
    seen: list[httpx.Request] = []
    transport = _transport({"/api/v1/models": (200, {"models": [
        {"type": "llm", "key": "openai/gpt-oss-20b"},
        {"type": "embedding", "key": "text-embedding-nomic"},
    ]})}, seen)
    models = list_models(CustomEndpoint("lmstudio", "http://h:1234/v1"), transport=transport)
    assert models == ["openai/gpt-oss-20b"]
    assert "Authorization" not in seen[0].headers


def test_list_models_lmstudio_falls_back_to_v0_listing():
    seen: list[httpx.Request] = []
    transport = _transport({"/api/v0/models": (200, {"data": [{"id": "old-model", "type": "llm"}]})}, seen)
    models = list_models(CustomEndpoint("lmstudio", "http://h:1234"), transport=transport)
    assert models == ["old-model"]
    assert [r.url.path for r in seen] == ["/api/v1/models", "/api/v0/models"]


@pytest.mark.parametrize("status, fragment", [(401, "rejected the API key"), (500, "HTTP 500")])
def test_list_models_reports_http_errors(status, fragment):
    transport = _transport({"/v1/models": (status, {"error": "x"})}, [])
    with pytest.raises(ModelListError, match=fragment):
        list_models(CustomEndpoint("openai", "http://h:1"), transport=transport)


def test_list_models_reports_unreachable_server():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(ModelListError, match="Could not reach"):
        list_models(CustomEndpoint("openai", "http://h:1"), transport=httpx.MockTransport(refuse))


def test_list_models_requires_base_url_and_nonempty_result():
    with pytest.raises(ModelListError, match="base URL"):
        list_models(CustomEndpoint("openai", "  "))
    transport = _transport({"/v1/models": (200, {"data": []})}, [])
    with pytest.raises(ModelListError, match="no chat models"):
        list_models(CustomEndpoint("openai", "http://h:1"), transport=transport)


# ── Chat routing by API format ─────────────────────────────────────────

@pytest.fixture
def http_calls(monkeypatch, tmp_path):
    """Route every httpx client (including the OpenAI/Anthropic SDKs') to a mock.

    Recent OpenAI/Anthropic SDKs build their HTTP layer on the ``httpx2``
    fork instead of ``httpx``, so client construction is intercepted for both
    libraries. The SDKs instantiate the module-global
    ``SyncHttpxClientWrapper`` in their ``_base_client`` modules, which is
    looked up at call time and therefore patchable.
    """
    import importlib

    calls: list = []

    def build_response(response_cls, request):
        calls.append(request)
        if request.url.path.endswith("/chat/completions"):
            return response_cls(200, json={
                "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "openai-answer"}}],
            })
        return response_cls(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "anthropic-answer"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    def handle_httpx(request: httpx.Request) -> httpx.Response:
        return build_response(httpx.Response, request)

    real_client = httpx.Client

    class Client(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handle_httpx)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)

    try:
        import httpx2
    except ImportError:
        httpx2 = None

    if httpx2 is not None:
        def handle_httpx2(request):
            return build_response(httpx2.Response, request)

        for mod_name in ("openai._base_client", "anthropic._base_client"):
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            wrapper = getattr(mod, "SyncHttpxClientWrapper", None)
            if wrapper is None:
                continue

            def factory(*args, _wrapper=wrapper, **kwargs):
                kwargs["transport"] = httpx2.MockTransport(handle_httpx2)
                return _wrapper(*args, **kwargs)

            monkeypatch.setattr(mod, "SyncHttpxClientWrapper", factory)

    yield calls
    llm.set_config(Config(project_root=tmp_path))


def test_custom_openai_format_posts_to_configured_server(http_calls, tmp_path):
    cfg = Config(project_root=tmp_path, llm_provider="custom", custom_api_format="openai",
                 custom_base_url="http://gw.test:8000", custom_model="m")
    llm.set_config(cfg)
    assert llm.chat(cfg.llm_model, [{"role": "user", "content": "hi"}]) == "openai-answer"
    assert str(http_calls[0].url) == "http://gw.test:8000/v1/chat/completions"
    # Keyless local servers still get a syntactically valid bearer header.
    assert http_calls[0].headers["Authorization"] == "Bearer not-needed"


def test_custom_anthropic_format_posts_messages_with_key(http_calls, tmp_path):
    cfg = Config(project_root=tmp_path, llm_provider="custom", custom_api_format="anthropic",
                 custom_base_url="http://gw.test:8000/v1", custom_api_key="gw-key",
                 custom_model="m")
    llm.set_config(cfg)
    reply = llm.chat(cfg.llm_model, [{"role": "system", "content": "S"},
                                      {"role": "user", "content": "hi"}])
    assert reply == "anthropic-answer"
    assert str(http_calls[0].url) == "http://gw.test:8000/v1/messages"
    assert http_calls[0].headers["x-api-key"] == "gw-key"
    assert json.loads(http_calls[0].content)["system"] == "S"


def test_invalid_custom_api_format_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="custom API format"):
        Config(project_root=tmp_path, custom_api_format="grpc")


# ── Legacy provider migration ─────────────────────────────────────────

def test_legacy_longcat_config_file_migrates_to_custom_openai(tmp_path):
    path = tmp_path / "research_config.yaml"
    path.write_text(
        "llm_provider: longcat\n"
        "longcat_base_url: https://api.longcat.chat/openai/v1\n"
        "longcat_model: LongCat-2.0-Preview\n"
        "abstract_longcat_model: LongCat-Flash\n"
    )
    cfg = Config.load_config(path, project_root=tmp_path)
    assert cfg.llm_provider == "custom"
    assert cfg.custom_api_format == "openai"
    assert cfg.custom_base_url == "https://api.longcat.chat/openai/v1"
    assert cfg.llm_model == "LongCat-2.0-Preview"
    assert cfg.llm_model_for("abstract") == "LongCat-Flash"


def test_migration_never_overwrites_explicit_custom_values():
    out = migrate_legacy_llm_fields({
        "llm_provider": "lmstudio", "lmstudio_model": "old",
        "custom_model": "new", "custom_api_format": "anthropic",
    })
    assert out == {"llm_provider": "custom", "custom_model": "new",
                   "custom_api_format": "anthropic"}


def test_legacy_provider_names_still_resolve_for_stage_runs(tmp_path):
    from gui.stage_runner import resolve_second_rater
    cfg = Config(project_root=tmp_path, llm_provider="custom", custom_model="primary")
    _, provider, model = resolve_second_rater(cfg, "lmstudio", "")
    assert (provider, model) == ("custom", "primary")
