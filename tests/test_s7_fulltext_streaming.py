"""Memory-focused tests for Stage 7 full-text streaming."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import cast

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acquire import fulltext as acquire_fulltext
from core.config import Config
from stages import s7_fulltext


class _FakeRequests:
    """Minimal fake requests module that yields deterministic HTML bodies."""

    class _Response:
        def __init__(
            self,
            text: str,
            status: int = 200,
            headers: dict | None = None,
            url: str = "https://example.test/paper",
        ):
            self.text = text
            self.status_code = status
            self.headers = headers or {"content-type": "text/html"}
            self.url = url
            self.content = text.encode("utf-8")

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class _Session:
        headers: dict = {}

        def get(self, *args, **kwargs):
            body = (
                "<html><body>"
                "<p>Sample size N = 120.</p>"
                "<p>Cohen's d = 0.85, 95% CI [0.45, 1.25].</p>"
                "<p>p < 0.001</p>"
                "</body></html>"
            )
            return _FakeRequests._Response(body)

    @classmethod
    def Session(cls):
        return cls._Session()


class _MislabelledPdfRequests(_FakeRequests):
    class _Session(_FakeRequests._Session):
        def get(self, *args, **kwargs):
            body = "<!DOCTYPE html><html><body>" + ("Article text. " * 40) + "</body></html>"
            return _FakeRequests._Response(
                body,
                headers={"content-type": "application/pdf"},
                url="https://example.test/paper.pdf",
            )


def _fake_identify(config, claims_path=None):
    return [
        {"doi": "10.1234/stream", "title": "Stream Paper",
         "pmid": "", "method_type": "randomized_controlled_trial"},
    ]


def _fake_check_oa(targets, **kwargs):
    for t in targets:
        t["is_oa"] = True
        t["oa_url"] = "https://example.test/stream"
    return targets


def _fake_identify_two(config, claims_path=None):
    return [
        {"doi": "10.1234/a", "title": "Paper A", "pmid": "",
         "method_type": "randomized_controlled_trial"},
        {"doi": "10.1234/b", "title": "Paper B", "pmid": "",
         "method_type": "randomized_controlled_trial"},
    ]


def test_retrieve_fulltext_iter_yields_without_accumulating(monkeypatch, tmp_path):
    """The iterator yields one result at a time instead of building a list."""
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _FakeRequests)

    papers = [
        {"doi": "10.1234/a", "title": "A", "oa_url": "https://example.test/a"},
        {"doi": "10.1234/b", "title": "B", "oa_url": "https://example.test/b"},
    ]

    results = list(s7_fulltext.retrieve_fulltext_iter(papers, output_dir=tmp_path))

    assert len(results) == 2
    assert all(r.get("text") for r in results)
    assert all("saved_file" in r for r in results)


def test_retrieve_fulltext_iter_does_not_parse_html_as_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _MislabelledPdfRequests)

    def fail_pdf_parse(*args, **kwargs):
        raise AssertionError("HTML response must not be sent to the PDF parser")

    monkeypatch.setattr(s7_fulltext, "_extract_text_from_pdf", fail_pdf_parse)
    results = list(s7_fulltext.retrieve_fulltext_iter(
        [{
            "doi": "10.1234/html-at-pdf-url",
            "title": "HTML response",
            "oa_url": "https://example.test/paper.pdf",
        }],
        output_dir=tmp_path,
    ))

    assert results[0]["format"] == "html"
    assert results[0]["success"] is True


def test_preferred_download_does_not_parse_html_as_pdf(monkeypatch, tmp_path):
    client = _MislabelledPdfRequests._Session()

    def fail_pdf_parse(*args, **kwargs):
        raise AssertionError("HTML response must not be sent to the PDF parser")

    monkeypatch.setattr(acquire_fulltext, "extract_pdf_text", fail_pdf_parse)
    result = acquire_fulltext.download_preferred_fulltext(
        client,
        "https://example.test/paper.pdf",
        tmp_path,
        "paper",
        logging.getLogger(__name__),
        min_text_chars=20,
    )

    assert result is not None
    assert result["source_format"] == "html"
    assert not (tmp_path / "paper.pdf").exists()


def test_run_streaming_writes_incremental_jsonl(monkeypatch, tmp_path):
    """Streaming Stage 7 produces JSON Lines files and final arrays."""
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _FakeRequests)
    monkeypatch.setattr(s7_fulltext, "identify_targets", _fake_identify)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)

    class _Config:
        data_dir = tmp_path
        pubmed_email = "test@example.test"
        fulltext_download_workers = 1
        fulltext_streaming = True
        fulltext_download_batch_size = 10

    config = _Config()
    ft_dir = tmp_path / "fulltext"
    ft_dir.mkdir(parents=True, exist_ok=True)

    result = s7_fulltext.run(config=cast(Config, config))

    assert result["downloaded"] == 1
    assert result["metrics_extracted"] >= 1
    assert (ft_dir / "auto_extracted_metrics.json").exists()
    assert (ft_dir / "download_log.json").exists()
    assert (ft_dir / "auto_extracted_metrics.jsonl").exists()
    assert (ft_dir / "download_log.jsonl").exists()
    metrics = json.loads((ft_dir / "auto_extracted_metrics.json").read_text())
    assert isinstance(metrics, list)
    assert len(metrics) == 1
    assert metrics[0]["doi"] == "10.1234/stream"


def test_run_streaming_resumes_jsonl_without_redownloading_completed_paper(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(s7_fulltext, "identify_targets", _fake_identify_two)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)

    ft_dir = tmp_path / "fulltext"
    ft_dir.mkdir(parents=True)
    (ft_dir / "download_log.jsonl").write_text(
        json.dumps({
            "resume_key": "doi:10.1234/a",
            "doi": "10.1234/a",
            "title": "Paper A",
            "status": "ok",
            "format": "html",
            "file": "paper-a.html",
            "size_bytes": 100,
        }) + "\n",
        encoding="utf-8",
    )
    (ft_dir / "auto_extracted_metrics.jsonl").write_text(
        json.dumps({
            "resume_key": "doi:10.1234/a",
            "doi": "10.1234/a",
            "title": "Paper A",
            "metrics": {"sample_sizes": [{"value": 40}]},
        }) + "\n" + '{"resume_key":',
        encoding="utf-8",
    )

    downloaded: list[str] = []

    def fake_retrieve(papers, **kwargs):
        downloaded.extend(paper["doi"] for paper in papers)
        for paper in papers:
            yield {
                **paper,
                "text": "Sample size N = 120. Cohen's d = 0.85. p < 0.001.",
                "format": "html",
                "saved_file": str(ft_dir / "paper-b.html"),
                "size_bytes": 200,
                "success": True,
            }

    monkeypatch.setattr(s7_fulltext, "retrieve_fulltext_iter", fake_retrieve)

    class _Config:
        data_dir = tmp_path
        pubmed_email = "test@example.test"
        fulltext_download_workers = 1
        fulltext_streaming = True
        fulltext_download_batch_size = 10

    result = s7_fulltext.run(config=cast(Config, _Config()))

    assert downloaded == ["10.1234/b"]
    assert result["resumed"] == 1
    assert result["downloaded"] == 2
    assert result["remaining"] == 0
    metrics = json.loads(
        (ft_dir / "auto_extracted_metrics.json").read_text(encoding="utf-8")
    )
    downloads = json.loads(
        (ft_dir / "download_log.json").read_text(encoding="utf-8")
    )
    assert {row["doi"] for row in metrics} == {"10.1234/a", "10.1234/b"}
    assert {row["doi"] for row in downloads} == {"10.1234/a", "10.1234/b"}


def test_cancelled_streaming_run_keeps_checkpoint_for_next_resume(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(s7_fulltext, "identify_targets", _fake_identify_two)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)

    class _CancelEvent:
        stopped = False

        def is_set(self):
            return self.stopped

    cancel_event = _CancelEvent()

    def cancelling_retrieve(papers, **kwargs):
        first = papers[0]
        yield {
            **first,
            "text": "Sample size N = 20. Cohen's d = 0.5.",
            "format": "html",
            "saved_file": str(tmp_path / "fulltext" / "paper-a.html"),
            "size_bytes": 100,
            "success": True,
        }
        cancel_event.stopped = True

    monkeypatch.setattr(s7_fulltext, "retrieve_fulltext_iter", cancelling_retrieve)

    class _Config:
        data_dir = tmp_path
        pubmed_email = "test@example.test"
        fulltext_download_workers = 1
        fulltext_streaming = True
        fulltext_download_batch_size = 10

    first = s7_fulltext.run(config=cast(Config, _Config()), cancel_event=cancel_event)
    ft_dir = tmp_path / "fulltext"

    assert first["cancelled"] is True
    assert first["downloaded"] == 1
    assert first["remaining"] == 1
    assert len((ft_dir / "download_log.jsonl").read_text().splitlines()) == 1
    assert len((ft_dir / "auto_extracted_metrics.jsonl").read_text().splitlines()) == 1

    cancel_event.stopped = False
    requested: list[str] = []

    def resume_retrieve(papers, **kwargs):
        requested.extend(paper["doi"] for paper in papers)
        for paper in papers:
            yield {
                **paper,
                "text": "Sample size N = 30. Cohen's d = 0.6.",
                "format": "html",
                "saved_file": str(ft_dir / "paper-b.html"),
                "size_bytes": 100,
                "success": True,
            }

    monkeypatch.setattr(s7_fulltext, "retrieve_fulltext_iter", resume_retrieve)
    second = s7_fulltext.run(config=cast(Config, _Config()), cancel_event=cancel_event)

    assert requested == ["10.1234/b"]
    assert second["resumed"] == 1
    assert second["downloaded"] == 2
    assert second["cancelled"] is False


def test_run_non_streaming_still_works(monkeypatch, tmp_path):
    """The legacy non-streaming path remains functional."""
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _FakeRequests)
    monkeypatch.setattr(s7_fulltext, "identify_targets", _fake_identify)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)

    class _Config:
        data_dir = tmp_path
        pubmed_email = "test@example.test"
        fulltext_download_workers = 1
        fulltext_streaming = False
        fulltext_download_batch_size = 10

    config = _Config()
    (tmp_path / "fulltext").mkdir(parents=True, exist_ok=True)

    result = s7_fulltext.run(config=cast(Config, config))

    assert result["downloaded"] == 1
    assert (tmp_path / "fulltext" / "auto_extracted_metrics.json").exists()


def test_run_non_streaming_honors_configured_buffer_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(s7_fulltext, "identify_targets", _fake_identify_two)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)
    batch_sizes: list[int] = []
    worker_values: list[int] = []

    def fake_retrieve(papers, **kwargs):
        batch_sizes.append(len(papers))
        worker_values.append(kwargs["workers"])
        return [
            {
                **paper,
                "text": "Sample size N = 50. Cohen's d = 0.7.",
                "format": "html",
                "saved_file": str(tmp_path / "fulltext" / f"{index}.html"),
                "size_bytes": 100,
                "success": True,
            }
            for index, paper in enumerate(papers)
        ]

    monkeypatch.setattr(s7_fulltext, "retrieve_fulltext", fake_retrieve)

    class _Config:
        data_dir = tmp_path
        pubmed_email = "test@example.test"
        fulltext_download_workers = 4
        fulltext_streaming = False
        fulltext_download_batch_size = 1

    result = s7_fulltext.run(config=cast(Config, _Config()))

    assert batch_sizes == [1, 1]
    assert worker_values == [4, 4]
    assert result["downloaded"] == 2


def test_identify_targets_releases_loaded_lists(tmp_path, monkeypatch):
    """Loaded profile/claims lists are deleted after processing to free memory."""
    method_dir = tmp_path / "method_review"
    method_dir.mkdir(parents=True, exist_ok=True)
    claims_d = tmp_path / "claims"
    claims_d.mkdir(parents=True, exist_ok=True)

    profiles = [
        {
            "doi": "10.1234/sr",
            "pmid": "1",
            "title": "A systematic review",
            "method_profile": {"study_method_type": "systematic review"},
        }
    ]
    claims = [
        {
            "doi": "10.1234/rct",
            "pmid": "2",
            "title": "An RCT",
            "methodology": "randomized controlled trial",
        }
    ]
    method_dir.joinpath("method_profiles_validated.json").write_text(
        json.dumps(profiles), encoding="utf-8"
    )
    claims_d.joinpath("claims.json").write_text(
        json.dumps(claims), encoding="utf-8"
    )

    class _Config:
        data_dir = tmp_path
        claims_dir = claims_d

    targets = s7_fulltext.identify_targets(_Config())
    dois = {t["doi"] for t in targets}
    assert dois == {"10.1234/sr", "10.1234/rct"}


def test_resume_retries_incomplete_and_failed_papers(monkeypatch, tmp_path):
    """Only a matching successful download + metric record can be skipped."""
    from types import SimpleNamespace
    from core.io import atomic_append_jsonl

    config = SimpleNamespace(data_dir=tmp_path, fulltext_streaming=True)
    papers = [{"doi": f"10.1234/{name}", "title": name}
              for name in ("done", "failed", "missing-metrics", "metrics-only")]
    monkeypatch.setattr(s7_fulltext, "identify_targets", lambda *a, **k: papers)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _FakeRequests)
    ft = tmp_path / "fulltext"
    for paper in papers[:3]:
        atomic_append_jsonl(ft / "download_log.jsonl", {
            **paper, "status": "failed" if paper["title"] == "failed" else "ok"})
    for paper in (papers[0], papers[1], papers[3]):
        atomic_append_jsonl(ft / "auto_extracted_metrics.jsonl", {
            **paper, "metrics": {"sample_sizes": []}})
    downloaded = []
    original = s7_fulltext.retrieve_fulltext_iter

    def retrieve(pending, **kwargs):
        downloaded.extend(p["title"] for p in pending)
        yield from original(pending, **kwargs)

    monkeypatch.setattr(s7_fulltext, "retrieve_fulltext_iter", retrieve)
    result = s7_fulltext.run(config)
    assert downloaded == ["failed", "missing-metrics", "metrics-only"]
    assert result["downloaded"] == 4
    assert result["metrics_extracted"] == 3  # completed zero-metric paper is valid
    for name in ("download_log", "auto_extracted_metrics"):
        final = json.loads((ft / f"{name}.json").read_text())
        assert len(final) == 4
        assert {r["doi"] for r in final} == {p["doi"] for p in papers}

    downloaded.clear()
    s7_fulltext.run(config, resume=False)
    assert downloaded == [p["title"] for p in papers]
    assert len((ft / "download_log.jsonl").read_text().splitlines()) == 4


def test_cancel_at_iterator_end_does_not_publish_partial_outputs(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    cancel = threading.Event()
    monkeypatch.setattr(s7_fulltext, "identify_targets", _fake_identify)
    monkeypatch.setattr(s7_fulltext, "check_oa_status", _fake_check_oa)

    def retrieve(papers, **kwargs):
        yield {**papers[0], "text": "N = 120"}
        cancel.set()

    monkeypatch.setattr(s7_fulltext, "retrieve_fulltext_iter", retrieve)
    result = s7_fulltext.run(SimpleNamespace(data_dir=tmp_path), cancel_event=cancel)
    assert result == {"cancelled": True, "downloaded": 1, "metrics_extracted": 1,
                      "resumed": 0, "remaining": 0}
    ft = tmp_path / "fulltext"
    for name in ("download_log", "auto_extracted_metrics"):
        assert (ft / f"{name}.jsonl").exists()
        assert not (ft / f"{name}.json").exists()


def test_empty_active_run_clears_both_final_outputs(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(s7_fulltext, "identify_targets", lambda *a, **k: [])
    ft = tmp_path / "fulltext"
    ft.mkdir()
    for name in ("download_log", "auto_extracted_metrics"):
        (ft / f"{name}.json").write_text('[{"doi": "stale"}]')
    result = s7_fulltext.run(SimpleNamespace(data_dir=tmp_path))
    assert result["targets"] == 0
    for name in ("download_log", "auto_extracted_metrics"):
        assert json.loads((ft / f"{name}.json").read_text()) == []
