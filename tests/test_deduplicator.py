"""Tests for corpus deduplication canonical-record selection."""
from __future__ import annotations

from acquire.deduplicator import deduplicate_papers


def test_deduplicate_prefers_full_text_duplicate_and_merges_doi():
    papers = [
        {
            "title": "Autism connectivity study",
            "doi": "10.1000/example",
            "abstract": "Short abstract",
            "source": "pubmed",
        },
        {
            "title": "Autism connectivity study",
            "abstract": "Short abstract",
            "full_text": "Downloaded full text body",
            "full_text_path": "/tmp/paper.txt",
            "source": "openalex",
        },
    ]

    deduped = deduplicate_papers(papers)

    assert len(deduped) == 1
    assert deduped[0]["full_text"] == "Downloaded full text body"
    assert deduped[0]["doi"] == "10.1000/example"
    assert deduped[0]["source"] == "openalex+pubmed"


def test_deduplicate_prefers_richer_identifier_record_when_full_text_ties():
    papers = [
        {
            "title": "Sparse provider title",
            "doi": "https://doi.org/10.1000/rich",
            "abstract": "A",
            "source": "s2",
        },
        {
            "title": "Richer provider title",
            "doi": "10.1000/rich",
            "pmid": "12345",
            "pmc": "PMC12345",
            "s2_id": "S2-1",
            "abstract": "A richer abstract with more context.",
            "source": "openalex",
        },
    ]

    deduped = deduplicate_papers(papers)

    assert len(deduped) == 1
    assert deduped[0]["title"] == "Richer provider title"
    assert deduped[0]["pmid"] == "12345"
    assert deduped[0]["pmc"] == "PMC12345"
    assert deduped[0]["source"] == "openalex+s2"


def test_deduplicate_collapses_pmid_duplicates_without_doi():
    papers = [
        {
            "title": "Provider A title",
            "pmid": "98765",
            "abstract": "First abstract",
            "source": "pubmed",
        },
        {
            "title": "Provider B title",
            "pmid": "98765",
            "abstract": "First abstract",
            "s2_id": "S2-98765",
            "source": "s2",
        },
    ]

    deduped = deduplicate_papers(papers)

    assert len(deduped) == 1
    assert deduped[0]["pmid"] == "98765"
    assert deduped[0]["s2_id"] == "S2-98765"
    assert deduped[0]["source"] == "s2+pubmed"


def test_deduplicate_bridges_identifier_aliases():
    papers = [
        {"title": "OpenAlex A", "oa_id": "https://openalex.org/W123", "source": "openalex"},
        {"title": "OpenAlex B", "openalex_id": "W123", "doi": "10.1/bridge", "source": "wos"},
        {"title": "OpenAlex C", "doi": "10.1/bridge", "source": "scopus"},
    ]

    deduped = deduplicate_papers(papers)

    assert len(deduped) == 1
    assert deduped[0]["doi"] == "10.1/bridge"
    assert deduped[0]["source"] == "wos+openalex+scopus"
