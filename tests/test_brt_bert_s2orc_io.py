from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_s2orc_io():
    import sys

    name = "brt_bert_s2orc_io_test"
    path = Path(__file__).resolve().parent.parent / "brt-bert" / "s2orc_io.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_passage_from_legacy_s2orc_shape():
    s2orc_io = _load_s2orc_io()
    record = {
        "corpusid": 12345678,
        "title": "Example study title",
        "abstract": "Short abstract text about outcomes.",
        "venue": "Example Journal",
        "year": 2020,
        "sections": [
            {"section": "Introduction", "text": "Background on the topic."},
        ],
    }
    passage = s2orc_io.build_passage(record, include_intro=True)
    assert passage is not None
    assert passage.paper_id == "12345678"
    assert passage.title == "Example study title"
    assert "Background on the topic." in passage.passage
    assert passage.year == "2020"


def test_title_jaccard_and_title_to_query():
    s2orc_io = _load_s2orc_io()
    left = "Criticality in autism spectrum disorder models"
    right = "Criticality models in autism spectrum conditions"
    score = s2orc_io.title_jaccard(left, right)
    assert score >= 0.4
    query = s2orc_io.title_to_query(left)
    assert query.endswith("?")


def test_write_passages_jsonl_roundtrip(tmp_path: Path):
    s2orc_io = _load_s2orc_io()
    out = tmp_path / "passages.jsonl"
    rows = [
        s2orc_io.S2orcPassage(
            paper_id="1",
            title="T",
            abstract="A",
            passage="T\n\nA",
            venue="J",
            year="2020",
            source="test",
        )
    ]
    s2orc_io.write_passages_jsonl(out, rows)
    loaded = list(s2orc_io.iter_jsonl(out))
    assert loaded[0]["paper_id"] == "1"
    assert loaded[0]["year"] == "2020"


def test_mine_script_load_gold_triplets(tmp_path: Path):
    import sys

    mine_path = Path(__file__).resolve().parent.parent / "brt-bert" / "mine_s2orc_triplets.py"
    name = "mine_s2orc_triplets_test"
    spec = importlib.util.spec_from_file_location(name, mine_path)
    assert spec is not None and spec.loader is not None
    mine = importlib.util.module_from_spec(spec)
    sys.modules[name] = mine
    spec.loader.exec_module(mine)

    source = tmp_path / "gold.jsonl"
    source.write_text(
        json.dumps(
            {
                "query": "Does X affect Y?",
                "positive": "Paper A title\n\nAbstract A",
                "negative": "Paper B title\n\nAbstract B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = mine.load_gold_triplets(source)
    assert len(rows) == 1
    assert rows[0].query == "Does X affect Y?"
