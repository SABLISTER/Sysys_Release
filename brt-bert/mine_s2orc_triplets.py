#!/usr/bin/env python3
"""Mine S2ORC hard-negative triplets for BRT-BERT cross-encoder fine-tuning.

Combines optional gold ``(query, positive, negative)`` JSONL with synthetic
and hard-mined rows from ``s2orc_passages.jsonl``.

Output schema::

    {
      "query": "What is the evidence regarding Example study title?",
      "positive": "Example study title\\n\\nShort abstract text.",
      "negative": "Confusable paper title\\n\\nDifferent abstract.",
      "source": "s2orc_hard",
      "hard": true
    }

Feeds directly into ``scripts/train_relevance_crossencoder.py --train``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s2orc_io


logger = logging.getLogger("mine_s2orc_triplets")

QUERY_ALIASES = ("query", "q", "anchor", "prompt", "question")
POS_ALIASES = ("positive", "pos", "doc_pos", "positive_text", "passage_pos", "relevant")
NEG_ALIASES = ("negative", "neg", "doc_neg", "negative_text", "passage_neg", "irrelevant")


@dataclass(frozen=True)
class Triplet:
    query: str
    positive: str
    negative: str
    source: str = "gold"
    hard: bool = False


def _first_present(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_gold_triplets(path: Path) -> list[Triplet]:
    rows: list[Triplet] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            query = _first_present(record, QUERY_ALIASES)
            positive = _first_present(record, POS_ALIASES)
            negative = _first_present(record, NEG_ALIASES)
            if query and positive and negative:
                rows.append(Triplet(query, positive, negative, source=str(path), hard=False))
    return rows


def load_passages(path: Path, *, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            if limit and index >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def _venue_bucket(venue: str) -> str:
    venue = venue.strip().lower()
    return venue[:48] if venue else "__unknown__"


def _index_by_venue(passages: list[dict]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(passages):
        buckets[_venue_bucket(str(row.get("venue") or ""))].append(index)
    return buckets


def _find_hard_negative(
    rng: random.Random,
    passages: list[dict],
    venue_index: dict[str, list[int]],
    *,
    anchor_title: str,
    query: str,
    positive_idx: int | None,
    min_jaccard: float,
    max_jaccard: float,
) -> str | None:
    bucket_key = (
        _venue_bucket(str(passages[positive_idx]["venue"]))
        if positive_idx is not None
        else "__unknown__"
    )
    candidates = venue_index.get(bucket_key) or list(range(len(passages)))
    rng.shuffle(candidates)

    best: tuple[float, str] | None = None
    for idx in candidates[:2000]:
        if positive_idx is not None and idx == positive_idx:
            continue
        row = passages[idx]
        title = str(row.get("title") or "")
        passage = str(row.get("passage") or "")
        if not passage:
            continue
        with_positive = s2orc_io.title_jaccard(anchor_title, title)
        with_query = s2orc_io.title_jaccard(query, title)
        if with_positive < min_jaccard or with_positive > max_jaccard:
            continue
        if with_query >= with_positive:
            continue
        score = with_positive - with_query
        if best is None or score > best[0]:
            best = (score, passage)
    return best[1] if best else None


def mine_hard_triplets(
    passages: list[dict],
    *,
    count: int,
    seed: int,
    min_jaccard: float,
    max_jaccard: float,
) -> list[Triplet]:
    rng = random.Random(seed)
    venue_index = _index_by_venue(passages)
    rows: list[Triplet] = []
    indices = list(range(len(passages)))
    rng.shuffle(indices)

    for idx in indices:
        if len(rows) >= count:
            break
        row = passages[idx]
        title = str(row.get("title") or "")
        passage = str(row.get("passage") or "")
        if not title or not passage:
            continue
        query = s2orc_io.title_to_query(title)
        negative = _find_hard_negative(
            rng,
            passages,
            venue_index,
            anchor_title=title,
            query=query,
            positive_idx=idx,
            min_jaccard=min_jaccard,
            max_jaccard=max_jaccard,
        )
        if not negative:
            continue
        rows.append(
            Triplet(
                query=query,
                positive=passage,
                negative=negative,
                source="s2orc_hard",
                hard=True,
            )
        )
    return rows


def augment_gold_with_hard_negatives(
    gold: list[Triplet],
    passages: list[dict],
    *,
    hard_ratio: float,
    seed: int,
    min_jaccard: float,
    max_jaccard: float,
) -> list[Triplet]:
    rng = random.Random(seed)
    venue_index = _index_by_venue(passages)
    title_to_idx = {
        str(row.get("title") or ""): index
        for index, row in enumerate(passages)
        if row.get("title")
    }
    augmented: list[Triplet] = []

    for row in gold:
        augmented.append(row)
        if rng.random() > hard_ratio:
            continue
        anchor_title = row.positive.split("\n\n", 1)[0]
        positive_idx = title_to_idx.get(anchor_title)
        negative = _find_hard_negative(
            rng,
            passages,
            venue_index,
            anchor_title=anchor_title,
            query=row.query,
            positive_idx=positive_idx,
            min_jaccard=min_jaccard,
            max_jaccard=max_jaccard,
        )
        if negative:
            augmented.append(
                Triplet(
                    query=row.query,
                    positive=row.positive,
                    negative=negative,
                    source=f"gold_hard:{row.source}",
                    hard=True,
                )
            )
    return augmented


def write_triplets(path: Path, rows: list[Triplet]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = (row.query, row.positive, row.negative)
            if key in seen:
                continue
            seen.add(key)
            payload = {
                "query": row.query,
                "positive": row.positive,
                "negative": row.negative,
                "source": row.source,
                "hard": row.hard,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine S2ORC hard-negative triplets")
    parser.add_argument("--corpus", type=Path, required=True, help="s2orc_passages.jsonl")
    parser.add_argument("--gold", nargs="*", type=Path, default=[], help="Gold pairwise JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-limit", type=int, default=0, help="Max passages to load")
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=50000,
        help="Pure S2ORC synthetic hard triplets to add",
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.5,
        help="Fraction of gold rows to augment with a hard negative",
    )
    parser.add_argument("--min-jaccard", type=float, default=0.25)
    parser.add_argument("--max-jaccard", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading corpus from %s", args.corpus)
    passages = load_passages(args.corpus, limit=args.corpus_limit or 0)
    if not passages:
        logger.error("No passages loaded from %s", args.corpus)
        return 1
    logger.info("Loaded %d passages", len(passages))

    all_rows: list[Triplet] = []
    for gold_path in args.gold:
        logger.info("Loading gold triplets from %s", gold_path)
        rows = load_gold_triplets(gold_path)
        logger.info("  -> %d triplets", len(rows))
        all_rows.extend(rows)

    if all_rows and args.hard_ratio > 0:
        logger.info("Augmenting gold triplets with hard negatives (ratio=%.2f)", args.hard_ratio)
        all_rows = augment_gold_with_hard_negatives(
            all_rows,
            passages,
            hard_ratio=args.hard_ratio,
            seed=args.seed,
            min_jaccard=args.min_jaccard,
            max_jaccard=args.max_jaccard,
        )

    if args.synthetic_count > 0:
        logger.info("Mining %d synthetic hard triplets from S2ORC", args.synthetic_count)
        synthetic = mine_hard_triplets(
            passages,
            count=args.synthetic_count,
            seed=args.seed + 1,
            min_jaccard=args.min_jaccard,
            max_jaccard=args.max_jaccard,
        )
        all_rows.extend(synthetic)
        logger.info("  -> mined %d synthetic rows", len(synthetic))

    written = write_triplets(args.output, all_rows)
    logger.info("Wrote %d unique triplets to %s", written, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
