#!/usr/bin/env python3
"""Stream S2ORC records into normalized passage JSONL for BRT-BERT training.

Output schema (one JSON object per line)::

    {
      "paper_id": "12345678",
      "title": "Example study title",
      "abstract": "Short abstract text.",
      "passage": "Example study title\\n\\nShort abstract text.",
      "venue": "Example Journal",
      "year": "2020",
      "source": "/data/s2orc/s2orc.jsonl.gz"
    }

Usage::

    python brt-bert/prepare_s2orc_corpus.py \\
        --input /data/s2orc/s2orc.jsonl.gz \\
        --output /data/s2orc/s2orc_passages.jsonl \\
        --limit 500000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s2orc_io


logger = logging.getLogger("prepare_s2orc_corpus")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare S2ORC passage JSONL")
    parser.add_argument("--input", type=Path, help="Local S2ORC JSONL or .jsonl.gz")
    parser.add_argument(
        "--hf-dataset",
        default="",
        help="HuggingFace dataset id (streaming). Ignored when --input is set.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output passages JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Max records (0 = all)")
    parser.add_argument(
        "--include-intro",
        action="store_true",
        help="Append intro/body section text after abstract",
    )
    parser.add_argument("--intro-max-chars", type=int, default=4000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input and not args.hf_dataset:
        parser.error("Provide --input or --hf-dataset")

    def record_iter():
        if args.input:
            yield from s2orc_io.iter_jsonl(args.input)
            return
        yield from s2orc_io.iter_hf_s2orc(args.hf_dataset)

    source_label = str(args.input or args.hf_dataset)
    skipped = 0

    def passage_iter():
        nonlocal skipped
        kept = 0
        for index, record in enumerate(record_iter()):
            if args.limit and index >= args.limit:
                break
            passage = s2orc_io.build_passage(
                record,
                include_intro=args.include_intro,
                intro_max_chars=args.intro_max_chars,
                source=source_label,
            )
            if passage is None:
                skipped += 1
                continue
            kept += 1
            if (index + 1) % 10000 == 0:
                logger.info(
                    "Processed %d records (%d kept, %d skipped)",
                    index + 1,
                    kept,
                    skipped,
                )
            yield passage

    written = s2orc_io.write_passages_jsonl(args.output, passage_iter())
    logger.info("Wrote %d passages to %s (skipped %d)", written, args.output, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
