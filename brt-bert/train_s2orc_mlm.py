#!/usr/bin/env python3
"""Continue MLM pretraining on S2ORC passages (Idea 1 — optional backbone stage).

Reads ``s2orc_passages.jsonl`` produced by ``prepare_s2orc_corpus.py`` and
continues masked-language-model training from SciBERT (or an existing MLM ckpt).

Usage::

    python brt-bert/train_s2orc_mlm.py \\
        --corpus /data/s2orc/s2orc_passages.jsonl \\
        --output brt-bert-s2orc-mlm \\
        --max-steps 100000

Then fine-tune the cross-encoder with::

    python scripts/train_relevance_crossencoder.py \\
        --checkpoint brt-bert-s2orc-mlm/final \\
        --train ... \\
        --output brt-bert-relevance-s2orc
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s2orc_io


logger = logging.getLogger("train_s2orc_mlm")

DEFAULT_BASE = "allenai/scibert_scivocab_uncased"


def _load_passage_texts(corpus: Path, *, limit: int = 0) -> list[str]:
    texts: list[str] = []
    for index, record in enumerate(s2orc_io.iter_jsonl(corpus)):
        if limit and index >= limit:
            break
        passage = str(record.get("passage") or "").strip()
        if len(passage) >= 80:
            texts.append(passage)
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S2ORC continued MLM on SciBERT")
    parser.add_argument("--corpus", type=Path, required=True, help="s2orc_passages.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="Cap training steps (0 = epoch-based)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--corpus-limit", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import torch
        from datasets import Dataset  # type: ignore[import-not-found]
        from transformers import (
            AutoTokenizer,
            BertForMaskedLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Install with: pip install torch transformers datasets")
        return 1

    limit = 128 if args.smoke_test else args.corpus_limit
    logger.info("Loading passages from %s (limit=%s)", args.corpus, limit or "all")
    texts = _load_passage_texts(args.corpus, limit=limit)
    if not texts:
        logger.error("No passage text loaded")
        return 1
    logger.info("Loaded %d passages", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = BertForMaskedLM.from_pretrained(args.base_model)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    dataset = Dataset.from_dict({"text": texts}).map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info(
        "Starting MLM: base=%s, passages=%d, max_length=%d",
        args.base_model,
        len(texts),
        args.max_length,
    )
    trainer.train()
    final_dir = args.output / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved MLM checkpoint to %s", final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
