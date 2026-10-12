"""Fine-tune a cross-encoder relevance scorer from the brt-bert MLM checkpoint.

This is the device-agnostic port of the original arch1 trainer.  It reads
pairwise JSONL triplets (query / positive / negative), converts them to binary
(sentence_pair, label) pairs, and fine-tunes via the sentence-transformers
``CrossEncoderTrainer`` with BCE loss.

The brt-bert MLM checkpoint (``BertForMaskedLM``) is loaded as a CrossEncoder:
the BERT encoder weights transfer and a fresh ``num_labels=1`` scoring head is
initialised on top.  The output directory is a standard HuggingFace model that
``sentence_transformers.CrossEncoder(path)`` can load directly — this becomes
the independent second rater for Stage 8 inter-rater reliability.

Hardware is auto-detected (CUDA / MPS / CPU); optimizer, precision, and
dataloader flags are chosen accordingly.  ``--precision auto`` (default) picks
bf16 on bf16-capable CUDA, fp16 on older CUDA, and fp32 on MPS/CPU.

Usage
-----
    python scripts/train_relevance_crossencoder.py \\
        --train /Volumes/Docker/arch1/pairwise_local.jsonl \\
                /Volumes/Docker/arch1/pairwise_longcat.jsonl \\
                /Volumes/Docker/arch1/pairwise_llm_q1.jsonl \\
        --epochs 3 --batch-size 16

Defaults point ``--checkpoint`` at brt-bert-irr-buddy-mlm and ``--output`` at
brt-bert-irr-buddy-mlm/relevance inside this repo.  If ``--eval`` is omitted,
10% of the shuffled data is held out automatically.

Requires: pip install datasets
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import re
import sys
from pathlib import Path
from random import Random

logger = logging.getLogger("train_relevance_crossencoder")

# Repo root = parent of this scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "brt-bert-irr-buddy-mlm"
DEFAULT_OUTPUT = REPO_ROOT / "brt-bert-irr-buddy-mlm" / "relevance"

QUERY_ALIASES = ("query", "q", "anchor", "prompt", "question")
POS_ALIASES = ("positive", "pos", "doc_pos", "positive_text", "passage_pos", "relevant")
NEG_ALIASES = ("negative", "neg", "doc_neg", "negative_text", "passage_neg", "irrelevant")


def _first_present(record: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_triplets(path: Path) -> list[tuple[str, str, str]]:
    """Read a JSONL file and return (query, positive, negative) triplets."""
    triplets: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            q = _first_present(obj, QUERY_ALIASES)
            p = _first_present(obj, POS_ALIASES)
            n = _first_present(obj, NEG_ALIASES)
            if q and p and n:
                triplets.append((q, p, n))
    return triplets


def triplets_to_pairs(triplets: list[tuple[str, str, str]]) -> list[dict]:
    """Convert triplets to binary pairs for BCE training.

    Each triplet produces two examples:
        (query, positive) -> label 1.0
        (query, negative) -> label 0.0
    """
    pairs = []
    for q, pos, neg in triplets:
        pairs.append({"sentence1": q, "sentence2": pos, "label": 1.0})
        pairs.append({"sentence1": q, "sentence2": neg, "label": 0.0})
    return pairs


def _natural_checkpoint_key(path: Path) -> tuple[int, str]:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (-1, path.name)


def resolve_base_model(checkpoint_root: Path) -> Path:
    """Choose latest checkpoint-* > final/ > root."""
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint root not found: {checkpoint_root}")
    ckpts = sorted(
        [p for p in checkpoint_root.glob("checkpoint-*") if p.is_dir()],
        key=_natural_checkpoint_key,
    )
    if ckpts:
        return ckpts[-1]
    final = checkpoint_root / "final"
    if final.is_dir():
        return final
    return checkpoint_root


def detect_device_config(requested_precision: str = "auto") -> dict:
    """Pick optimizer/precision/dataloader flags for the available hardware.

    Returns a dict consumed when building CrossEncoderTrainingArguments.
    fused AdamW and tf32 are CUDA-only; bf16/fp16 training is unstable on
    MPS/CPU so those fall back to fp32 under ``auto``.
    """
    import torch

    is_windows = platform.system().lower().startswith("win")

    if torch.cuda.is_available():
        bf16_ok = torch.cuda.is_bf16_supported()
        if requested_precision == "auto":
            precision = "bf16" if bf16_ok else "fp16"
        else:
            precision = requested_precision
        return {
            "device": "cuda",
            "precision": precision,
            "optim": "adamw_torch_fused",
            "tf32": True,
            "pin_memory": True,
            # Windows + CUDA: multiprocessing dataloaders are flaky, keep at 0.
            "dataloader_workers": 0 if is_windows else 2,
        }

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        precision = "fp32" if requested_precision == "auto" else requested_precision
        return {
            "device": "mps",
            "precision": precision,
            "optim": "adamw_torch",
            "tf32": False,
            "pin_memory": False,
            "dataloader_workers": 0,
        }

    # CPU
    precision = "fp32" if requested_precision == "auto" else requested_precision
    return {
        "device": "cpu",
        "precision": precision,
        "optim": "adamw_torch",
        "tf32": False,
        "pin_memory": False,
        "dataloader_workers": 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fine-tune brt-bert into a cross-encoder relevance scorer",
    )
    p.add_argument(
        "--train", nargs="+", type=Path, required=True,
        help="One or more JSONL triplet files for training (combined automatically)",
    )
    p.add_argument(
        "--eval", nargs="*", type=Path, default=None,
        help="Optional eval JSONL files. If omitted, 10%% of train data is held out.",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="MLM checkpoint root (resolves latest checkpoint-* automatically)",
    )
    p.add_argument("--base-model", type=str, default=None,
                   help="Override: exact HF model name/path (skips checkpoint resolution)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output directory for the trained relevance model")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-split", type=float, default=0.1,
                   help="Fraction held out for eval when --eval is not given")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="auto",
                   choices=("auto", "bf16", "fp16", "fp32"),
                   help="auto picks bf16/fp16 on CUDA, fp32 on MPS/CPU")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --output")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Imported here so --help works without the heavy deps installed.
    try:
        import torch  # noqa: F401  (used by detect_device_config)
        from datasets import Dataset
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder import (
            CrossEncoderTrainingArguments,
            CrossEncoderTrainer,
        )
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        from sentence_transformers.cross_encoder.evaluation import (
            CEBinaryClassificationEvaluator,
        )
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Install with: pip install datasets sentence-transformers")
        return 1

    # ── Device config ───────────────────────────────────────────────────
    dev = detect_device_config(args.precision)
    logger.info(
        "Device: %s | precision=%s | optim=%s | tf32=%s | pin_memory=%s | workers=%d",
        dev["device"], dev["precision"], dev["optim"],
        dev["tf32"], dev["pin_memory"], dev["dataloader_workers"],
    )

    # ── Resolve model path ──────────────────────────────────────────────
    if args.base_model:
        model_path = args.base_model
    else:
        model_path = str(resolve_base_model(args.checkpoint))
    logger.info("Base model: %s", model_path)

    # ── Load triplets ───────────────────────────────────────────────────
    all_triplets: list[tuple[str, str, str]] = []
    for path in args.train:
        logger.info("Loading triplets from %s ...", path)
        rows = load_triplets(path)
        logger.info("  -> %d triplets", len(rows))
        all_triplets.extend(rows)

    if not all_triplets:
        logger.error("No triplets loaded from any input file.")
        return 1

    eval_triplets: list[tuple[str, str, str]] = []
    if args.eval:
        for path in args.eval:
            logger.info("Loading eval triplets from %s ...", path)
            rows = load_triplets(path)
            logger.info("  -> %d triplets", len(rows))
            eval_triplets.extend(rows)
    else:
        rng = Random(args.seed)
        rng.shuffle(all_triplets)
        split_idx = max(1, int(len(all_triplets) * (1 - args.eval_split)))
        eval_triplets = all_triplets[split_idx:]
        all_triplets = all_triplets[:split_idx]
        logger.info(
            "Auto-split: %d train triplets, %d eval triplets (%.0f%% held out)",
            len(all_triplets), len(eval_triplets), args.eval_split * 100,
        )

    # ── Convert to binary pairs ─────────────────────────────────────────
    train_pairs = triplets_to_pairs(all_triplets)
    eval_pairs = triplets_to_pairs(eval_triplets)

    rng = Random(args.seed)
    rng.shuffle(train_pairs)
    rng.shuffle(eval_pairs)

    logger.info("Binary pairs: train=%d, eval=%d", len(train_pairs), len(eval_pairs))

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs)

    # ── Init model ──────────────────────────────────────────────────────
    model = CrossEncoder(
        model_path,
        num_labels=1,
        max_length=args.max_length,
    )

    n_params = sum(p_.numel() for p_ in model.parameters())
    logger.info("Model: %.1fM params, max_length=%d", n_params / 1e6, args.max_length)

    # ── Training args ───────────────────────────────────────────────────
    use_bf16 = dev["precision"] == "bf16"
    use_fp16 = dev["precision"] == "fp16"

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        optim=dev["optim"],
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=dev["tf32"],
        logging_steps=args.log_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        dataloader_num_workers=dev["dataloader_workers"],
        dataloader_pin_memory=dev["pin_memory"],
        seed=args.seed,
    )

    # ── Loss ────────────────────────────────────────────────────────────
    loss = BinaryCrossEntropyLoss(model)

    # ── Evaluator ───────────────────────────────────────────────────────
    evaluator = CEBinaryClassificationEvaluator(
        sentence_pairs=[[p_["sentence1"], p_["sentence2"]] for p_ in eval_pairs],
        labels=[int(p_["label"]) for p_ in eval_pairs],
        name="relevance",
        write_csv=True,
    )

    # ── Resume ──────────────────────────────────────────────────────────
    resume_from = None
    if args.resume:
        existing = sorted(
            args.output.glob("checkpoint-*"),
            key=_natural_checkpoint_key,
        )
        if existing:
            resume_from = str(existing[-1])
            logger.info("Resuming from %s", resume_from)

    # ── Train ───────────────────────────────────────────────────────────
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
        evaluator=evaluator,
    )

    steps_total = math.ceil(
        len(train_ds) / max(1, args.batch_size * args.grad_accum)
    ) * args.epochs
    logger.info(
        "Starting training: %d steps, effective_batch=%d, lr=%.1e, epochs=%d",
        steps_total, args.batch_size * args.grad_accum, args.lr, args.epochs,
    )

    result = trainer.train(resume_from_checkpoint=resume_from)
    logger.info("Training complete: %s", result.metrics)

    # ── Save final model ────────────────────────────────────────────────
    final_dir = args.output / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved final model to %s", final_dir)

    # ── Final evaluation ────────────────────────────────────────────────
    eval_metrics = trainer.evaluate()
    logger.info("Final eval: %s", eval_metrics)

    evaluator_results = evaluator(model)
    logger.info("Binary classification metrics: %s", evaluator_results)

    metrics_path = args.output / "final_metrics.json"
    all_metrics = {**eval_metrics, **evaluator_results}
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)
    logger.info("Wrote metrics to %s", metrics_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
