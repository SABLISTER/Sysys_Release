#!/usr/bin/env python3
"""Train DR-BRT-BERT, a dense retriever distilled from BRT-BERT.

The script is designed to be copied into ``C:\\systs\\scripts`` and run on
the Windows training machine.  Its defaults match the original data layout:

* ``C:/home/arch1/worker1/pairwise_local.jsonl``
* ``C:/home/arch1/worker2/pairwise_longcat.jsonl``
* ``C:/home/arch1/worker3/pairwise_llm_q1.jsonl`` (optional)
* ``C:/GLHF_bert/relevance/train`` and ``.../eval`` (Arrow fallback)
* ``C:/systs/brt-bert-relevance/final`` (BRT-BERT teacher/backbone)
* ``C:/GLHF_bert/dr-brt-bert`` (output)

DR-BRT-BERT is a shared-weight bi-encoder.  Questions and papers are encoded
separately, mean pooled, L2-normalized, and trained with:

1. in-batch contrastive retrieval loss;
2. an explicit hard-negative pairwise loss; and
3. optional BRT-BERT margin distillation (enabled by default).

The final directory is a standard Sentence-Transformers model that SystS can
load with ``SentenceTransformer(path)``.  Evaluation is split by complete
research question, never by individual pair, to prevent query leakage.

Install on the Windows/CUDA machine::

    py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    py -m pip install sentence-transformers transformers datasets pyarrow numpy

Smoke test::

    py C:\\systs\\scripts\\train_dr_brt_bert.py --smoke-test

Full run::

    py C:\\systs\\scripts\\train_dr_brt_bert.py \
        --epochs 3 --batch-size 32 --grad-accum 2 --precision bf16

Use ``--data-source arrow`` to train from the prepared Arrow splits instead
of the pairwise JSONL files.  Use ``--teacher-weight 0`` to disable online
teacher-score precomputation while retaining BRT-BERT backbone initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger("train_dr_brt_bert")

DEFAULT_JSONLS = (
    Path("C:/home/arch1/worker1/pairwise_local.jsonl"),
    Path("C:/home/arch1/worker2/pairwise_longcat.jsonl"),
    Path("C:/home/arch1/worker3/pairwise_llm_q1.jsonl"),
)
DEFAULT_TRAIN_ARROW = Path("C:/GLHF_bert/relevance/train")
DEFAULT_EVAL_ARROW = Path("C:/GLHF_bert/relevance/eval")
DEFAULT_BASE_MODEL = Path("C:/systs/brt-bert-relevance/final")
DEFAULT_OUTPUT = Path("C:/GLHF_bert/dr-brt-bert")

QUERY_COLUMNS = ("sentence1", "query", "question", "text1", "s1")
DOCUMENT_COLUMNS = ("sentence2", "passage", "document", "text2", "s2")
LABEL_COLUMNS = ("label", "labels", "score", "relevance", "target")


@dataclass(frozen=True)
class Triplet:
    query: str
    positive: str
    negative: str
    source: str = ""


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _triplet_key(row: Triplet) -> str:
    payload = "\x1f".join((row.query, row.positive, row.negative))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_fingerprint(rows: Sequence[Triplet]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_triplet_key(row).encode("ascii"))
    return digest.hexdigest()


def load_pairwise_jsonls(
    paths: Iterable[Path],
    *,
    limit: int | None = None,
) -> list[Triplet]:
    """Load and deduplicate query/positive/negative JSONL records."""
    rows: list[Triplet] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            logger.info("Optional JSONL not found, skipping: %s", path)
            continue
        logger.info("Loading pairwise JSONL: %s", path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if limit is not None and len(rows) >= limit:
                    return rows
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON in {path}:{line_number}: {exc}") from exc
                row = Triplet(
                    query=_clean(record.get("query")),
                    positive=_clean(record.get("positive")),
                    negative=_clean(record.get("negative")),
                    source=str(path),
                )
                if not (row.query and row.positive and row.negative):
                    continue
                key = _triplet_key(row)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def _find_arrow_files(path: Path, shard_limit: int | None = None) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".arrow":
            raise ValueError(f"Expected an Arrow file, got: {path}")
        return [path]
    files = sorted(path.glob("*.arrow")) if path.exists() else []
    if shard_limit is not None and shard_limit > 0:
        files = files[:shard_limit]
    return files


def _read_arrow_table(path: Path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            pass
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all()


def _pick_column(columns: Sequence[str], choices: Sequence[str], role: str) -> str:
    for choice in choices:
        if choice in columns:
            return choice
    raise ValueError(
        f"No {role} column found. Tried {list(choices)}; available={list(columns)}"
    )


def load_arrow_triplets(
    path: Path,
    *,
    shard_limit: int | None = None,
    row_limit: int | None = None,
) -> list[Triplet]:
    """Convert labeled Arrow pairs into one hard-negative triplet per positive."""
    files = _find_arrow_files(path, shard_limit)
    if not files:
        raise FileNotFoundError(f"No .arrow shards found in {path}")

    grouped: dict[str, dict[str, list[str]]] = {}
    loaded = 0
    for file_path in files:
        logger.info("Loading Arrow shard: %s", file_path)
        table = _read_arrow_table(file_path)
        columns = table.column_names
        q_col = _pick_column(columns, QUERY_COLUMNS, "query")
        d_col = _pick_column(columns, DOCUMENT_COLUMNS, "document")
        label_col = _pick_column(columns, LABEL_COLUMNS, "label")
        for record in table.select([q_col, d_col, label_col]).to_pylist():
            if row_limit is not None and loaded >= row_limit:
                break
            loaded += 1
            query = _clean(record.get(q_col))
            document = _clean(record.get(d_col))
            if not query or not document:
                continue
            try:
                positive = float(record.get(label_col) or 0) >= 0.5
            except (TypeError, ValueError):
                positive = False
            bucket = grouped.setdefault(query, {"positive": [], "negative": []})
            bucket["positive" if positive else "negative"].append(document)
        if row_limit is not None and loaded >= row_limit:
            break

    rows: list[Triplet] = []
    seen: set[str] = set()
    for query, bucket in grouped.items():
        positives = list(dict.fromkeys(bucket["positive"]))
        negatives = list(dict.fromkeys(bucket["negative"]))
        if not positives or not negatives:
            continue
        for index, positive in enumerate(positives):
            row = Triplet(
                query=query,
                positive=positive,
                negative=negatives[index % len(negatives)],
                source=str(path),
            )
            key = _triplet_key(row)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def split_by_query(
    rows: Sequence[Triplet],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[Triplet], list[Triplet]]:
    """Deterministically split complete query groups into train and eval."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    queries = sorted({row.query for row in rows})
    if len(queries) < 2:
        raise ValueError("At least two distinct research questions are required")
    rng = random.Random(seed)
    rng.shuffle(queries)
    eval_count = max(1, min(len(queries) - 1, round(len(queries) * eval_fraction)))
    eval_queries = set(queries[:eval_count])
    train = [row for row in rows if row.query not in eval_queries]
    evaluation = [row for row in rows if row.query in eval_queries]
    return train, evaluation


def _deduplicate(rows: Iterable[Triplet]) -> list[Triplet]:
    result: list[Triplet] = []
    seen: set[str] = set()
    for row in rows:
        key = _triplet_key(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def load_training_data(args: argparse.Namespace) -> tuple[list[Triplet], list[Triplet], str]:
    jsonl_paths = [Path(path) for path in (args.pairwise_jsonl or DEFAULT_JSONLS)]
    jsonl_paths_present = [path for path in jsonl_paths if path.exists()]
    source = args.data_source
    if source == "auto":
        source = "jsonl" if len(jsonl_paths_present) >= 2 else "arrow"

    if source == "jsonl":
        if not jsonl_paths_present:
            raise FileNotFoundError("No pairwise JSONL files were found")
        rows = load_pairwise_jsonls(jsonl_paths_present, limit=args.record_limit)
        train, evaluation = split_by_query(
            rows, eval_fraction=args.eval_fraction, seed=args.seed
        )
        return train, evaluation, "pairwise_jsonl"

    train_rows = load_arrow_triplets(
        Path(args.train_arrow),
        shard_limit=args.shard_limit,
        row_limit=args.record_limit,
    )
    eval_rows = load_arrow_triplets(
        Path(args.eval_arrow),
        shard_limit=args.shard_limit,
        row_limit=args.record_limit,
    )
    overlap = {row.query for row in train_rows} & {row.query for row in eval_rows}
    if overlap:
        logger.warning(
            "Arrow splits share %d queries; regrouping and splitting by query", len(overlap)
        )
        combined = _deduplicate([*train_rows, *eval_rows])
        train_rows, eval_rows = split_by_query(
            combined, eval_fraction=args.eval_fraction, seed=args.seed
        )
    return train_rows, eval_rows, "arrow"


def _resolve_base_model(path: Path) -> Path:
    candidates = [path, Path("brt-bert-relevance/final"), Path("brt-bert-relevance")]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "BRT-BERT checkpoint not found. Pass --base-model, for example "
        "C:/systs/brt-bert-relevance/final"
    )


def _device_name(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(base_model: Path, max_length: int, device: str):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

    transformer = Transformer(str(base_model), max_seq_length=max_length)
    pooling = Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode="mean",
    )
    return SentenceTransformer(modules=[transformer, pooling], device=device)


def _encode_for_training(model, texts: Sequence[str], device: str):
    features = model.tokenize(list(texts))
    features = {key: value.to(device) for key, value in features.items()}
    return model(features)["sentence_embedding"]


def _identity_collate(batch):
    return batch


def _teacher_cache_paths(output: Path) -> tuple[Path, Path]:
    return output / "teacher_margins.npy", output / "teacher_margins.json"


def teacher_margins(
    rows: Sequence[Triplet],
    *,
    teacher_path: Path,
    device: str,
    batch_size: int,
    output: Path,
) -> list[float]:
    """Precompute and cache teacher positive-minus-negative score margins."""
    import numpy as np

    score_path, meta_path = _teacher_cache_paths(output)
    fingerprint = dataset_fingerprint(rows)
    if score_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        cached = np.load(score_path)
        if metadata.get("fingerprint") == fingerprint and len(cached) == len(rows):
            logger.info("Using cached BRT-BERT teacher margins: %s", score_path)
            return cached.astype("float32").tolist()

    from sentence_transformers import CrossEncoder

    logger.info("Precomputing BRT-BERT teacher scores for %d triplets", len(rows))
    teacher = CrossEncoder(str(teacher_path), device=device)
    margins: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        positive_pairs = [(row.query, row.positive) for row in batch]
        negative_pairs = [(row.query, row.negative) for row in batch]
        positive = np.asarray(
            teacher.predict(positive_pairs, show_progress_bar=False), dtype="float32"
        ).reshape(-1)
        negative = np.asarray(
            teacher.predict(negative_pairs, show_progress_bar=False), dtype="float32"
        ).reshape(-1)
        margins.extend((positive - negative).tolist())
        if start == 0 or (start // batch_size) % 100 == 0:
            logger.info("Teacher scoring: %d/%d", min(start + len(batch), len(rows)), len(rows))

    output.mkdir(parents=True, exist_ok=True)
    np.save(score_path, np.asarray(margins, dtype="float32"))
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "rows": len(rows),
                "teacher": str(teacher_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del teacher
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    return margins


def evaluate(model, rows: Sequence[Triplet], *, device: str, batch_size: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    if not rows:
        return {}

    def encode_all(texts: Sequence[str]):
        chunks = []
        for start in range(0, len(texts), batch_size):
            embedding = functional.normalize(
                _encode_for_training(model, texts[start : start + batch_size], device),
                dim=1,
            )
            # Keep the complete evaluation matrices off the GPU and use
            # float32 for stable ranking calculations.
            chunks.append(embedding.detach().float().cpu())
        return torch.cat(chunks, dim=0)

    model.eval()
    with torch.no_grad():
        query = encode_all([row.query for row in rows])
        positive = encode_all([row.positive for row in rows])
        negative = encode_all([row.negative for row in rows])

    ranks: list[int] = []
    # Rank each query against every held-out positive. Chunking prevents a
    # full N-by-N score matrix while retaining corpus-level Recall@K.
    for start in range(0, len(rows), batch_size):
        query_chunk = query[start : start + batch_size]
        scores = query_chunk @ positive.T
        for local_index in range(len(query_chunk)):
            global_index = start + local_index
            target = float(scores[local_index, global_index])
            ranks.append(1 + int((scores[local_index] > target).sum().item()))

    pos_sim = (query * positive).sum(dim=1)
    neg_sim = (query * negative).sum(dim=1)
    hard_correct = int((pos_sim > neg_sim).sum().item())
    margins = (pos_sim - neg_sim).tolist()

    total = len(ranks)
    return {
        "examples": total,
        "recall_at_1": sum(rank <= 1 for rank in ranks) / total,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / total,
        "mrr": sum(1.0 / rank for rank in ranks) / total,
        "hard_negative_accuracy": hard_correct / total,
        "mean_positive_negative_margin": sum(margins) / len(margins),
    }


def _latest_checkpoint(output: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    for path in (output / "checkpoints").glob("epoch-*"):
        match = re.fullmatch(r"epoch-(\d+)", path.name)
        if match and (path / "config_sentence_transformers.json").exists():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=None)


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from sentence_transformers import SentenceTransformer
        from torch.utils.data import DataLoader, Dataset
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(
            f"Missing training dependency: {exc}. Install sentence-transformers, "
            "transformers, datasets, pyarrow, numpy, and a CUDA build of torch."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.smoke_test and args.record_limit is None:
        args.record_limit = 512
    output = Path(args.output)
    if args.smoke_test:
        output = output / "smoke-test"
    output.mkdir(parents=True, exist_ok=True)
    base_model = _resolve_base_model(Path(args.base_model))
    device = _device_name(args.device)
    train_rows, eval_rows, data_source = load_training_data(args)
    if args.smoke_test:
        train_rows = train_rows[: min(128, len(train_rows))]
        eval_rows = eval_rows[: min(64, len(eval_rows))]
        args.epochs = 1
        args.teacher_weight = 0.0
        args.save_every = 1
    if not train_rows or not eval_rows:
        raise RuntimeError(
            f"Insufficient data after loading: train={len(train_rows)}, eval={len(eval_rows)}"
        )
    train_queries = {row.query for row in train_rows}
    eval_queries = {row.query for row in eval_rows}
    if train_queries & eval_queries:
        raise RuntimeError("Query leakage detected between train and evaluation sets")

    logger.info(
        "Data source=%s train=%d (%d queries) eval=%d (%d queries)",
        data_source,
        len(train_rows),
        len(train_queries),
        len(eval_rows),
        len(eval_queries),
    )
    logger.info("Backbone/teacher=%s device=%s output=%s", base_model, device, output)

    start_epoch = 1
    checkpoint = _latest_checkpoint(output) if args.resume else None
    if checkpoint:
        completed_epoch, checkpoint_path = checkpoint
        logger.info("Resuming model from %s", checkpoint_path)
        model = SentenceTransformer(str(checkpoint_path), device=device)
        start_epoch = completed_epoch + 1
    else:
        model = _build_model(base_model, args.max_length, device)

    margins: list[float] | None = None
    if args.teacher_weight > 0:
        margins = teacher_margins(
            train_rows,
            teacher_path=base_model,
            device=device,
            batch_size=args.teacher_batch_size,
            output=output,
        )
    training_items = list(zip(train_rows, margins or [0.0] * len(train_rows)))

    class _TrainingItemDataset(Dataset):
        """Wraps the list so DataLoader sees a real `Dataset[_T_co]`.

        A plain list works at runtime but doesn't satisfy DataLoader's
        typed `Dataset` parameter.
        """

        def __init__(self, items: list[tuple[Triplet, float]]):
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int) -> tuple[Triplet, float]:
            return self._items[index]

    loader = DataLoader(
        _TrainingItemDataset(training_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_identity_collate,
        pin_memory=device == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )

    optimizer_state = output / "optimizer_state.pt"
    if checkpoint and optimizer_state.exists():
        state = torch.load(optimizer_state, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    use_cuda_amp = device == "cuda" and args.precision in {"fp16", "bf16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    from torch.amp.grad_scaler import GradScaler

    scaler = GradScaler("cuda", enabled=use_cuda_amp and args.precision == "fp16")
    history: list[dict[str, Any]] = []
    log_path = output / "training_log.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader, 1):
            rows = [item[0] for item in batch]
            teacher_target = torch.tensor(
                [item[1] for item in batch], dtype=torch.float32, device=device
            )
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_cuda_amp,
            ):
                query = functional.normalize(
                    _encode_for_training(model, [row.query for row in rows], device), dim=1
                )
                positive = functional.normalize(
                    _encode_for_training(model, [row.positive for row in rows], device), dim=1
                )
                negative = functional.normalize(
                    _encode_for_training(model, [row.negative for row in rows], device), dim=1
                )
                targets = torch.arange(len(rows), device=device)
                logits = (query @ positive.T) / args.temperature
                in_batch = 0.5 * (
                    functional.cross_entropy(logits, targets)
                    + functional.cross_entropy(logits.T, targets)
                )
                pos_sim = (query * positive).sum(dim=1)
                neg_sim = (query * negative).sum(dim=1)
                hard_logits = torch.stack((pos_sim, neg_sim), dim=1) / args.temperature
                hard_loss = functional.cross_entropy(
                    hard_logits, torch.zeros(len(rows), dtype=torch.long, device=device)
                )
                distill_loss = functional.mse_loss(
                    pos_sim - neg_sim, teacher_target
                )
                loss = (
                    in_batch
                    + args.hard_negative_weight * hard_loss
                    + args.teacher_weight * distill_loss
                ) / args.grad_accum

            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(loader):
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * args.grad_accum
            if step == 1 or step % args.log_steps == 0:
                logger.info(
                    "epoch=%d step=%d/%d loss=%.5f lr=%.3g",
                    epoch,
                    step,
                    len(loader),
                    running / step,
                    scheduler.get_last_lr()[0],
                )

        evaluation = evaluate(
            model,
            eval_rows[: args.eval_limit] if args.eval_limit else eval_rows,
            device=device,
            batch_size=args.eval_batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(loader)),
            "learning_rate": scheduler.get_last_lr()[0],
            "evaluation": evaluation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        logger.info("Epoch %d evaluation: %s", epoch, json.dumps(evaluation, sort_keys=True))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = output / "checkpoints" / f"epoch-{epoch}"
            model.save(str(checkpoint_path))
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                optimizer_state,
            )

    final_path = output / "final"
    model.save(str(final_path))
    final_eval_rows = eval_rows[: args.eval_limit] if args.eval_limit else eval_rows
    final_metrics = evaluate(
        model,
        final_eval_rows,
        device=device,
        batch_size=args.eval_batch_size,
    )
    manifest = {
        "model_name": "DR-BRT-BERT",
        "model_type": "sentence-transformers-bi-encoder",
        "base_model": str(base_model),
        "data_source": data_source,
        "train_triplets": len(train_rows),
        "eval_triplets": len(eval_rows),
        "train_queries": len(train_queries),
        "eval_queries": len(eval_queries),
        "query_overlap": 0,
        "train_fingerprint": dataset_fingerprint(train_rows),
        "final_path": str(final_path),
        "parameters": _json_safe(vars(args)),
        "final_metrics": final_metrics,
        "history": history,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Saved DR-BRT-BERT to %s", final_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source", choices=("auto", "jsonl", "arrow"), default="auto"
    )
    parser.add_argument("--pairwise-jsonl", nargs="*", type=Path)
    parser.add_argument("--train-arrow", type=Path, default=DEFAULT_TRAIN_ARROW)
    parser.add_argument("--eval-arrow", type=Path, default=DEFAULT_EVAL_ARROW)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record-limit", type=int)
    parser.add_argument("--shard-limit", type=int)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--eval-limit", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for in-batch negatives")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
