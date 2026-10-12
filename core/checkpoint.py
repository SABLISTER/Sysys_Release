"""Shared checkpoint and sliding-window execution utilities.

CheckpointManager: single-file atomic checkpoint for long-running stages.
SlidingWindowExecutor: ordered concurrent execution with checkpoint integration.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

from core.io import atomic_write_json

logger = logging.getLogger(__name__)

DEFAULT_FUTURE_TIMEOUT = 120.0  # seconds before treating a worker as stuck


class CheckpointManager:
    """Single-file atomic checkpoint for list-accumulating stages.

    Combines data + progress into one JSON file to eliminate the race condition
    where two separate files (partial + progress) can become inconsistent if
    the process crashes between writes.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None,
        filename: str,
        checkpoint_every: int = 1,
    ):
        self._dir = checkpoint_dir
        self._filename = filename
        self._every = checkpoint_every

    @property
    def path(self) -> Path | None:
        return self._dir / self._filename if self._dir else None

    def load(self) -> tuple[list[dict], int]:
        """Load checkpoint and return (data, start_index).

        Returns ([], 0) if no checkpoint exists or it's corrupt.
        """
        if self._dir is None:
            return [], 0
        path = self._dir / self._filename
        if not path.exists():
            return [], 0
        try:
            with open(path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if isinstance(ckpt, list):
                data = ckpt
                count = len(data)
            elif isinstance(ckpt, dict) and "extracted" in ckpt:
                data = ckpt.get("extracted", [])
                count = ckpt.get("next_index", len(data))
            elif isinstance(ckpt, dict):
                data = ckpt.get("data", [])
                count = ckpt.get("processed_count", len(data))
            else:
                logger.warning(
                    "Unsupported checkpoint format in %s: %s",
                    path, type(ckpt).__name__,
                )
                return [], 0
            if not isinstance(data, list):
                logger.warning(
                    "Checkpoint data is not a list in %s: %s",
                    path, type(data).__name__,
                )
                return [], 0
            try:
                count = int(count)
            except (TypeError, ValueError):
                logger.warning(
                    "Checkpoint processed count is invalid in %s: %r. Using data length.",
                    path, count,
                )
                count = len(data)
            # Validate consistency: count must match data length
            if count != len(data):
                logger.warning(
                    "Checkpoint inconsistent: processed_count=%d but data has %d items. "
                    "Using data length.",
                    count, len(data),
                )
                count = len(data)
            logger.info("Resuming from checkpoint: %d already processed", count)
            return data, count
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
            logger.warning("Could not load checkpoint %s: %s", path, e)
            return [], 0

    def save(self, data: list[dict], total: int) -> bool:
        """Atomically save checkpoint if due (based on checkpoint_every)."""
        if self._dir is None or self._every <= 0:
            return False
        if len(data) % self._every != 0:
            return False
        return self._write(data, total)

    def force_save(self, data: list[dict], total: int) -> bool:
        """Atomically save checkpoint unconditionally."""
        if self._dir is None:
            return False
        return self._write(data, total)

    def cleanup(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self._dir is None:
            return
        path = self._dir / self._filename
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning("Could not remove checkpoint %s: %s", path, e)

    # Also clean up legacy two-file checkpoints from older runs
    def cleanup_legacy(self, *filenames: str) -> None:
        """Remove legacy checkpoint files (e.g. progress + partial from old format)."""
        if self._dir is None:
            return
        for name in filenames:
            p = self._dir / name
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    def _write(self, data: list[dict], total: int) -> bool:
        if self._dir is None:
            return False
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / self._filename
        try:
            atomic_write_json(
                path,
                {"processed_count": len(data), "total": total, "data": data},
                default=str,
            )
            return True
        except OSError as e:
            logger.warning("Checkpoint write failed for %s: %s", path, e)
            return False


def run_sliding_window(
    items: list,
    worker_fn: Callable,
    result_fn: Callable[[int, Any], dict],
    start_index: int = 0,
    workers: int = 2,
    stagger: float = 2.0,
    checkpoint_fn: Callable[[], None] | None = None,
    log_fn: Callable[[int, int], None] | None = None,
    cancel_event=None,
    future_timeout: float = DEFAULT_FUTURE_TIMEOUT,
) -> tuple[list[dict], bool]:
    """Execute worker_fn over items with a sliding window of concurrent futures.

    Args:
        items: Full list of items to process.
        worker_fn: Callable(item) -> result for each item.
        result_fn: Callable(index, worker_result) -> dict to build output record.
        start_index: Index to resume from (0-based).
        workers: Number of concurrent futures.
        stagger: Seconds between initial submissions.
        checkpoint_fn: Called after each result is collected.
        log_fn: Called with (completed_count, total_count) after each result.

    Returns:
        Tuple of (results, cancelled) where results is a list of dicts built by
        result_fn, in order.
    """
    results: list[dict] = []
    remaining = len(items) - start_index
    actual_workers = min(workers, remaining)
    cancelled = False

    if actual_workers <= 1:
        # Sequential fallback
        for i in range(start_index, len(items)):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            raw = worker_fn(items[i])
            results.append(result_fn(i, raw))
            if checkpoint_fn:
                checkpoint_fn()
            if log_fn:
                log_fn(len(results), len(items))
        return results, cancelled

    # Sliding window: keep `actual_workers` calls in flight, collect in order
    futures: dict[int, Future] = {}
    next_submit = start_index
    next_collect = start_index

    pool = ThreadPoolExecutor(max_workers=actual_workers)
    try:
        # Seed the window with staggered submissions
        while next_submit < len(items) and len(futures) < actual_workers:
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            if futures:
                time.sleep(stagger)
            futures[next_submit] = pool.submit(worker_fn, items[next_submit])
            next_submit += 1

        # Drain results in order, refilling as we go
        while next_collect < len(items) and next_collect in futures:
            deadline = time.monotonic() + future_timeout
            future = futures[next_collect]
            try:
                while True:
                    if cancel_event and cancel_event.is_set():
                        cancelled = True
                        raw = None
                        break
                    remaining_timeout = max(0.0, deadline - time.monotonic())
                    if remaining_timeout <= 0:
                        raise FutureTimeout
                    try:
                        raw = future.result(timeout=min(1.0, remaining_timeout))
                        break
                    except FutureTimeout:
                        # Distinguish our polling timeout from a worker function that
                        # itself raised TimeoutError (same exception class).
                        if future.done():
                            worker_exc = future.exception()
                            if worker_exc is None:
                                raw = future.result()
                                break
                            raise RuntimeError(
                                f"{type(worker_exc).__name__}: {worker_exc}"
                            ) from worker_exc
                        continue
                if cancelled:
                    break
            except FutureTimeout:
                logger.error(
                    "Worker timed out after %ds on item %d/%d. Skipping.",
                    future_timeout, next_collect, len(items),
                )
                raw = None
            except Exception as e:
                logger.error(
                    "Worker failed on item %d/%d. Skipping. Error: %s",
                    next_collect, len(items), e,
                )
                raw = None
            del futures[next_collect]

            results.append(result_fn(next_collect, raw))
            if checkpoint_fn:
                checkpoint_fn()
            if log_fn:
                log_fn(len(results), len(items))
            next_collect += 1

            if cancel_event and cancel_event.is_set():
                cancelled = True
                break

            # Submit next item to keep the window full
            if next_submit < len(items):
                futures[next_submit] = pool.submit(worker_fn, items[next_submit])
                next_submit += 1
    finally:
        pool.shutdown(wait=not cancelled, cancel_futures=cancelled)

    return results, cancelled
