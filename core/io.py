"""Shared atomic file I/O utilities for the SysS pipeline.

All writers follow the same pattern: write to a temp file in the same
directory, then atomically rename into place.  A crash between the write and
the rename leaves a .tmp file behind but never corrupts the target.

JSON Lines appends use standard file append with fsync for O(1) per-record
I/O; a crash may leave a partial final line.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class _JsonlWriter:
    """Append-only JSON Lines writer.

    Each record is appended in O(1) I/O. A crash mid-write may leave a
    partial final line; readers should skip malformed trailing lines.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = _target_file_mode(path)

    def append(self, record: Any) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        separator = ""
        try:
            if self.path.stat().st_size:
                with open(self.path, "rb") as existing:
                    existing.seek(-1, os.SEEK_END)
                    if existing.read(1) != b"\n":
                        # Isolate a crash-truncated final record so the new
                        # complete append remains independently readable.
                        separator = "\n"
        except OSError:
            pass
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(separator + line)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.path, self._mode)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _default_file_mode() -> int:
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def _target_file_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return _default_file_mode()


def _replace_preserving_mode(tmp_path: str, path: Path, mode: int) -> None:
    os.chmod(tmp_path, mode)
    Path(tmp_path).replace(path)


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    default=str,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON atomically via a temp file — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=ensure_ascii, default=default)
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_write_json_array(
    path: Path,
    records: Iterable[Any],
    *,
    default=str,
    ensure_ascii: bool = False,
) -> int:
    """Stream an iterable of records into an atomic JSON array.

    Use this instead of ``atomic_write_json(path, list(...))`` when the caller
    can produce records lazily and each one is large — building the list first
    holds every record in memory at once. Returns the number of records
    written.

    The array is still written atomically: a crash mid-stream leaves the
    previous file untouched and only a stray .tmp behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    count = 0
    try:
        with open(tmp_fd, "w", encoding="utf-8") as out:
            out.write("[\n")
            for record in records:
                if count:
                    out.write(",\n")
                out.write(
                    "  " + json.dumps(record, ensure_ascii=ensure_ascii,
                                      default=default)
                )
                count += 1
            out.write("\n]\n" if count else "]\n")
        _replace_preserving_mode(tmp_path, path, mode)
        return count
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via a temp file — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically via a temp file — safe against mid-write crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "wb") as f:
            f.write(data)
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_append_jsonl(path: Path, record: Any) -> None:
    """Append a single JSON object to a JSON Lines file atomically."""
    writer = _JsonlWriter(path)
    writer.append(record)


def load_json(path: Path, default: Any = None) -> Any:
    """Load a JSON file, returning *default* on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def iter_jsonl_records(path: Path) -> Iterator[Any]:
    """Yield valid JSON Lines records, ignoring a crash-truncated line."""
    if not path.exists():
        return
    with open(path, "rb") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # JSONL is append-only and fsynced per record, but a process
                # crash can still leave the last write incomplete.
                continue


def jsonl_to_json_array(
    jsonl_path: Path,
    json_path: Path,
    *,
    record_key: Callable[[Any], str] | None = None,
    include_keys: set[str] | None = None,
) -> None:
    """Stream valid JSON Lines records into an atomic JSON array.

    When ``record_key`` is supplied, only the latest record for each key is
    materialized. ``include_keys`` can further restrict the final array to the
    records relevant to the current run while leaving the append-only JSONL
    checkpoint intact for later resume attempts.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(json_path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(json_path.parent), suffix=".tmp", prefix=json_path.stem
    )
    try:
        last_index: dict[str, int] = {}
        if record_key is not None:
            for index, record in enumerate(iter_jsonl_records(jsonl_path)):
                key = str(record_key(record) or "")
                if key and (include_keys is None or key in include_keys):
                    last_index[key] = index

        with open(tmp_fd, "w", encoding="utf-8") as out:
            out.write("[\n")
            first = True
            for index, record in enumerate(iter_jsonl_records(jsonl_path)):
                if record_key is not None:
                    key = str(record_key(record) or "")
                    if not key or (include_keys is not None and key not in include_keys):
                        continue
                    if last_index.get(key) != index:
                        continue
                if not first:
                    out.write(",\n")
                first = False
                out.write(
                    "  " + json.dumps(record, ensure_ascii=False, default=str)
                )
            out.write("\n]\n")
        _replace_preserving_mode(tmp_path, json_path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
