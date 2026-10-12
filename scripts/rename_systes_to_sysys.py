#!/usr/bin/env python3
"""Rename leftover Systes → Sysys across comments, code, and documents.

A naive case-insensitive replace of Systes with Sysys breaks Python identifiers:

    systes_nl_query      -> Sysys_nl_query      # wrong
    _start_systes_search -> _start_Sysys_search # wrong

This script keeps snake_case identifiers lowercase (sysys_*) while using Sysys
as the product name in prose, comments, YAML headers, and QSettings org strings.

By default this is a dry run. Pass --apply to write files.

Usage
-----
    python scripts/rename_systes_to_sysys.py
    python scripts/rename_systes_to_sysys.py --apply
    python scripts/rename_systes_to_sysys.py --apply --fix-pascal-identifiers
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

OLD = "systes"
NEW = "sysys"

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".temp_ip",
    ".specstory",
    ".cursor",
    "dist",
    "build",
    ".tox",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".xz",
    ".7z",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}

TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".cfg",
    ".ini",
    ".csv",
    ".tsv",
    ".xml",
}

IDENT_CHAR = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def is_ident_char(ch: str | None) -> bool:
    return bool(ch) and ch in IDENT_CHAR


def mapped_product_name(matched: str) -> str:
    """Preserve obvious casings; fall back to title case for mixed junk."""
    if matched.isupper():
        return NEW.upper()
    if matched.islower():
        return NEW.lower()
    return NEW.capitalize()


def rewrite_text(text: str, *, fix_pascal_identifiers: bool) -> str:
    """Replace Systes with Sysys, forcing lowercase inside identifiers."""
    lower = text.lower()
    out: list[str] = []
    i = 0
    n = len(text)
    old_len = len(OLD)

    while i < n:
        if lower.startswith(OLD, i):
            prev_ch = text[i - 1] if i > 0 else None
            next_ch = text[i + old_len] if i + old_len < n else None
            in_identifier = is_ident_char(prev_ch) or is_ident_char(next_ch)
            matched = text[i : i + old_len]
            if in_identifier:
                out.append(NEW.lower())
            else:
                out.append(mapped_product_name(matched))
            i += old_len
            continue
        out.append(text[i])
        i += 1

    rewritten = "".join(out)
    if fix_pascal_identifiers:
        rewritten = lowercase_pascal_sysys_identifiers(rewritten)
    return rewritten


def lowercase_pascal_sysys_identifiers(text: str) -> str:
    """Turn leftover Pascal-case identifier fragments into snake_case.

    A previous naive replace left names like Sysys_nl_query / _start_Sysys_search.
    Standalone product-name uses of Sysys (followed by space or punctuation) are
    left alone.
    """
    needle = NEW.capitalize()  # Sysys
    out: list[str] = []
    i = 0
    n = len(text)
    needle_len = len(needle)

    while i < n:
        if text.startswith(needle, i):
            prev_ch = text[i - 1] if i > 0 else None
            next_ch = text[i + needle_len] if i + needle_len < n else None
            if is_ident_char(next_ch) or (is_ident_char(prev_ch) and is_ident_char(next_ch)):
                out.append(NEW.lower())
                i += needle_len
                continue
            if is_ident_char(prev_ch) and next_ch == "_":
                out.append(NEW.lower())
                i += needle_len
                continue
            if prev_ch in {".", "_"} and is_ident_char(next_ch):
                out.append(NEW.lower())
                i += needle_len
                continue
            if prev_ch in {".", "_"} and next_ch == "_":
                # self.Sysys_nl_query / _Sysys_search_runs
                out.append(NEW.lower())
                i += needle_len
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    script_name = Path(__file__).name
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not should_skip_dir(name)]
        for name in filenames:
            if name == script_name:
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or path.suffix == "":
                files.append(path)
    return sorted(files)


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"skip unreadable {path}: {exc}", file=sys.stderr)
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None


def python_still_parses(path: Path, text: str) -> str | None:
    if path.suffix != ".py":
        return None
    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return f"{path}:{exc.lineno}: {exc.msg}"
    return None


def process(root: Path, *, apply: bool, fix_pascal_identifiers: bool) -> int:
    changed = 0
    parse_errors: list[str] = []
    for path in iter_files(root):
        original = read_text(path)
        if original is None:
            continue
        updated = rewrite_text(original, fix_pascal_identifiers=fix_pascal_identifiers)
        if updated == original:
            continue
        err = python_still_parses(path, updated)
        if err:
            parse_errors.append(err)
            print(f"REFUSE (would break syntax): {path}")
            continue
        rel = path.relative_to(root)
        print(f"{'WRITE' if apply else 'DRY  '} {rel}")
        if apply:
            path.write_bytes(updated.encode("utf-8"))
        changed += 1

    print(f"{changed} file(s) {'updated' if apply else 'would change'}")
    if parse_errors:
        print("Python parse errors; those files were left unchanged:", file=sys.stderr)
        for err in parse_errors:
            print(f"  {err}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rename Systes → Sysys without breaking snake_case identifiers.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Default is a dry run.",
    )
    parser.add_argument(
        "--fix-pascal-identifiers",
        action="store_true",
        help=(
            "Also rewrite leftover Pascal-case identifier fragments such as "
            "Sysys_nl_query / _start_Sysys_search to sysys_nl_query / "
            "_start_sysys_search. Use this after a naive replace already "
            "broke identifier casing in gui/main_window.py."
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"root is not a directory: {root}", file=sys.stderr)
        return 2
    return process(
        root,
        apply=args.apply,
        fix_pascal_identifiers=args.fix_pascal_identifiers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
