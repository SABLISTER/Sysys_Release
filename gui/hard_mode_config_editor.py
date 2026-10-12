"""In-app editor for hard_mode_config.yaml."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Callable

from hard_mode.config_text import validate_hard_mode_yaml_text, write_text_atomically

logger = logging.getLogger(__name__)


class HardModeConfigEditor(tk.Toplevel):
    """Modal editor for hard_mode_config.yaml text."""

    def __init__(
        self,
        parent,
        *,
        config_path: Path,
        on_saved: Callable[[], None] | None = None,
        on_open_settings: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.title("Hard Mode Config Editor")
        self.transient(parent)
        self.grab_set()
        self.geometry("980x760")
        self.minsize(760, 520)

        self._config_path = Path(config_path)
        self._on_saved = on_saved
        self._on_open_settings = on_open_settings
        self._loaded_text = ""
        self.status_var = tk.StringVar(value="")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._reload_from_disk()

        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, px)}+{max(0, py)}")

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Config file:").grid(row=0, column=0, sticky="nw")
        self.path_lbl = ttk.Label(top, text=str(self._config_path), wraplength=760, justify="left")
        self.path_lbl.grid(row=0, column=1, sticky="nw")

        note = (
            "Edit hard-mode YAML here. Provider/API/LLM environment keys still live in .env and "
            "can be changed from Pipeline Settings."
        )
        ttk.Label(top, text=note, wraplength=860, foreground="gray").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 8)
        )

        btn_row = ttk.Frame(top)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Button(btn_row, text="Reload from disk", command=self._reload_from_disk).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Validate YAML", command=self._validate).pack(side="left", padx=(0, 6))
        if self._on_open_settings is not None:
            ttk.Button(btn_row, text="Pipeline Settings…", command=self._on_open_settings).pack(side="left")

        self.editor = scrolledtext.ScrolledText(
            top,
            wrap="none",
            undo=True,
            font=("Menlo", 10),
        )
        self.editor.grid(row=3, column=0, columnspan=2, sticky="nsew")

        status_lbl = ttk.Label(top, textvariable=self.status_var, foreground="gray")
        status_lbl.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        bottom = ttk.Frame(top)
        bottom.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(bottom, text="Close", command=self._on_close).pack(side="right")
        ttk.Button(bottom, text="Save", command=self._save_only).pack(side="right", padx=(0, 6))
        ttk.Button(bottom, text="Save & Reload", command=self._save_and_reload).pack(side="right", padx=(0, 6))

        top.columnconfigure(1, weight=1)
        top.rowconfigure(3, weight=1)

    def _current_text(self) -> str:
        return self.editor.get("1.0", "end-1c")

    def _is_dirty(self) -> bool:
        return self._current_text() != self._loaded_text

    def _reload_from_disk(self) -> None:
        if not self._config_path.exists():
            self.editor.delete("1.0", "end")
            self._loaded_text = ""
            self.status_var.set(f"Missing file: {self._config_path}")
            return
        text = self._config_path.read_text(encoding="utf-8")
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", text)
        self._loaded_text = text
        self.status_var.set("Loaded from disk")

    def _validate(self) -> bool:
        try:
            data = validate_hard_mode_yaml_text(self._current_text())
        except Exception as exc:
            messagebox.showerror("Validate hard mode config", str(exc), parent=self)
            self.status_var.set("YAML validation failed")
            return False
        families = len(data.get("query_families") or [])
        self.status_var.set(f"YAML valid. Top-level keys: {len(data)}. query_families: {families}.")
        return True

    def _save(self, *, reload_after: bool) -> None:
        if not self._validate():
            return
        text = self._current_text()
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomically(self._config_path, text)
        except Exception as exc:
            messagebox.showerror("Save hard mode config", str(exc), parent=self)
            self.status_var.set("Save failed")
            return

        self._loaded_text = text
        self.status_var.set(f"Saved {self._config_path.name}")
        logger.info("Saved hard mode config: %s", self._config_path)
        if reload_after and self._on_saved is not None:
            try:
                self._on_saved()
            except Exception as exc:
                messagebox.showerror("Reload hard mode config", str(exc), parent=self)
                self.status_var.set("Saved, but reload failed")
                return
            self.status_var.set(f"Saved and reloaded {self._config_path.name}")

    def _save_only(self) -> None:
        self._save(reload_after=False)

    def _save_and_reload(self) -> None:
        self._save(reload_after=True)

    def _on_close(self) -> None:
        if self._is_dirty():
            discard = messagebox.askyesno(
                "Discard changes?",
                "You have unsaved edits in the hard mode config editor. Close without saving?",
                parent=self,
            )
            if not discard:
                return
        self.destroy()
