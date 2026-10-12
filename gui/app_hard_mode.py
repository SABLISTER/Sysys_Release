#!/usr/bin/env python3
"""Hard mode GUI — query families, scoring, review. Run: python -m gui.app_hard_mode"""

from __future__ import annotations

import atexit
import csv
import json
import logging
import queue
import signal
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

PIPELINE_DIR = Path(__file__).resolve().parent.parent
ROOT = PIPELINE_DIR.parent
DEFAULT_HM_CONFIG = PIPELINE_DIR / "hard_mode_config.yaml"
DEFAULT_RESEARCH_CONFIG = PIPELINE_DIR / "research_config.yaml"

if (PIPELINE_DIR / "src").is_dir():
    sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PIPELINE_DIR / ".env")
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import yaml

from gui.hard_mode_config_editor import HardModeConfigEditor
from gui.hard_mode_pipeline_guide import PIPELINE_PHASE_GUIDE
from gui.search_runner import AsyncBridge
from core.config import Config
from hard_mode.dedupe import merge_and_dedupe_files
from hard_mode.dive import run_citation_dive, select_seeds
from hard_mode.fulltext import enrich_run_with_fulltext, export_run_to_regular_pipeline
from hard_mode.graph import graph_summary_text
from hard_mode.ingestion import run_family_retrieval
from hard_mode.load_config import load_hard_mode_config
from hard_mode.metrics import append_metrics_log, compute_metrics, write_metrics
from hard_mode.paths import ensure_run_dir, hard_mode_run_dir
from hard_mode.records import ensure_spine
from hard_mode.review_store import init_db, load_queue, save_review, sync_from_corpus
from hard_mode.scoring import score_corpus
from hard_mode.stopping import should_stop_retrieval
from hard_mode.triage import (
    benchmark_embedding_models_against_reviewed,
    cheap_score_corpus,
    compare_embedding_models,
    merge_scored_subset,
    select_for_pass3,
)
from process.llm import set_config as set_llm_config

logger = logging.getLogger(__name__)


class _QueueLogHandler(logging.Handler):
    """Put formatted log lines onto a queue for the Tk event loop."""

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self._log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log_queue.put_nowait(self.format(record))
        except Exception:
            self.handleError(record)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class HardModeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Research Pipeline — Hard Mode")
        self.root.geometry("980x860")
        self.root.minsize(800, 600)

        self._hm_path = DEFAULT_HM_CONFIG
        self._research_path = DEFAULT_RESEARCH_CONFIG
        research = _load_yaml(self._research_path)
        self.hm = load_hard_mode_config(self._hm_path, PIPELINE_DIR, research=research)
        self.config = Config(project_root=self._resolve_project_root(research))
        self.config.ensure_dirs()
        set_llm_config(self.config)

        self.run_id_var = tk.StringVar(value="default")
        self.hard_mode_base_var = tk.StringVar()
        self.keep_only_fulltext_var = tk.BooleanVar(
            value=bool((self.hm.get("full_text") or {}).get("keep_only_full_text", False))
        )
        self.triage_embedding_var = tk.StringVar()
        self.compare_embedding_var = tk.StringVar()
        self.compare_subset_var = tk.IntVar(value=100)
        self.review_t1_only_var = tk.BooleanVar(value=True)
        self.bridge = AsyncBridge()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._setup_logging()
        self._active_cancel: threading.Event | None = None
        self._cycle_var = tk.IntVar(value=0)

        self._queue_list: list[dict] = []
        self._current_uid: str | None = None
        self._detail_cache = ""
        self._output_window: tk.Toplevel | None = None
        self._output_text: tk.Text | None = None
        self._log_history: list[str] = []
        self._log_window: tk.Toplevel | None = None
        self._log_window_text: tk.Text | None = None
        self._apply_hard_mode_base_dir(self._resolve_hard_mode_base_dir(), update_var=True)
        self._sync_cheap_triage_controls_from_hm()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_menu()
        self._build_ui()
        self.run_id_var.trace_add("write", lambda *_args: self._refresh_run_dir_label())
        self._poll_logs()

    def _resolve_project_root(self, research: dict) -> Path:
        paths_cfg = (research or {}).get("paths") or {}
        raw = str(paths_cfg.get("project_root") or "").strip()
        if raw:
            pr = Path(raw).expanduser()
            if not pr.is_absolute():
                pr = (self._research_path.parent / pr).resolve()
            return pr
        return PIPELINE_DIR

    def _normalize_path_input(self, raw: str) -> Path:
        path = Path(str(raw).strip()).expanduser()
        if not path.is_absolute():
            path = (self.config.project_root / path).resolve()
        return path

    def _resolve_hard_mode_base_dir(self) -> Path:
        paths_cfg = (self.hm or {}).get("paths") or {}
        raw = str(paths_cfg.get("hard_mode_root") or "").strip()
        if raw:
            return self._normalize_path_input(raw)
        return self.config.data_dir / "hard_mode"

    def _apply_hard_mode_base_dir(self, raw_path: str | Path, *, update_var: bool = False):
        path = self._normalize_path_input(str(raw_path))
        setattr(self.config, "hard_mode_base_dir", path)
        if update_var:
            self.hard_mode_base_var.set(str(path))
        self._refresh_run_dir_label()

    def _refresh_llm_setup_label(self):
        rc = self.config.runtime_config("abstract")
        ft_rc = self.config.runtime_config("fulltext")
        ls = self.hm.get("llm_scoring") or {}
        ft = self.hm.get("full_text") or {}
        abs_n = ls.get("abstract_max_chars")
        tok_n = ls.get("max_output_tokens")
        reason = ls.get("use_reasoning")
        full_n = ft.get("score_max_chars")
        extra = ""
        if abs_n is not None or full_n is not None or tok_n is not None or reason is not None:
            extra = "\nhard_mode llm_scoring: abstract_max_chars=%s fulltext_max_chars=%s max_output_tokens=%s use_reasoning=%s" % (
                abs_n if abs_n is not None else "(config SCORE_ABSTRACT_CHARS)",
                full_n if full_n is not None else "(config SCORE_FULLTEXT_CHARS)",
                tok_n if tok_n is not None else "(auto from SCORE_MAX_TOKENS)",
                reason,
            )
        full_text_note = "\nfull_text: enabled=%s triage=%s pass3=%s" % (
            ft.get("enabled", True),
            ft.get("prefer_for_cheap_triage", True),
            ft.get("prefer_for_pass3", True),
        )
        full_text_note += " keep_only=%s" % bool(ft.get("keep_only_full_text", False))
        triage_cfg = self.hm.get("cheap_triage") or {}
        triage_model = str(triage_cfg.get("embedding_model") or self.config.embedding_model or "").strip()
        compare_model = str(triage_cfg.get("compare_embedding_model") or "").strip() or "(unset)"
        triage_note = "\ncheap_triage embeddings: primary=%s compare=%s subset=%s" % (
            triage_model,
            compare_model,
            int(triage_cfg.get("compare_subset_size") or 100),
        )
        self.llm_setup_lbl.configure(
            text="Pass 3 abstract: %s / %s\nPass 3 full text: %s / %s%s%s%s\n.env: LLM_PROVIDER=ollama|custom|openai|anthropic + host/model vars"
            % (
                rc.llm_provider,
                rc.llm_model,
                ft_rc.llm_provider,
                ft_rc.llm_model,
                extra,
                full_text_note,
                triage_note,
            ),
        )

    def _setup_logging(self):
        qh = _QueueLogHandler(self.log_queue)
        qh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(qh)
        logging.getLogger().setLevel(logging.INFO)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_m = tk.Menu(menubar, tearoff=0)
        file_m.add_command(label="Edit hard_mode_config…", command=self._open_hm_editor)
        file_m.add_command(label="Load hard_mode_config…", command=self._load_hm_config)
        file_m.add_command(label="Load research_config…", command=self._load_research_config)
        file_m.add_command(label="Pipeline Settings…", command=self._open_settings)
        file_m.add_command(label="Open output window…", command=self._open_log_window)
        file_m.add_separator()
        file_m.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_m)

        run_m = tk.Menu(menubar, tearoff=0)
        run_m.add_command(label="Stop current stage", command=self._on_stop, state="disabled")
        menubar.add_cascade(label="Run", menu=run_m)
        self.run_menu = run_m
        self._stop_idx = 0

        self.root.config(menu=menubar)

    def _set_stop_enabled(self, on: bool):
        self.run_menu.entryconfig(self._stop_idx, state="normal" if on else "disabled")

    def _on_stop(self):
        if self._active_cancel:
            self._active_cancel.set()
            self._set_stop_enabled(False)

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        setup = ttk.Frame(nb, padding=8)
        nb.add(setup, text="Setup")
        ttk.Label(setup, text="Run ID:").grid(row=0, column=0, sticky="w")
        ttk.Entry(setup, textvariable=self.run_id_var, width=24).grid(row=0, column=1, sticky="w")
        ttk.Label(setup, text="Hard mode config:").grid(row=1, column=0, sticky="nw", pady=4)
        self.hm_path_lbl = ttk.Label(setup, text=str(self._hm_path), wraplength=560)
        self.hm_path_lbl.grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(setup, text="Research config:").grid(row=2, column=0, sticky="nw", pady=4)
        self.rc_path_lbl = ttk.Label(setup, text=str(self._research_path), wraplength=560)
        self.rc_path_lbl.grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(setup, text="Hard mode base dir:").grid(row=3, column=0, sticky="nw", pady=4)
        base_row = ttk.Frame(setup)
        base_row.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Entry(base_row, textvariable=self.hard_mode_base_var, width=58).pack(side="left", fill="x", expand=True)
        ttk.Button(base_row, text="Browse…", command=self._browse_hard_mode_base_dir).pack(side="left", padx=(6, 0))
        ttk.Button(base_row, text="Set", command=self._set_hard_mode_base_dir).pack(side="left", padx=(6, 0))
        ttk.Label(setup, text="Current run dir:").grid(row=4, column=0, sticky="nw", pady=4)
        self.run_dir_lbl = ttk.Label(setup, text="", wraplength=560, justify="left")
        self.run_dir_lbl.grid(row=4, column=1, sticky="w", pady=4)
        ttk.Checkbutton(
            setup,
            text="Keep only papers with full text after Pass 1b / export",
            variable=self.keep_only_fulltext_var,
            command=self._toggle_keep_only_full_text,
        ).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(setup, text="Cheap triage embeddings:").grid(row=6, column=0, sticky="nw", pady=4)
        triage_row = ttk.Frame(setup)
        triage_row.grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(triage_row, text="Primary:").pack(side="left")
        ttk.Entry(triage_row, textvariable=self.triage_embedding_var, width=28).pack(side="left", padx=(5, 10))
        ttk.Label(triage_row, text="Compare:").pack(side="left")
        ttk.Entry(triage_row, textvariable=self.compare_embedding_var, width=28).pack(side="left", padx=(5, 10))
        ttk.Label(triage_row, text="Subset:").pack(side="left")
        ttk.Spinbox(triage_row, from_=10, to=5000, textvariable=self.compare_subset_var, width=7).pack(
            side="left",
            padx=(5, 0),
        )
        ttk.Label(
            setup,
            text="Leave Primary blank to fall back to the main pipeline embedding_model.",
            foreground="gray",
        ).grid(row=7, column=1, sticky="w", pady=(0, 4))
        ttk.Label(setup, text="LLM (Pass 3):").grid(row=8, column=0, sticky="nw", pady=4)
        self.llm_setup_lbl = ttk.Label(setup, text="", wraplength=560, justify="left")
        self.llm_setup_lbl.grid(row=8, column=1, sticky="w", pady=4)
        self._refresh_llm_setup_label()
        setup_btns = ttk.Frame(setup)
        setup_btns.grid(row=9, column=1, sticky="w", pady=(6, 0))
        ttk.Button(setup_btns, text="Edit hard mode YAML…", command=self._open_hm_editor).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(setup_btns, text="Pipeline Settings…", command=self._open_settings).pack(side="left")
        setup.columnconfigure(1, weight=1)
        self._refresh_run_dir_label()

        pipe = ttk.Frame(nb, padding=4)
        nb.add(pipe, text="Pipeline")
        pipe_pw = ttk.PanedWindow(pipe, orient=tk.VERTICAL)
        pipe_pw.pack(fill="both", expand=True)
        guide_fr = ttk.LabelFrame(pipe_pw, text="Pipeline map (read while Pass 3 / LLM runs)", padding=4)
        pipe_pw.add(guide_fr, weight=4)
        btn_fr = ttk.Frame(pipe_pw, padding=4)
        pipe_pw.add(btn_fr, weight=1)
        self.pipeline_guide = scrolledtext.ScrolledText(
            guide_fr,
            height=18,
            wrap="word",
            font=("TkDefaultFont", 10),
            state="disabled",
            bg="#f8f8f6",
        )
        self.pipeline_guide.pack(fill="both", expand=True)
        self.pipeline_guide.configure(state="normal")
        self.pipeline_guide.insert("1.0", PIPELINE_PHASE_GUIDE)
        self.pipeline_guide.configure(state="disabled")

        ttk.Button(
            btn_fr,
            text="1. Pass 1 — Retrieve (bounded per family)",
            command=self._run_retrieve,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="2. Pass 1 — Dedupe & merge (intake complete)",
            command=self._run_dedupe,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="3. Pass 1b — Enrich full text (OA)",
            command=self._run_fulltext,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="4. Pass 2 — Cheap triage",
            command=self._run_triage,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="4b. Compare embeddings (subset)",
            command=self._run_compare_triage_embeddings,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="4c. Benchmark embeddings vs reviewed",
            command=self._run_benchmark_triage_embeddings,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="5. Pass 3 — Score (LLM)",
            command=self._run_score,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="6. Pass 4 prep — Sync review database",
            command=self._run_sync_db,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="7. Expansion — Citation dive",
            command=self._run_dive,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="8. Expansion — Merge dive (re-dedupe)",
            command=self._merge_dive,
        ).pack(fill="x", pady=2)
        ttk.Button(
            btn_fr,
            text="9. Export — Regular pipeline corpus",
            command=self._run_export_regular_pipeline,
        ).pack(fill="x", pady=2)
        ttk.Button(btn_fr, text="10. Metrics / stop check", command=self._run_metrics).pack(fill="x", pady=2)

        review = ttk.Frame(nb, padding=8)
        nb.add(review, text="Review (Pass 4)")
        ttk.Label(
            review,
            text="Pass 4 — Human review: label relevance, tag domains, mark citation seeds.",
            wraplength=720,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(
            review,
            text="Show only Tier 1 in review queue",
            variable=self.review_t1_only_var,
            command=self._reload_queue,
        ).pack(anchor="w", pady=(0, 6))
        list_fr = ttk.Frame(review)
        list_fr.pack(fill="both", expand=True, side="left")
        ttk.Label(list_fr, text="Queue").pack(anchor="w")
        self.paper_list = tk.Listbox(list_fr, width=40, height=20, font=("Menlo", 10))
        self.paper_list.pack(fill="both", expand=True)
        self.paper_list.bind("<<ListboxSelect>>", self._on_select_paper)

        form = ttk.Frame(review)
        form.pack(fill="both", expand=True, side="right", padx=(12, 0))
        self.detail_text = tk.Text(form, height=12, wrap="word", font=("TkDefaultFont", 10))
        self.detail_text.pack(fill="both", expand=True)
        action_row = ttk.Frame(form)
        action_row.pack(anchor="w", pady=(6, 4), fill="x")
        ttk.Button(action_row, text="Open output window", command=self._open_output_window).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(action_row, text="Save review / submit", command=self._save_review).pack(side="left")

        top_controls = ttk.Frame(form)
        top_controls.pack(fill="x", pady=(0, 6))

        label_col = ttk.Frame(top_controls)
        label_col.pack(side="left", anchor="n")
        ttk.Label(label_col, text="Reviewer label:").pack(anchor="w")
        self.review_label_var = tk.StringVar(value="maybe")
        ttk.Combobox(
            label_col,
            textvariable=self.review_label_var,
            values=["relevant", "maybe", "irrelevant"],
            state="readonly",
            width=14,
        ).pack(anchor="w")

        comment_col = ttk.Frame(top_controls)
        comment_col.pack(side="left", fill="x", expand=True, padx=(12, 0), anchor="n")
        ttk.Label(comment_col, text="Comment:").pack(anchor="w")
        self.comment_text = tk.Text(comment_col, height=3, width=30)
        self.comment_text.pack(fill="x")

        self.axis_vars: dict[str, tk.BooleanVar] = {}
        self.axes_frame = ttk.LabelFrame(form, text="Axes (human)", padding=4)
        self.axes_frame.pack(fill="x", pady=6)
        self._rebuild_axis_checkboxes()
        self.integration_var = tk.StringVar(value="unsure")
        ttk.Label(form, text="Integration:").pack(anchor="w")
        ttk.Combobox(
            form,
            textvariable=self.integration_var,
            values=["real", "superficial", "unsure"],
            state="readonly",
            width=14,
        ).pack(anchor="w")
        self.seed_flag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Citation dive seed", variable=self.seed_flag_var).pack(anchor="w")
        ttk.Label(
            form,
            text="Citation dive only uses papers saved as relevant with Citation dive seed checked.",
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 4))
        self.review_status_var = tk.StringVar(
            value="Select a paper, then click Save review / submit to use it in citation dive."
        )
        ttk.Label(
            form,
            textvariable=self.review_status_var,
            wraplength=360,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Button(form, text="Reload queue", command=self._reload_queue).pack(anchor="w")

        metrics_f = ttk.Frame(nb, padding=8)
        nb.add(metrics_f, text="Metrics / Hints")
        ttk.Label(
            metrics_f,
            text="Expansion hint: concept dive (partial) — graph hints suggest bridge terms from reviewed papers.",
            wraplength=720,
        ).pack(anchor="w", pady=(0, 4))
        self.metrics_text = tk.Text(metrics_f, height=24, wrap="word", font=("Menlo", 10))
        self.metrics_text.pack(fill="both", expand=True)
        ttk.Button(metrics_f, text="Graph hints", command=self._show_graph_hints).pack(anchor="w")

        log_f = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_f.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_widget = tk.Text(log_f, height=7, state="disabled", bg="#1e1e1e", fg="#8c8")
        self.log_widget.pack(fill="both", expand=True)

        atexit.register(self.bridge.shutdown)

    def _rebuild_axis_checkboxes(self):
        for w in self.axes_frame.winfo_children():
            w.destroy()
        self.axis_vars.clear()
        for ax in self.hm.get("axes") or []:
            aid = str(ax.get("id", ""))
            if not aid:
                continue
            v = tk.BooleanVar(value=False)
            self.axis_vars[aid] = v
            row = ttk.Frame(self.axes_frame)
            row.pack(fill="x", anchor="w", pady=(0, 4))

            display = str(ax.get("display_name") or aid)
            short = aid.split("_")[-1].upper() if aid.startswith("axis_") else ""
            if short and short != display.strip().upper():
                heading = f"{short} — {display}"
            else:
                heading = display

            ttk.Checkbutton(
                row,
                text=heading,
                variable=v,
            ).pack(anchor="w")

            desc = str(ax.get("description_for_prompt") or "").strip()
            if desc:
                ttk.Label(
                    row,
                    text=desc,
                    wraplength=360,
                    justify="left",
                ).pack(anchor="w", padx=(24, 0))

    def _refresh_run_dir_label(self):
        if not hasattr(self, "run_dir_lbl"):
            return
        run_dir = hard_mode_run_dir(self.config, self.run_id_var.get().strip() or "default")
        self.run_dir_lbl.configure(text=str(run_dir))

    def _toggle_keep_only_full_text(self):
        self.hm.setdefault("full_text", {})
        self.hm["full_text"]["keep_only_full_text"] = bool(self.keep_only_fulltext_var.get())
        self._refresh_llm_setup_label()

    def _sync_cheap_triage_controls_from_hm(self):
        cfg = self.hm.get("cheap_triage") or {}
        self.triage_embedding_var.set(str(cfg.get("embedding_model") or ""))
        self.compare_embedding_var.set(str(cfg.get("compare_embedding_model") or ""))
        self.compare_subset_var.set(max(10, int(cfg.get("compare_subset_size") or 100)))

    def _apply_cheap_triage_controls_to_hm(self):
        cfg = self.hm.setdefault("cheap_triage", {})
        primary = self.triage_embedding_var.get().strip()
        compare = self.compare_embedding_var.get().strip()
        cfg["embedding_model"] = primary or None
        cfg["compare_embedding_model"] = compare or None
        cfg["compare_subset_size"] = max(10, int(self.compare_subset_var.get() or 100))
        self._refresh_llm_setup_label()

    def _browse_hard_mode_base_dir(self):
        path = filedialog.askdirectory(
            initialdir=self.hard_mode_base_var.get() or str(self.config.project_root),
        )
        if path:
            self.hard_mode_base_var.set(path)
            self._set_hard_mode_base_dir()

    def _set_hard_mode_base_dir(self):
        raw = self.hard_mode_base_var.get().strip()
        if not raw:
            raw = str(self.config.data_dir / "hard_mode")
        self._apply_hard_mode_base_dir(raw, update_var=True)

    def _set_detail_output(self, text: str):
        self._detail_cache = text
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", text)
        if self._output_text and self._output_window and self._output_window.winfo_exists():
            self._output_text.configure(state="normal")
            self._output_text.delete("1.0", "end")
            self._output_text.insert("end", text)
            self._output_text.configure(state="disabled")

    def _open_output_window(self):
        if self._output_window and self._output_window.winfo_exists():
            self._output_window.deiconify()
            self._output_window.lift()
            self._output_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Hard Mode Review Output")
        win.geometry("820x680")
        text = tk.Text(win, wrap="word", font=("TkDefaultFont", 10))
        text.pack(fill="both", expand=True)
        text.insert("1.0", self._detail_cache)
        text.configure(state="disabled")
        win.protocol("WM_DELETE_WINDOW", self._close_output_window)
        self._output_window = win
        self._output_text = text

    def _close_output_window(self):
        if self._output_window and self._output_window.winfo_exists():
            self._output_window.destroy()
        self._output_window = None
        self._output_text = None

    def _append_log_line(self, line: str):
        self._log_history.append(line)
        if len(self._log_history) > 5000:
            self._log_history = self._log_history[-5000:]

        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", line + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

        if self._log_window_text and self._log_window and self._log_window.winfo_exists():
            self._log_window_text.configure(state="normal")
            self._log_window_text.insert("end", line + "\n")
            self._log_window_text.see("end")
            self._log_window_text.configure(state="disabled")

    def _open_log_window(self):
        if self._log_window and self._log_window.winfo_exists():
            self._log_window.deiconify()
            self._log_window.lift()
            self._log_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Hard Mode Output")
        win.geometry("960x420")
        text = tk.Text(win, wrap="word", font=("Menlo", 10), bg="#1e1e1e", fg="#8c8")
        text.pack(fill="both", expand=True)
        if self._log_history:
            text.insert("1.0", "\n".join(self._log_history) + "\n")
        text.configure(state="disabled")
        win.protocol("WM_DELETE_WINDOW", self._close_log_window)
        self._log_window = win
        self._log_window_text = text

    def _close_log_window(self):
        if self._log_window and self._log_window.winfo_exists():
            self._log_window.destroy()
        self._log_window = None
        self._log_window_text = None

    def _poll_logs(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log_line(line)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_logs)

    def _run_dir(self) -> Path:
        return ensure_run_dir(self.config, self.run_id_var.get().strip() or "default")

    def _db_path(self) -> Path:
        return self._run_dir() / "review.db"

    def _load_hm_config(self):
        path = filedialog.askopenfilename(
            initialdir=str(PIPELINE_DIR),
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if not path:
            return
        self._hm_path = Path(path)
        self.hm_path_lbl.configure(text=str(self._hm_path))
        self._reload_hard_mode_from_disk()

    def _load_research_config(self):
        path = filedialog.askopenfilename(
            initialdir=str(PIPELINE_DIR),
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if not path:
            return
        self._research_path = Path(path)
        research = _load_yaml(self._research_path)
        self.config = Config(project_root=self._resolve_project_root(research))
        self.config.ensure_dirs()
        set_llm_config(self.config)
        self.hm = load_hard_mode_config(self._hm_path, PIPELINE_DIR, research=research)
        self.rc_path_lbl.configure(text=str(self._research_path))
        self._apply_hard_mode_base_dir(self._resolve_hard_mode_base_dir(), update_var=True)
        self.keep_only_fulltext_var.set(bool((self.hm.get("full_text") or {}).get("keep_only_full_text", False)))
        self._sync_cheap_triage_controls_from_hm()
        self._rebuild_axis_checkboxes()
        self._refresh_llm_setup_label()

    def _reload_hard_mode_from_disk(self):
        self.hm = load_hard_mode_config(
            self._hm_path,
            PIPELINE_DIR,
            research=_load_yaml(self._research_path),
        )
        self.hm_path_lbl.configure(text=str(self._hm_path))
        self._apply_hard_mode_base_dir(self._resolve_hard_mode_base_dir(), update_var=True)
        self.keep_only_fulltext_var.set(bool((self.hm.get("full_text") or {}).get("keep_only_full_text", False)))
        self._sync_cheap_triage_controls_from_hm()
        self._rebuild_axis_checkboxes()
        self._refresh_llm_setup_label()

    def _open_hm_editor(self):
        dialog = HardModeConfigEditor(
            self.root,
            config_path=self._hm_path,
            on_saved=self._reload_hard_mode_from_disk,
            on_open_settings=self._open_settings,
        )
        self.root.wait_window(dialog)

    def _open_settings(self):
        messagebox.showinfo(
            "Pipeline Settings",
            "API keys and LLM backends are configured in the main Sysys window "
            "(File → Settings). Hard-mode options are in File → Edit hard_mode_config…",
        )
        self._refresh_llm_setup_label()

    def _on_close(self):
        self.bridge.shutdown()
        self.root.destroy()

    def _run_retrieve(self):
        cancel = threading.Event()
        self._active_cancel = cancel
        self._set_stop_enabled(True)
        rid = self.run_id_var.get().strip() or "default"

        async def _job():
            try:
                return await run_family_retrieval(
                    self.config,
                    self.hm,
                    rid,
                    cancel_event=cancel,
                )
            finally:
                self.root.after(0, lambda: self._set_stop_enabled(False))
                self._active_cancel = None

        fut = self.bridge.submit(_job())

        def _done(_fut):
            try:
                n = len(_fut.result())
                messagebox.showinfo("Retrieve", f"Saved {n} raw hits.")
            except Exception as e:
                messagebox.showerror("Retrieve", str(e))

        self._poll_future(fut, _done)

    def _poll_future(self, fut, on_done):
        if fut.done():
            on_done(fut)
        else:
            self.root.after(150, lambda: self._poll_future(fut, on_done))

    def _run_dedupe(self):
        run_dir = self._run_dir()
        paths = [run_dir / "candidates_raw.json"]
        dive = run_dir / "candidates_dive.json"
        if dive.exists():
            paths.append(dive)
        merged = merge_and_dedupe_files(paths)
        for p in merged:
            ensure_spine(p)
        out = run_dir / "corpus_merged.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False, default=str)
        messagebox.showinfo("Dedupe", f"{len(merged)} papers -> corpus_merged.json")

    def _log_progress_payload(self, prefix: str, payload: dict | None):
        if not payload:
            return
        msg = str(payload.get("message") or "").strip()
        current = payload.get("current")
        total = payload.get("total")
        if msg and current is not None and total is not None:
            logger.info("%s: %s (%s/%s)", prefix, msg, current, total)
        elif msg:
            logger.info("%s: %s", prefix, msg)

    def _run_fulltext(self):
        run_dir = self._run_dir()
        src = run_dir / "corpus_merged.json"
        if not src.exists():
            messagebox.showwarning("Full text", f"Missing {src}")
            return

        cancel = threading.Event()
        self._active_cancel = cancel
        self._set_stop_enabled(True)
        rid = self.run_id_var.get().strip() or "default"

        def _work():
            return enrich_run_with_fulltext(
                self.config,
                self.hm,
                rid,
                cancel_event=cancel,
                progress_callback=lambda payload: self._log_progress_payload("Hard mode full text", payload),
            )

        fut = self.bridge.submit(self._async_score_wrapper(_work))

        def _done(_fut):
            self._set_stop_enabled(False)
            self._active_cancel = None
            result = _fut.result()
            if isinstance(result, Exception):
                messagebox.showerror("Full text", str(result))
                return
            if result.get("cancelled"):
                messagebox.showinfo(
                    "Full text",
                    "Stopped after enriching %s papers.\nOutput: %s" % (
                        result.get("fulltext_count", 0),
                        result.get("corpus_path", ""),
                    ),
                )
                return
            messagebox.showinfo(
                "Full text",
                (
                    "Enriched %s/%s papers with full text.\nOutput: %s"
                    % (
                        result.get("fulltext_count", 0),
                        result.get("paper_count_before_filter", result.get("paper_count", 0)),
                        result.get("corpus_path", ""),
                    )
                    + (
                        "\nKept %s full-text papers and dropped %s without full text."
                        % (
                            result.get("paper_count", 0),
                            result.get("dropped_without_fulltext", 0),
                        )
                        if result.get("kept_only_full_text")
                        else ""
                    )
                ),
            )

        self._poll_future(fut, _done)

    def _run_triage(self):
        self._apply_cheap_triage_controls_to_hm()
        run_dir = self._run_dir()
        src = run_dir / "corpus_fulltext.json"
        if not src.exists():
            src = run_dir / "corpus_merged.json"
        if not src.exists():
            messagebox.showwarning("Cheap triage", f"Missing {src}")
            return
        with open(src, encoding="utf-8") as f:
            papers = json.load(f)

        triaged = cheap_score_corpus(self.config, self.hm, papers)
        out = run_dir / "corpus_triaged.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(triaged, f, indent=2, ensure_ascii=False, default=str)

        band_counts: dict[str, int] = {}
        for paper in triaged:
            band = str((paper.get("cheap") or {}).get("queue_band") or "")
            if band:
                band_counts[band] = band_counts.get(band, 0) + 1
        msg = "Wrote %d papers.\nBands: %s" % (
            len(triaged),
            ", ".join(f"{k}={v}" for k, v in sorted(band_counts.items())) or "none",
        )
        if any((paper.get("cheap") or {}).get("text_source") == "full_text" for paper in triaged):
            msg += "\nUsed full text when available."
        used_models = sorted(
            {
                str((paper.get("cheap") or {}).get("semantic_embedding_model") or "").strip()
                for paper in triaged
                if str((paper.get("cheap") or {}).get("semantic_embedding_model") or "").strip()
            }
        )
        if used_models:
            msg += "\nEmbedding model: %s" % ", ".join(used_models)
        messagebox.showinfo("Cheap triage", msg)

    def _run_compare_triage_embeddings(self):
        self._apply_cheap_triage_controls_to_hm()
        run_dir = self._run_dir()
        src = run_dir / "corpus_fulltext.json"
        if not src.exists():
            src = run_dir / "corpus_merged.json"
        if not src.exists():
            messagebox.showwarning("Compare embeddings", f"Missing {src}")
            return
        with open(src, encoding="utf-8") as f:
            papers = json.load(f)

        try:
            result = compare_embedding_models(self.config, self.hm, papers)
        except Exception as e:
            messagebox.showerror("Compare embeddings", str(e))
            return

        json_path = run_dir / "triage_embedding_compare.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        csv_path = run_dir / "triage_embedding_compare.csv"
        rows = result.get("rows") or []
        if rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("")

        messagebox.showinfo(
            "Compare embeddings",
            "Compared %s sampled papers.\n%s vs %s\nBand agreement: %s\nTop-family agreement: %s\nMean |score delta|: %s\nJSON: %s\nCSV: %s"
            % (
                result.get("sample_size", 0),
                result.get("model_a", ""),
                result.get("model_b", ""),
                result.get("band_agreement", 0.0),
                result.get("top_family_agreement", 0.0),
                result.get("mean_absolute_score_delta", 0.0),
                json_path,
                csv_path,
            ),
        )

    def _run_benchmark_triage_embeddings(self):
        self._apply_cheap_triage_controls_to_hm()
        db = self._db_path()
        if not db.exists():
            messagebox.showwarning("Benchmark embeddings", "Sync and review some papers first.")
            return
        papers = load_queue(db)
        try:
            result = benchmark_embedding_models_against_reviewed(self.config, self.hm, papers)
        except Exception as e:
            messagebox.showerror("Benchmark embeddings", str(e))
            return

        run_dir = self._run_dir()
        json_path = run_dir / "triage_embedding_review_benchmark.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        csv_path = run_dir / "triage_embedding_review_benchmark.csv"
        rows = result.get("rows") or []
        if rows:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write("")

        summary_a = result.get("summary_a") or {}
        summary_b = result.get("summary_b") or {}
        messagebox.showinfo(
            "Benchmark embeddings",
            "%s reviewed papers.\n%s recall high+medium: %s | precision high+medium: %s\n%s recall high+medium: %s | precision high+medium: %s\nRelevant promoted to high by compare model: %s\nRelevant demoted from high by compare model: %s\nJSON: %s\nCSV: %s"
            % (
                result.get("reviewed_count", 0),
                result.get("model_a", ""),
                summary_a.get("relevant_recall_high_or_medium"),
                summary_a.get("precision_high_or_medium"),
                result.get("model_b", ""),
                summary_b.get("relevant_recall_high_or_medium"),
                summary_b.get("precision_high_or_medium"),
                result.get("relevant_promoted_to_high_in_b", 0),
                result.get("relevant_demoted_from_high_in_b", 0),
                json_path,
                csv_path,
            ),
        )

    def _scroll_pipeline_guide_to_pass3(self):
        w = self.pipeline_guide
        w.configure(state="normal")
        idx = w.search("Pass 3", "1.0", "end")
        if idx:
            w.see(idx)
        w.configure(state="disabled")

    def _run_score(self):
        run_dir = self._run_dir()
        triaged_src = run_dir / "corpus_triaged.json"
        merged_src = run_dir / "corpus_merged.json"
        if triaged_src.exists():
            src = triaged_src
        elif merged_src.exists():
            score_all = messagebox.askyesno(
                "Score without Pass 2?",
                "corpus_triaged.json is missing. Score the full merged corpus anyway?",
            )
            if not score_all:
                return
            src = merged_src
        else:
            messagebox.showwarning("Score", f"Missing {triaged_src} and {merged_src}")
            return
        self._scroll_pipeline_guide_to_pass3()
        with open(src, encoding="utf-8") as f:
            papers = json.load(f)

        selected_pool, selection_stats = select_for_pass3(self.hm, papers)
        selected = [
            paper
            for paper in selected_pool
            if bool((paper.get("derived") or {}).get("pass3_selected"))
        ]

        def _work():
            scored_subset = score_corpus(self.config, self.hm, selected, skip_if_machine=False)
            return merge_scored_subset(selected_pool, scored_subset)

        fut = self.bridge.submit(self._async_score_wrapper(_work))

        def _done(_fut):
            result = _fut.result()
            if isinstance(result, Exception):
                messagebox.showerror("Score", str(result))
                return
            out = run_dir / "corpus_scored.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            messagebox.showinfo(
                "Score",
                "Wrote %d papers.\nLLM scored %d/%d selected for Pass 3.\nBand counts: %s" % (
                    len(result),
                    selection_stats.get("selected", 0),
                    selection_stats.get("total", 0),
                    ", ".join(
                        f"{k}={v}" for k, v in sorted((selection_stats.get("by_band") or {}).items())
                    ) or "none",
                ),
            )

        self._poll_future(fut, _done)

    async def _async_score_wrapper(self, fn):
        import asyncio

        return await asyncio.to_thread(fn)

    def _run_sync_db(self):
        run_dir = self._run_dir()
        src = run_dir / "corpus_scored.json"
        if not src.exists():
            messagebox.showwarning("Sync", f"Missing {src}")
            return
        with open(src, encoding="utf-8") as f:
            papers = json.load(f)
        for p in papers:
            ensure_spine(p)
        db = self._db_path()
        init_db(db)
        sync_from_corpus(db, papers)
        self._reload_queue()
        t1_count = len(load_queue(db, max_tier=1))
        messagebox.showinfo(
            "Sync",
            "Review database updated.\nTier 1 queue: %s papers." % t1_count,
        )

    def _run_dive(self):
        db = self._db_path()
        if not db.exists():
            messagebox.showwarning("Dive", "Sync review database first.")
            return
        papers = load_queue(db)
        seeds = select_seeds(papers, self.hm)
        if not seeds:
            messagebox.showinfo(
                "Dive",
                "No seeds: in Review, set Reviewer label to relevant, check Citation dive seed, then click Save review / submit.",
            )
            return
        cancel = threading.Event()
        self._active_cancel = cancel
        self._set_stop_enabled(True)
        rid = self.run_id_var.get().strip() or "default"

        async def _job():
            try:
                return await run_citation_dive(
                    self.config,
                    self.hm,
                    rid,
                    seeds,
                    cancel_event=cancel,
                )
            finally:
                self.root.after(0, lambda: self._set_stop_enabled(False))
                self._active_cancel = None

        fut = self.bridge.submit(_job())

        def _done(_fut):
            try:
                n = len(_fut.result())
                messagebox.showinfo("Dive", f"Wrote {n} to candidates_dive.json")
            except Exception as e:
                messagebox.showerror("Dive", str(e))

        self._poll_future(fut, _done)

    def _merge_dive(self):
        self._run_dedupe()

    def _run_metrics(self):
        run_dir = self._run_dir()
        src = run_dir / "corpus_scored.json"
        if not src.exists():
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("end", "No corpus_scored.json.\n")
            return
        with open(src, encoding="utf-8") as f:
            papers = json.load(f)
        rid = self.run_id_var.get().strip() or "default"
        snap = compute_metrics(self.config, rid, papers)
        write_metrics(self.config, rid, snap)
        append_metrics_log(self.config, rid, snap)
        cy = self._cycle_var.get()
        stop, reason = should_stop_retrieval(self.hm, cy, int(snap.get("human_relevant") or 0))
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("end", json.dumps(snap, indent=2, default=str))
        tail = "\n\nStop check (cycle %s): stop=%s\n%s\n" % (cy, stop, reason)
        self.metrics_text.insert("end", tail)

    def _run_export_regular_pipeline(self):
        rid = self.run_id_var.get().strip() or "default"
        try:
            result = export_run_to_regular_pipeline(self.config, self.hm, rid)
        except Exception as e:
            messagebox.showerror("Export", str(e))
            return
        msg = "Exported %s papers to regular pipeline corpus.\n%s" % (
            result.get("paper_count", 0),
            result.get("corpus_path", ""),
        )
        if result.get("kept_only_full_text"):
            msg += "\nFull-text-only mode is ON."
        if result.get("fulltext_path"):
            msg += "\nFull text: %s enriched papers -> %s" % (
                result.get("fulltext_count", 0),
                result.get("fulltext_path", ""),
            )
        messagebox.showinfo("Export", msg)

    def _show_graph_hints(self):
        db = self._db_path()
        if not db.exists():
            messagebox.showwarning("Graph", "No review DB.")
            return
        papers = load_queue(db)
        txt = graph_summary_text(papers, self.hm)
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("end", txt)

    def _reload_queue(self):
        db = self._db_path()
        if not db.exists():
            self._queue_list = []
            self.paper_list.delete(0, "end")
            return
        max_tier = 1 if self.review_t1_only_var.get() else None
        self._queue_list = load_queue(db, max_tier=max_tier)
        self.paper_list.delete(0, "end")
        for p in self._queue_list:
            tier = (p.get("derived") or {}).get("priority_tier", "?")
            title = (p.get("title") or "")[:58]
            self.paper_list.insert("end", "T%s | %s" % (tier, title))

    def _on_select_paper(self, _evt=None):
        sel = self.paper_list.curselection()
        if not sel:
            return
        p = self._queue_list[sel[0]]
        self._current_uid = str(p.get("hard_mode_uid", ""))
        cheap = p.get("cheap") or {}
        m = p.get("machine") or {}
        d = p.get("derived") or {}
        h = p.get("human") or {}
        hits = p.get("retrieval_hits") or []
        lines = [
            "Title: %s" % p.get("title", ""),
            "Year: %s  DOI: %s" % (p.get("year", ""), p.get("doi", "")),
            "",
            "Cheap triage: band=%s tier=%s score=%s top_family=%s" % (
                cheap.get("queue_band"),
                cheap.get("priority_tier"),
                cheap.get("composite_score"),
                cheap.get("top_family_label"),
            ),
            "Cheap triage text source: %s" % (cheap.get("text_source") or "abstract"),
            "Pass 3 selected: %s (%s)" % (
                d.get("pass3_selected"),
                d.get("pass3_selection_reason", ""),
            ),
            "Tier: %s  Overlap: %s" % (d.get("priority_tier"), d.get("overlap_bucket")),
            "Pass 3 text source: %s" % (m.get("text_source") or "abstract"),
            "Scores: %s" % json.dumps(m.get("axis_scores")),
            "Rationale: %s" % m.get("rationale", ""),
            "",
            (p.get("abstract") or "")[:3500],
            "",
            "Provenance:",
        ]
        for hitem in hits[:15]:
            lines.append("  %s" % (hitem,))
        if len(hits) > 15:
            lines.append("  ... +%d more" % (len(hits) - 15))
        self._set_detail_output("\n".join(lines))

        self.review_label_var.set(h.get("reviewer_label") or "maybe")
        axes_h = h.get("axes_human") or []
        for aid, var in self.axis_vars.items():
            var.set(aid in axes_h)
        self.integration_var.set(h.get("integration_verdict") or "unsure")
        self.seed_flag_var.set(bool(h.get("citation_seed_flag")))
        self.comment_text.delete("1.0", "end")
        self.comment_text.insert("end", h.get("comment") or "")
        reviewed_at = h.get("reviewed_at")
        if reviewed_at:
            seed_ready = (
                (h.get("reviewer_label") == "relevant")
                and bool(h.get("citation_seed_flag"))
            )
            self.review_status_var.set(
                "Saved review loaded. Citation dive %s use this paper."
                % ("will" if seed_ready else "will not")
            )
        else:
            self.review_status_var.set(
                "Unsaved review state. Click Save review / submit after editing."
            )

    def _save_review(self):
        if not self._current_uid:
            messagebox.showwarning("Review", "Select a paper.")
            return
        axes = [aid for aid, v in self.axis_vars.items() if v.get()]
        comment = self.comment_text.get("1.0", "end").strip()
        save_review(
            self._db_path(),
            self._current_uid,
            reviewer_label=self.review_label_var.get(),
            axes_human=axes,
            integration_verdict=self.integration_var.get(),
            citation_seed_flag=self.seed_flag_var.get(),
            comment=comment,
        )
        self._reload_queue()
        seed_ready = (
            self.review_label_var.get() == "relevant"
            and self.seed_flag_var.get()
        )
        self.review_status_var.set(
            "Saved. Citation dive %s use this paper."
            % ("will" if seed_ready else "will not")
        )


def main():
    root = tk.Tk()
    HardModeApp(root)

    def _sigint(_s, _f):
        root.destroy()

    signal.signal(signal.SIGINT, _sigint)
    root.mainloop()


if __name__ == "__main__":
    main()
