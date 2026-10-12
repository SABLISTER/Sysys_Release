# SystS — Systematic Review Pipeline

## What This Is

SystS is a **PySide6 desktop application** for automated systematic literature reviews. It searches academic databases, scores/filters papers with LLMs, extracts claims, runs meta-analysis, and generates PRISMA-compliant reports. Entry point is `main.py`.

**~20,000 lines of Python across ~75 files.** Win98-themed GUI with Tahoma font.

## Architecture Overview

```
main.py                  # App entry point (PySide6 QApplication)
core/                    # Config dataclass, data models, snapshots
  config.py              # Config — all settings, YAML/JSON loading, env vars
  models.py              # Article, ClaimData, MethodProfile dataclasses
  snapshots.py           # Run snapshots for living review diffs
gui/                     # PySide6 GUI layer
  main_window.py         # MainWindow — 13+ tabs, all stage wiring (~5000 lines, largest file)
  dialogs.py             # SettingsDialog (API keys, LLM config, remember-keys privacy)
  workers.py             # QThread workers: PipelineSearchWorker, PipelineStageWorker
  stage_runner.py        # _STAGE_REGISTRY — maps stage names to async functions
  search_runner.py       # AsyncBridge, multi-source search orchestration
  theme.py               # Win98 stylesheet generator
  widgets.py             # Reusable widgets: MetricCard, GradeBadge, SectionHeader
  pipeline_board.py      # Visual pipeline progress board
  project_manager.py     # Project save/load/recent-projects
  network_view.py        # Force-directed network graph visualization
  prompt_preview.py      # LLM prompt preview dialog
  log_handler.py         # Thread-safe Python logging → GUI bridge
  help_content.py        # Per-tab help text
acquire/                 # Data acquisition from academic APIs
  orchestrator.py        # Multi-source search coordinator (PubMed, S2, OpenAlex, etc.)
  pubmed.py, semantic_scholar.py, openalex.py, scopus.py, web_of_science.py, elicit.py
  fulltext.py            # Full-text PDF retrieval (Unpaywall, OA sources)
  deduplicator.py        # Cross-source deduplication
process/                 # Analysis and processing modules
  llm.py                 # Central LLM dispatcher (Ollama, OpenAI, Anthropic, custom endpoint)
  custom_endpoint.py     # Custom endpoint URL normalization + model listing
  relevance_filter.py    # LLM scoring with checkpoint/resume
  claim_extractor.py     # Structured claim extraction
  method_review.py       # Study design classification
  synthesizer.py         # Claim clustering + narrative generation
  meta_analysis.py       # DerSimonian-Laird pooling, forest/funnel plots, Egger's test
  grade.py               # GRADE evidence assessment
  prisma.py              # PRISMA flow diagram + 27-item checklist
  export.py              # BibTeX/RIS/EndNote + report bundle
  coordinate_extractor.py # Brain coordinate extraction for neuroimaging
  neuroquery_bridge.py   # NeuroQuery convergence analysis
stages/                  # Pipeline stage entry points (s1 through s11)
  s1_search.py → s11_compare_runs.py
hard_mode/               # Advanced review mode (stubs, not fully implemented)
templates/               # YAML research config templates
tests/                   # Test files
```

## 11-Stage Pipeline

| # | Stage | Input → Output | Key File |
|---|-------|---------------|----------|
| 1 | Search | API queries → `corpus.json` | `stages/s1_search.py` |
| 2 | Score | corpus → `relevant.json` | `stages/s2_score.py` |
| 3 | Extract | relevant → `claims.json` | `stages/s3_extract.py` |
| 4 | Validate | claims → `method_profiles_validated.json` | `stages/s4_validate.py` |
| 5 | Audit | profiles → `claims_filtered.json` | `stages/s5_audit.py` |
| 6 | Synthesize | filtered → `clusters.json` + `abstract_draft.md` | `stages/s6_synthesize.py` |
| 7 | Fulltext | papers → `auto_extracted_metrics.json` | `stages/s7_fulltext.py` |
| 8 | Interrater | claims → `interrater_report.md` | `stages/s8_interrater.py` |
| 9 | Robustness | data → `robustness_results.json` | `stages/s9_robustness.py` |
| 10 | Verify | abstract → `verification_report.md` | `stages/s10_verify.py` |
| 11 | Compare | snapshots → `report.md` | `stages/s11_compare_runs.py` |

## GUI Structure

`gui/main_window.py` is the heart of the app. Key patterns:

- **Tab builder methods**: `_build_*_tab()` — each returns a QWidget
- **Stage launchers**: `_start_sysys_score()`, `_start_extract()`, etc. — each creates a `PipelineStageWorker`, connects signals, calls `.start()`
- **LLM backend rows**: `_make_llm_backend_row()` returns `(layout, provider_combo, model_edit, api_key_edit)` — used on every LLM stage tab
- **Config override**: `_apply_config_overrides_for_stage(provider, model, api_key)` — deep-copies config and applies per-stage LLM settings
- **Cancel flow**: `_cancel_pipeline_stage()` sets worker cancel flag AND re-enables tracked buttons via `_active_stage_run_btn` / `_active_stage_cancel_btn`
- **Corpus browser**: `_populate_corpus_browser(corpus)` fills the table; `_load_corpus_file()` lets users load previous JSON; `_auto_load_corpus()` runs on startup and project open

## LLM Backend (`process/llm.py`)

Four providers, all with 3x retry + exponential backoff:

| Backend | Auth | Config Fields |
|---------|------|--------------|
| Ollama | None | `ollama_model`, `ollama_base_url` |
| Custom | Key (optional) | `custom_api_format` (`openai`/`anthropic`/`lmstudio`), `custom_base_url`, `custom_model`, `custom_api_key` |
| OpenAI | API key | `openai_model`, `openai_base_url`, `openai_api_key` |
| Anthropic | API key | `anthropic_model`, `anthropic_api_key` |

`custom` covers LM Studio, LongCat, vLLM, OpenRouter, gateways, etc. The former
`lmstudio`/`longcat` providers migrate to it via `migrate_legacy_llm_fields`
(config files) and `restore_saved_settings` (QSettings). Stage rows and Settings
have a **Fetch Models** button (`gui/model_picker.py`) that lists the endpoint's models.

API keys are set per-stage in the GUI (inline Key field on each tab) and/or globally in File → Settings. The Settings dialog has a "Remember Keys" checkbox that controls QSettings persistence.

## Key Design Patterns

- **Workers run in QThreads**, communicate via Qt signals (`progress`, `finished`, `error`, `cancelled`)
- **AsyncBridge** wraps Python asyncio coroutines for use in QThread workers
- **Checkpoint/resume**: `process/relevance_filter.py` and `process/claim_extractor.py` save progress to allow resuming interrupted runs
- **Config is a dataclass** (`core/config.py`) loaded from YAML/JSON with env var resolution
- **QSettings** persists user preferences (API keys, font size, window state) across sessions

## Known Remaining Items

- `hard_mode/` package has stubs but internals may not match the original implementation
- `research_config.yaml` may differ from the original's 170-line version with 12 search domains

## Conventions

- Python 3.10+, PySide6 for GUI
- No type stubs needed — use `from __future__ import annotations` where needed
- Imports use package-relative paths: `core.config`, `process.llm`, `acquire.orchestrator`
- All stage functions are registered in `gui/stage_runner.py`'s `_STAGE_REGISTRY`
- Test with `python -c "import ast; ast.parse(open('file.py').read())"` for quick syntax checks
- The app launches with `python main.py`

## Verification Requirement

Resolve all encountered test, collection/import, and startup failures before
reporting completion. "Pre-existing", "unrelated", or an unchanged failure count
does not make a failed run acceptable. Fix stale fixtures while preserving their
behavioral assertions; do not hide failures through skips, `xfail`, deletion, or
weaker coverage. Run `venv/bin/python -m pytest tests/ -q`, including the real-window
startup test, and follow [docs/verification.md](docs/verification.md). Historical
reports and archived instructions do not override this requirement.
