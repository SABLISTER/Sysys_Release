# SystS — Systematic Review Pipeline

**SystS** is a desktop application that automates systematic literature reviews
end to end. It searches major academic databases, scores and filters papers with
LLMs (or a local fine-tuned cross-encoder), extracts structured claims, runs
meta-analysis, grades evidence, and generates PRISMA-compliant reports — all from
a retro Win98-themed PySide6 GUI.

Built for living reviews: every stage checkpoints to disk, runs are resumable,
and snapshots let you diff one run against another as the literature evolves.

## Highlights

- **Multi-source search** — PubMed, Semantic Scholar, OpenAlex, Scopus,
  Web of Science, and Elicit, with cross-source deduplication
- **Flexible LLM backends** — Ollama, OpenAI, Anthropic, or any
  OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, …) with per-stage
  model overrides and an endpoint model picker
- **BRT-BERT relevance scorer** — a fine-tuned SciBERT cross-encoder
  (`brt-bert/`) for fast local title/abstract screening, with S2ORC training
  and hard-negative mining tooling included
- **Structured claim extraction** — findings, mechanisms, entities, and study
  metadata per paper, with checkpoint/resume and concurrent workers
- **Full meta-analysis toolkit** — DerSimonian–Laird and REML pooling, forest
  and funnel plots, Egger's test, and robustness checks
- **GRADE evidence assessment** and **PRISMA** flow diagram + 27-item checklist
- **Neuroimaging extras** — brain coordinate extraction and NeuroQuery
  convergence analysis
- **Exports** — BibTeX, RIS, EndNote, and a complete report bundle
- **Domain-agnostic** — research questions, terminology, and exclusion rules
  come from YAML config templates (`templates/`), not hardcoded strings

## The 11-Stage Pipeline

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Search | Query all configured databases → `corpus.json` |
| 2 | Score | Relevance screening (LLM or BRT-BERT) → `relevant.json` |
| 3 | Extract | Structured claim extraction → `claims.json` |
| 4 | Validate | Study design & method validation → `method_profiles_validated.json` |
| 5 | Audit | Claim auditing and filtering → `claims_filtered.json` |
| 6 | Synthesize | Claim clustering + narrative draft → `clusters.json`, `abstract_draft.md` |
| 7 | Fulltext | PDF retrieval & metric extraction → `auto_extracted_metrics.json` |
| 8 | Interrater | Cross-model reliability check → `interrater_report.md` |
| 9 | Robustness | Sensitivity analyses → `robustness_results.json` |
| 10 | Verify | Fact-check the synthesized abstract → `verification_report.md` |
| 11 | Compare | Diff against previous runs → `report.md` |

Every stage can be launched independently from its own GUI tab, with live
progress, cancellation, and resumable checkpoints.

## Getting Started

Requires **Python 3.13+**.

```sh
# With uv (recommended — resolves from uv.lock)
uv sync
uv run python main.py

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

On first launch, open **File → Settings** to configure an LLM backend
(Ollama needs no key; cloud providers need an API key). Keys are only persisted
if you check "Remember Keys".

## Configuration

- `research_config.yaml` — the active research question, search strategy,
  and domain vocabulary
- `templates/` — ready-made configs (neuroscience, digital mental health,
  crop drought, sociology, plus a blank template to start your own)
- `hard_mode_config.yaml` — settings for the advanced hard-mode review package

## Project Layout

```
main.py        Entry point (PySide6 app)
core/          Config dataclass, data models, atomic I/O, snapshots
gui/           MainWindow (13+ tabs), QThread workers, Win98 theme
acquire/       Academic API clients, full-text retrieval, deduplication
process/       LLM dispatcher, relevance scoring, claim extraction,
               synthesis, meta-analysis, GRADE, PRISMA, export
stages/        Stage entry points s1–s11
hard_mode/     Advanced review mode
brt-bert/      SciBERT cross-encoder relevance scorer + training scripts
templates/     YAML research config templates
tests/         533-test suite including a real-window GUI smoke test
```

## Development

```sh
uv run --with pytest python -m pytest tests/ -q
```

The suite includes a real `MainWindow` construction test (headless, offscreen).
See [docs/verification.md](docs/verification.md) for the verification policy,
and [AGENTS.md](AGENTS.md) for architecture notes and conventions.
