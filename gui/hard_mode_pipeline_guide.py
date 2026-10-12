"""Long-form pipeline narrative for Hard Mode (shown in the GUI during slow steps)."""

PIPELINE_PHASE_GUIDE = """
FOUR-PASS CORE PIPELINE
══════════════════════

Pass 1 — Retrieval and intake
Pull a bounded slice from each source and each query family, not another giant corpus dump.
The goal is diversity, not volume. Every paper gets normalized metadata, deduplicated IDs,
title, abstract, keywords, source provenance, and citation links when available.

GUI: steps 1 (Retrieve) and 2 (Dedupe).


Pass 1b — Full-text enrichment and export
After dedupe, try to attach legally accessible open-access full text to the merged corpus.
This makes Pass 2 and Pass 3 better at filtering borderline papers that look weak from the
abstract alone but become obviously relevant once methods, named entities, measures,
conditions, or model details appear in the body text.

GUI: step 3 (Enrich full text). Step 9 exports the hard-mode run into the regular pipeline
corpus files so the classic pipeline can continue from the stronger intake set.


Pass 2 — Cheap scoring (fast triage)
This is where sentence-transformer similarity, TF-IDF, token overlap, and a simple composite
score belong. Use this layer to sort papers into broad queue bands: high-likelihood,
ambiguous, and low-likelihood. The metrics are correlated, so keep this stage simple — do
not over-engineer it. It is a fast triage gate before any expensive model. When full text is
available, this pass should prefer it over abstract-only evidence.

GUI: step 4 (Cheap triage).


Pass 3 — Targeted model scoring
Run the slower LLM only on papers that survived Pass 2, plus a small random sample from the
lower band for calibration. The model should not emit one giant relevance score. It should
emit structured fields aligned with your domains — for example (from a developmental
review): cognitive-construct relevance,
developmental-population relevance, developmental-design relevance, contextual-moderator
relevance, integration strength, domain-context confidence, and evidence type.

When full text exists, Pass 3 can switch to the full-text workload profile and use the
full-text model/provider overrides from the main config.

GUI: step 5 (Score LLM). This step can take a while — that is why this panel is here.

Local LLMs: Pass 3 uses the same stack as the rest of the pipeline — Ollama, LM Studio
(native API), or OpenAI-compatible endpoints (including LM Studio’s /v1 server with
LLM_PROVIDER=openai). You do not need a huge cloud model: keep prompts bounded via
hard_mode_config → llm_scoring and hard_mode_config → full_text
(abstract_max_chars, score_max_chars, max_output_tokens, use_reasoning).


Pass 4 — Human review
The reviewer sees title, abstract, provenance, a cheap-score summary, and model outputs.
They mark relevant, maybe, or not relevant, and tag which of the four thematic domains are
truly present. That reviewed set becomes your trusted seed set.

GUI: step 6 (Sync review DB), then the Review tab.


EXPANSION (only after you trust your reviewed seed set)
═══════════════════════════════════════════════════════

Start expansion once you have a reviewed seed set you believe in. Split it into three dive
modes (not all need to run every time):

1) Citation dive — Use the strongest reviewed papers as seeds. Go backward for foundations
   and forward for bridge papers. This is the cleanest way to fill sparse overlap buckets
   (e.g. executive function linked to school readiness in longitudinal child studies).

   GUI: step 7 (Citation dive), step 8 (merge / re-dedupe).

2) Author dive — From reviewed relevant papers, extract author names and build author
   scores. Boost authors who repeatedly appear in high-confidence bridge papers, then pull
   their other papers that keyword search missed.

   (Target: dedicated stage — not yet exposed as a button.)

3) Concept dive — Extract terms from the reviewed relevant set, weighted not by raw frequency
   but by how much each term distinguishes relevant from irrelevant papers. That surfaces
   bridge terms the original search may have missed (e.g. synonyms, older terminology,
   instrument or measure names, related constructs, adjacent methods, latent dimensions).

   GUI: Graph hints / metrics tab is a partial step toward concept-style suggestions.
""".strip()
