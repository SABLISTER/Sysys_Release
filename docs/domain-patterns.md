# Domain vocabulary and heuristic rules

Use **Settings → Load Config** to load a research YAML or JSON file. The
Configuration File group displays the selected vocabulary pack. **Save Config**
exports the domain settings along with processing settings; **File → Save Project**
keeps them in `project_state.json`, and project reopen restores them after loading
the project's research config file.

[`templates/crop_drought.yaml`](../templates/crop_drought.yaml) is a complete
non-neuroscience example. Its Stage 3 fallback extracts water stress, Stage 4
recognizes field experiments with interviews as mixed methods, and Stage 9 links
drought with crop yield.

## Defaults and migration

`defaults.pattern_pack` accepts `general` (the default) or `neuroscience`.
The general pack supplies finding phrases, methodology hints, and broad study-design
rules. It contains no condition, mechanism, brain-region, criticality, or network
vocabulary. With no graph targets configured, Stage 9 reports network analysis as
skipped and writes an empty network artifact.

The neuroscience pack preserves the previous specialized vocabulary in
[`templates/domain_patterns/neuroscience.yaml`](../templates/domain_patterns/neuroscience.yaml).
The shipped autism and generic-neuroscience templates, and the active autism/E-I
research config, select it explicitly. Older projects that relied on implicit
neuroscience defaults should add:

```yaml
defaults:
  pattern_pack: neuroscience
```

## Graph patterns

`graph_analysis` accepts `conditions`, `entities`, `structural_regions`, and
`functional_networks`. Each category maps entity labels to lists of regular
expressions. The last two names support existing neuroscience projects; use
`entities` for neutral targets. Matching ignores case. Invalid regexes, non-list
patterns, and non-string entries raise a configuration error with the field path.

Condition precedence is:

1. Explicit `graph_analysis.conditions`, including an empty `{}`.
2. Literal whole-term matches for `co_occurring_conditions`.
3. Conditions from the explicitly selected vocabulary pack.

For other categories, an explicit map replaces the pack's category. Omitted
categories do not acquire neuroscience defaults unless that pack is selected.
An explicit empty map disables a category.

Stage 9 writes `condition_networks.json` and `condition_networks_report.md`.
The GUI prefers the new JSON name and falls back to `autism_condition_networks.json`
for older results. The import path `process.autism_graph_analysis` remains a
compatibility shim for the functions in `process.graph_analysis`; it does not
implicitly select the neuroscience pack. Existing edge/result keys are retained
for compatibility (`n_regions` counts target entities).

## Stage 3 extraction

`extraction.heuristics` accepts a regex map named `conditions`, plus literal-term
lists named `mechanisms`, `brain_regions`, `methodology_hints`, `finding_hints`,
`criticality_terms`, and `network_terms`. Explicit values replace the corresponding
pack values, and empty lists/maps disable them. Without an explicit extraction
condition map, project graph conditions or the configured condition list override
pack conditions.

Literal terms match at word boundaries, including punctuation-bearing phrases.
Search terms still inform the primary-question heuristic but are no longer
automatically labeled as mechanisms. Existing output field names remain unchanged;
the legacy `supports_criticality` and `network_effects_described` flags are false
unless their configured terms match. These are text heuristics, not evidence of
causation or a substitute for review.

The same configuration applies to heuristic-only extraction, failed LLM extraction,
and supplementation of empty LLM fields.

## Stage 4 method classification

`method_review.heuristic_rules` replaces the default ordered list in
[`templates/domain_patterns/general.yaml`](../templates/domain_patterns/general.yaml).
Each rule has a supported `study_method_type`, a numeric `confidence` in `[0, 1]`,
and one or both of `any_terms` and `all_terms`. The first matching rule wins.
When both lists are present, at least one `any_terms` phrase and every `all_terms`
phrase must match. Terms are literal, case-insensitive whole terms: `rat` does not
match `concentration`, and `PET` does not match `competition`.

An empty rule list disables classification heuristics. Unmatched text remains
`unknown` at confidence `0.1`; unsupported method types and malformed rules are
rejected. Custom rules map to the existing method taxonomy and apply both without
an LLM and when the LLM fails. The LLM prompt and method-validation taxonomy are
unchanged.
