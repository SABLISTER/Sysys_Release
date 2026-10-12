"""
Help content for the Hypothesis Consensus Analyzer (Sysys) GUI.

Provides structured help text for all 12 pipeline tabs and welcome dialog
content.  This module has NO project imports to avoid circular dependencies.
"""

HELP_CONTENT = {
    # ------------------------------------------------------------------
    # Tab 0: Search
    # ------------------------------------------------------------------
    "search": {
        "title": "Stage 1: Literature Search",
        "icon": "\U0001f50d",
        "what": (
            "This stage executes parallel literature searches across multiple "
            "academic databases\u2014PubMed, Semantic Scholar, OpenAlex, Scopus, "
            "Web of Science, and Elicit\u2014using queries derived from your "
            "hypothesis components. Results are automatically de-duplicated "
            "using exact connected identifiers (DOI, PMID, PMCID, and "
            "provider IDs) plus fuzzy title similarity so that each unique "
            "article appears only once in your corpus. When duplicate records "
            "disagree, SystS keeps the copy with full text first, then the "
            "copy with richer connected identifiers and bibliographic "
            "metadata, while merging missing fields from the other copies. "
            "You can also "
            "supply a local folder of PDFs to supplement the database hits."
        ),
        "why": (
            "Comprehensive, reproducible searching is the foundation of any "
            "systematic review. Searching only one database risks missing "
            "relevant studies: PubMed emphasises biomedical literature while "
            "Semantic Scholar and OpenAlex have broader coverage of "
            "computational and interdisciplinary work. The Cochrane Handbook "
            "recommends searching at least two databases and documenting the "
            "search strategy so that it can be replicated. Automated "
            "de-duplication reduces the manual screening burden and keeps "
            "your PRISMA flow-diagram counts accurate."
        ),
        "outputs": (
            "The corpus table shows every retrieved article with its title, "
            "authors, year, DOI, and source database. The 'Unique' column "
            "indicates whether the article survived de-duplication. The "
            "progress bar tracks how many provider queries have completed. "
            "Once searching finishes, the total unique-article count is your "
            "initial screening population (N\u2080 in PRISMA terminology)."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 4: Searching for and selecting studies",
                "url": "https://training.cochrane.org/handbook/current/chapter-04",
            },
            {
                "text": "Lefebvre et al. (2022) \u2013 Searching for and selecting studies",
                "url": "https://doi.org/10.1002/14651858.ED000169",
            },
            {
                "text": "Page et al. (2021) \u2013 PRISMA 2020 statement",
                "url": "https://doi.org/10.1136/bmj.n71",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 1: Score & Filter
    # ------------------------------------------------------------------
    "score_filter": {
        "title": "Stage 2: Score & Filter",
        "icon": "\U0001f3af",
        "what": (
            "This stage uses the BRT-BERT relevance model to score each "
            "article\u2019s abstract (or full text when available) against your "
            "hypothesis on a 0\u20135 scale. Articles scoring below the "
            "relevance threshold are filtered out. When advanced options are "
            "enabled, you can swap in another configured scorer without "
            "changing the rest of the review workflow. The human calibration "
            "queue lets a reviewer score a small sample while BRT-BERT runs."
        ),
        "why": (
            "Manual title-and-abstract screening is the most time-consuming "
            "step in a traditional systematic review, often requiring two "
            "independent reviewers. BRT-BERT-first scoring accelerates this "
            "process while keeping the default path deterministic and "
            "review-specific. Setting an appropriate threshold balances "
            "recall (not missing relevant studies) against precision (not "
            "drowning in irrelevant ones). A threshold of 3\u20134 is "
            "conservative; 6+ is aggressive."
        ),
        "outputs": (
            "Stage 2 writes relevant.json for papers above threshold and "
            "scored_all.json for the full scored corpus. Human calibration "
            "reviews are saved to data/validation/human_relevance_review.json; "
            "agreement against BRT-BERT is saved to "
            "data/validation/human_relevance_agreement.json."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 4.6: Selecting studies",
                "url": "https://training.cochrane.org/handbook/current/chapter-04#section-4-6",
            },
            {
                "text": "Alshami et al. (2023) \u2013 LLMs for systematic review screening",
                "url": "https://doi.org/10.1016/j.jbi.2023.104519",
            },
            {
                "text": "Cohen et al. (2006) \u2013 Reducing workload in systematic reviews",
                "url": "https://doi.org/10.1197/jamia.M1929",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 2: Extract & Validate
    # ------------------------------------------------------------------
    "extract_validate": {
        "title": "Stage 3: Extract & Validate",
        "icon": "\U0001f4cb",
        "what": (
            "Extraction prompts the LLM to pull structured claims from each "
            "article\u2014specific findings, effect sizes, populations studied, "
            "and the direction of evidence (supports, contradicts, or mixed). "
            "Each claim is tagged with the hypothesis component it addresses. "
            "Validation cross-checks extracted claims against the source text "
            "and classifies the study design (RCT, cohort, case-control, "
            "cross-sectional, case report, review, or meta-analysis)."
        ),
        "why": (
            "Data extraction is the bridge between reading a paper and "
            "synthesising evidence. Structured extraction enforces "
            "consistency: every claim gets the same fields, making it "
            "possible to aggregate and compare across studies. Study-design "
            "classification feeds into the quality weighting later\u2014a "
            "well-powered RCT carries more evidential weight than a case "
            "report. The validation step catches hallucinated claims that "
            "lack textual support."
        ),
        "outputs": (
            "The claims table lists each extracted claim with its parent "
            "article, hypothesis component, evidence direction, confidence, "
            "and supporting quote. The study-design column shows the "
            "classified design. Validation flags appear as coloured icons: "
            "green (verified), yellow (partial match), red (unsupported)."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 5: Collecting data",
                "url": "https://training.cochrane.org/handbook/current/chapter-05",
            },
            {
                "text": "Higgins et al. (2019) \u2013 Data collection forms for intervention reviews",
                "url": "https://doi.org/10.1002/jrsm.1355",
            },
            {
                "text": "Wang et al. (2023) \u2013 GPT-based data extraction for systematic reviews",
                "url": "https://doi.org/10.1101/2023.10.05.23296583",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Stage 2c: Statistical Evidence Extraction
    # ------------------------------------------------------------------
    "statistical_extraction": {
        "title": "Stage 2c: Statistical Evidence Extraction",
        "icon": "\U0001f4ca",
        "what": (
            "This stage scans scored articles for reported statistics\u2014test "
            "names, test statistics, p-values, sample sizes, effect sizes\u2014and "
            "assembles them into a review table. An optional LLM context check "
            "validates only the local text windows around each extracted "
            "statistic, and items with missing or ambiguous fields are flagged "
            "for human review."
        ),
        "why": (
            "Meta-analysis and GRADE assessment depend on accurate numeric "
            "inputs. Pulling statistics into a structured, reviewable table "
            "makes it easy to spot extraction errors before they propagate "
            "into pooled estimates, and keeps a human in the loop for the "
            "cases automated extraction is least confident about."
        ),
        "outputs": (
            "The review table lists each extracted statistic with its status, "
            "whether an LLM check ran, the source article, test, statistic, "
            "p-value, N, and any missing fields. Selecting a row shows the "
            "surrounding source context. Results are written to "
            "statistical_extractions.json for use in later stages."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 6: Choosing effect measures and computing estimates",
                "url": "https://training.cochrane.org/handbook/current/chapter-06",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 3: Audit
    # ------------------------------------------------------------------
    "audit": {
        "title": "Stage 5: Span Audit",
        "icon": "\U0001f50e",
        "what": (
            "The audit step computes pairwise semantic similarity between "
            "evidence spans using sentence embeddings, flagging clusters of "
            "near-duplicate or suspiciously similar text."
        ),
        "why": (
            "Span-similarity auditing catches two problems: (1) genuine "
            "duplicates where the same finding appears in multiple papers "
            "(common with conference/journal pairs), and (2) LLM extraction "
            "artifacts where the model paraphrases the same source sentence "
            "differently for different claims."
        ),
        "outputs": (
            "The similarity matrix highlights span pairs exceeding the "
            "cosine-similarity threshold (default 0.85). Flagged pairs are "
            "listed for manual review."
        ),
        "reading": [
            {
                "text": "Reimers & Gurevych (2019) \u2013 Sentence-BERT for semantic similarity",
                "url": "https://doi.org/10.18653/v1/D19-1410",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 4: Synthesis
    # ------------------------------------------------------------------
    "synthesis": {
        "title": "Stage 6: Synthesis",
        "icon": "\U0001f9e9",
        "what": (
            "The synthesis step groups related claims by hypothesis "
            "component and generates a cluster-level narrative summary\u2014"
            "a paragraph describing what the evidence collectively says "
            "about each component."
        ),
        "why": (
            "Narrative synthesis converts a long list of individual claims "
            "into a readable summary, similar to the \u2018Summary of "
            "findings\u2019 tables recommended by GRADE."
        ),
        "outputs": (
            "The cluster summary table lists each cluster with its paper "
            "count and key terms. The sub-tabs below it hold the generated "
            "abstract, the deterministic \u2018Why This Hypothesis\u2019 "
            "provenance report, per-claim source citations, and the "
            "evidence network graph."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 12: Synthesizing and presenting findings using other methods",
                "url": "https://training.cochrane.org/handbook/current/chapter-12",
            },
            {
                "text": "Popay et al. (2006) \u2013 Guidance on narrative synthesis",
                "url": "https://doi.org/10.13140/2.1.1018.4643",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 5: Fulltext & Interrater
    # ------------------------------------------------------------------
    "fulltext_interrater": {
        "title": "Fulltext & Interrater",
        "icon": "\U0001f4da",
        "what": (
            "This stage attempts to retrieve full-text PDFs or HTML for "
            "every article in your corpus via Unpaywall, Semantic Scholar, "
            "and publisher open-access links. Successfully retrieved texts "
            "are re-scored and re-extracted to improve claim quality. "
            "Stage 7 can stream each paper into resumable JSONL checkpoints; "
            "corpus enrichment can optionally checkpoint a PaperQA index. "
            "Interrater reliability then sends a random subsample to a "
            "configured interrater scorer and "
            "computes Cohen\u2019s kappa and percentage agreement between the "
            "two sets of scores."
        ),
        "why": (
            "Abstracts omit methods, limitations, and effect-size details "
            "that are critical for evidence synthesis. Full-text retrieval "
            "allows the pipeline to extract richer, more accurate claims. "
            "Interrater reliability is a core methodological requirement: "
            "Cochrane mandates independent screening by at least two people. "
            "Using a model-based second rater provides an "
            "analogous check. Kappa \u2265 0.61 is typically considered "
            "substantial agreement."
        ),
        "outputs": (
            "The fulltext panel shows retrieval status per article: "
            "\u2018retrieved\u2019, \u2018abstract only\u2019, or \u2018failed\u2019 with the error "
            "reason. Stage 7 keeps auto_extracted_metrics.jsonl and "
            "download_log.jsonl for restart-safe progress, then materializes "
            "deduplicated JSON arrays for downstream stages. The interrater "
            "panel displays the kappa statistic, "
            "percentage agreement, and a confusion matrix. Articles where "
            "the two raters disagree beyond a threshold are flagged for "
            "human adjudication."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 4.4: Selecting studies \u2013 resolving disagreements",
                "url": "https://training.cochrane.org/handbook/current/chapter-04#section-4-4",
            },
            {
                "text": "McHugh (2012) \u2013 Interrater reliability: the kappa statistic",
                "url": "https://doi.org/10.11613/BM.2012.031",
            },
            {
                "text": "Piwowar et al. (2018) \u2013 The state of OA: a large-scale analysis",
                "url": "https://doi.org/10.7717/peerj.4375",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 6: Robustness & Verify
    # ------------------------------------------------------------------
    "robustness_verify": {
        "title": "Stage 6: Robustness, Verify & Meta-Analysis",
        "icon": "\U0001f9ea",
        "what": (
            "Robustness testing probes how stable your conclusions are under "
            "perturbation. It measures one outcome: the share of papers whose "
            "extracted claims support the research question "
            "(supports_primary_question, or the field set in "
            "robustness.outcome_field). The stage runs bootstrap resampling "
            "of relevance scores, split-half reliability of the per-condition "
            "support profile, sensitivity analysis across relevance "
            "thresholds 1\u20135, a permutation test of whether your "
            "co_occurring_conditions ranking tracks support rates, and a "
            "per-condition sensitivity check (each condition's papers "
            "included vs. excluded). Verification re-checks every number in "
            "the synthesized abstract against the corpus and robustness "
            "results to catch hallucinated or drifted statistics.\n\n"
            "The Meta-Analytic Synthesis section performs random-effects "
            "meta-analysis using the DerSimonian\u2013Laird method to pool "
            "effect sizes (Hedges\u2019 g) extracted during Stage 7 (Fulltext). "
            "It produces a forest plot showing each study\u2019s effect size "
            "and confidence interval alongside the pooled estimate, a "
            "contour-enhanced funnel plot for visualising publication bias, "
            "and three formal bias tests: Egger\u2019s regression test, Duval "
            "& Tweedie trim-and-fill, and Rosenthal\u2019s fail-safe N."
        ),
        "why": (
            "A systematic review whose conclusions change dramatically when "
            "one or two papers are removed is fragile. Robustness analyses "
            "quantify this fragility. Bootstrap confidence intervals tell "
            "you how precise your relevance estimates are; split-half "
            "reliability tells you whether random halves of your corpus "
            "agree on which conditions show more support. The permutation "
            "test gives a p-value for \u2018does the condition ranking track "
            "support more than chance?\u2019, and the condition sensitivity "
            "check shows whether one condition\u2019s literature drives the "
            "pooled result.\n\n"
            "Effect size pooling via a random-effects model is the "
            "quantitative backbone of meta-analysis. The DerSimonian\u2013Laird "
            "method allows between-study variance (\u03c4\u00b2), so the pooled "
            "estimate accounts for heterogeneity rather than assuming all "
            "studies estimate the same true effect. The I\u00b2 statistic "
            "describes what percentage of variability is due to real "
            "differences between studies rather than sampling error: I\u00b2 "
            "< 25% is low heterogeneity, 25\u201375% is moderate, and > 75% "
            "is high. A prediction interval shows where the true effect is "
            "likely to fall in a future study.\n\n"
            "Publication bias occurs when studies with statistically "
            "significant results are more likely to be published, skewing "
            "the evidence base. Egger\u2019s regression test detects funnel "
            "plot asymmetry (significant intercept at p < 0.10 suggests "
            "bias). Trim-and-fill estimates how many studies are \u2018missing\u2019 "
            "and recalculates the pooled effect after imputing them. "
            "Rosenthal\u2019s fail-safe N tells you how many unpublished null "
            "studies would be needed to render the pooled result non-"
            "significant \u2014 a large fail-safe N means the finding is robust."
        ),
        "outputs": (
            "robustness_report.md names the outcome measured and reports the "
            "bootstrap 95% confidence interval, the Spearman\u2013Brown "
            "corrected split-half reliability, a table of support rates per "
            "relevance threshold, the permutation test\u2019s Spearman r and "
            "p-value with the per-condition gradient, and a per-condition "
            "sensitivity table. Verification results appear as PASS / FAIL / "
            "WARN per number in the abstract.\n\n"
            "The forest plot displays each study as a horizontal line "
            "(confidence interval) with a square (effect size, sized by "
            "weight). The diamond at the bottom is the pooled random-effects "
            "estimate. Heterogeneity statistics (I\u00b2, \u03c4\u00b2, Q, p) are "
            "annotated below the plot.\n\n"
            "The funnel plot places each study\u2019s effect size on the x-axis "
            "and its standard error on the y-axis (inverted, so larger "
            "studies are at the top). A symmetric funnel indicates no "
            "publication bias; asymmetry suggests small studies with non-"
            "significant results may be missing. Shaded contour regions "
            "show significance thresholds (p < .05, .01, .001). If "
            "trim-and-fill imputes studies, they appear as open circles. "
            "Egger\u2019s regression line is overlaid when the test is run.\n\n"
            "The summary panel reports the pooled effect and 95% CI, "
            "heterogeneity metrics, Egger\u2019s test result, trim-and-fill "
            "adjustment, and fail-safe N. A warning is shown if fewer than "
            "10 studies are available, as bias tests have low statistical "
            "power in small meta-analyses."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 10.14: Sensitivity analyses",
                "url": "https://training.cochrane.org/handbook/current/chapter-10#section-10-14",
            },
            {
                "text": "Efron & Tibshirani (1993) \u2013 An Introduction to the Bootstrap",
                "url": "https://doi.org/10.1201/9780429246593",
            },
            {
                "text": "Mathur & VanderWeele (2020) \u2013 Sensitivity analysis for publication bias",
                "url": "https://doi.org/10.1177/0962280218820575",
            },
            {
                "text": "DerSimonian & Laird (1986) \u2013 Meta-analysis in clinical trials",
                "url": "https://doi.org/10.1016/0197-2456(86)90046-2",
            },
            {
                "text": "Cochrane Handbook Ch. 10.4: Detecting reporting biases",
                "url": "https://training.cochrane.org/handbook/current/chapter-10#section-10-4",
            },
            {
                "text": "Duval & Tweedie (2000) \u2013 Trim and fill: a simple funnel-plot-based method",
                "url": "https://doi.org/10.1111/j.0006-341X.2000.00455.x",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 7: Text Scaling
    # ------------------------------------------------------------------
    "text_scaling": {
        "title": "Stage 7: Text Scaling",
        "icon": "\U0001f4d0",
        "what": (
            "Text scaling applies the Wordfish algorithm to position each "
            "article on a latent ideological/thematic dimension estimated "
            "from word frequencies. Articles that use similar vocabulary "
            "cluster together; those using different terminology are placed "
            "further apart. A complementary stylometric analysis computes "
            "per-article features such as average sentence length, "
            "type\u2013token ratio, passive-voice frequency, and hedging-word "
            "density."
        ),
        "why": (
            "Wordfish was originally developed for scaling legislative texts "
            "on a left\u2013right dimension but is equally useful for mapping the "
            "terminological landscape of a scientific literature. In this "
            "context it can reveal whether your corpus separates into "
            "distinct communities (e.g., experimental vs. observational "
            "research traditions) that frame the hypothesis differently. "
            "Stylometric features help identify outlier articles\u2014a paper "
            "with extremely high hedging language may be a speculative "
            "commentary rather than an empirical study."
        ),
        "outputs": (
            "The Wordfish panel lists articles on the estimated dimension "
            "(theta values) and shows the most discriminating words for its "
            "negative and positive poles. The convergence indicator reports "
            "whether the Poisson model met its likelihood tolerance. "
            "The stylometry table lists per-article features and highlights "
            "differences in hedging, certainty language, reading grade, and "
            "lexical diversity. Select a row for additional details."
        ),
        "reading": [
            {
                "text": "Slapin & Proksch (2008) \u2013 A scaling model for estimating time-series party positions",
                "url": "https://doi.org/10.1111/j.1540-5907.2008.00338.x",
            },
            {
                "text": "Grimmer & Stewart (2013) \u2013 Text as data: the promise and pitfalls",
                "url": "https://doi.org/10.1093/pan/mps028",
            },
            {
                "text": "Stamatatos (2009) \u2013 A survey of modern authorship attribution methods",
                "url": "https://doi.org/10.1002/asi.21001",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 8: Evidence Map
    # ------------------------------------------------------------------
    "evidence_map": {
        "title": "Stage 8: Evidence Map",
        "icon": "\U0001f5fa\ufe0f",
        "what": (
            "The evidence map is an article \u00d7 component matrix where each "
            "cell indicates whether a given article provides evidence for "
            "a given hypothesis component, and in which direction (supports, "
            "contradicts, or mixed). Trend charts show how the volume and "
            "direction of evidence have changed over publication years. "
            "Gap analysis highlights components with little or no coverage."
        ),
        "why": (
            "Evidence maps provide a bird\u2019s-eye view of where the literature "
            "is dense and where it is sparse. Identifying gaps is as "
            "important as summarising what is known\u2014it tells you where "
            "future research is needed. The temporal trend view reveals "
            "whether interest in a component is growing, stable, or "
            "declining, which is useful context for interpreting the "
            "overall alignment score."
        ),
        "outputs": (
            "The heat-map table uses colour-coded cells: green for "
            "supporting evidence, red for contradicting, yellow for mixed, "
            "and grey for no data. Row totals show per-article coverage; "
            "column totals show per-component coverage. The trend chart "
            "plots publication counts by year, split by evidence direction."
        ),
        "reading": [
            {
                "text": "Miake-Lye et al. (2016) \u2013 What is an evidence map?",
                "url": "https://doi.org/10.1186/s13643-016-0204-x",
            },
            {
                "text": "Cochrane Handbook Ch. 12.2: Approaches to synthesis without meta-analysis",
                "url": "https://training.cochrane.org/handbook/current/chapter-12#section-12-2",
            },
            {
                "text": "Snilstveit et al. (2016) \u2013 Evidence & gap maps: a tool for evidence synthesis",
                "url": "https://doi.org/10.4073/cmdp.2016.3",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 9: Results Dashboard
    # ------------------------------------------------------------------
    "results_dashboard": {
        "title": "Stage 9: Results Dashboard & GRADE Assessment",
        "icon": "\U0001f4ca",
        "what": (
            "The results dashboard displays the final alignment score for "
            "each hypothesis component, aggregated from all supporting and "
            "contradicting evidence weighted by study design and relevance "
            "score. An overall composite score summarises the degree to "
            "which the literature supports the hypothesis as a whole. "
            "Metric cards show key summary statistics: total articles, "
            "mean relevance score, and the number of verified claims.\n\n"
            "The dashboard also includes a GRADE (Grading of Recommendations "
            "Assessment, Development and Evaluation) certainty assessment. "
            "GRADE is the international standard for rating the certainty of "
            "evidence in systematic reviews. Click 'Run GRADE Assessment' to "
            "evaluate the quality of evidence for each hypothesis component "
            "across five domains: risk of bias, inconsistency, indirectness, "
            "imprecision, and publication bias."
        ),
        "why": (
            "The dashboard is where all upstream processing converges into "
            "an interpretable answer. The alignment score for each component "
            "ranges from \u22121 (strong contradiction) through 0 (balanced or "
            "insufficient evidence) to +1 (strong support). Weighting by "
            "study design prevents a single low-quality case report from "
            "having the same influence as a large meta-analysis. Comparing "
            "component scores reveals which aspects of the hypothesis are "
            "well-supported and which are contested.\n\n"
            "GRADE adds a crucial second dimension: certainty of evidence. "
            "A high support score with Low GRADE certainty means the "
            "literature agrees but the evidence may not be trustworthy \u2014 "
            "for example, agreement from many low-quality case reports. "
            "The five GRADE domains capture different threats to "
            "trustworthiness:\n\n"
            "\u2022 Risk of Bias \u2014 study design quality (RCTs vs. case reports, "
            "control groups, flagged studies).\n"
            "\u2022 Inconsistency \u2014 heterogeneity across studies (interrater "
            "agreement, robustness stability).\n"
            "\u2022 Indirectness \u2014 how directly evidence addresses the "
            "hypothesis component (keyword match rate).\n"
            "\u2022 Imprecision \u2014 wide confidence intervals or small sample "
            "sizes (bootstrap CI width, number of studies).\n"
            "\u2022 Publication Bias \u2014 whether studies with null results are "
            "underrepresented (assessed via Egger\u2019s test when available).\n\n"
            "Certainty starts at High for bodies of evidence dominated by "
            "RCTs or systematic reviews, and Low for observational studies. "
            "Each domain can downgrade certainty by one or two levels, "
            "yielding a final rating of High, Moderate, Low, or Very Low."
        ),
        "outputs": (
            "Each component is displayed as an AlignmentBar ranging from "
            "\u22121 to +1. The bar colour shifts from red (contradiction) "
            "through grey (neutral) to green (support). After running the "
            "GRADE assessment, a coloured certainty badge appears beside "
            "each bar: green (High), amber (Moderate), orange (Low), or "
            "red (Very Low).\n\n"
            "The Summary of Findings (SoF) table below lists every "
            "component with colour-coded cells for each GRADE domain. "
            "Click a row to see the detailed rationale for each domain "
            "rating in the detail panel. Metric cards at the top summarise "
            "the corpus size, filtering yield, and overall confidence."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 14: Completing \u2018Summary of findings\u2019 tables",
                "url": "https://training.cochrane.org/handbook/current/chapter-14",
            },
            {
                "text": "Guyatt et al. (2011) \u2013 GRADE guidelines: rating quality of evidence",
                "url": "https://doi.org/10.1016/j.jclinepi.2010.07.015",
            },
            {
                "text": "Schunemann et al. (2019) \u2013 GRADE approach for diagnostic tests",
                "url": "https://doi.org/10.1016/j.jclinepi.2018.01.015",
            },
            {
                "text": "GRADE Working Group \u2013 Official website and resources",
                "url": "https://www.gradeworkinggroup.org/",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 11: Compare Runs
    # ------------------------------------------------------------------
    "compare_runs": {
        "title": "Stage 10: Compare Runs & Living Reviews",
        "icon": "\U0001f504",
        "what": (
            "Compare Runs lets you load results from multiple pipeline "
            "executions and view them side-by-side. This is useful when you "
            "change the LLM provider, adjust the relevance threshold, swap "
            "search databases, or modify hypothesis components between runs. "
            "A diff table highlights which articles changed score, which "
            "claims were added or dropped, and how alignment scores shifted.\n\n"
            "The Living Review Snapshots section supports longitudinal "
            "tracking of your systematic review. Click \u2018Save Current "
            "Snapshot\u2019 to freeze the current pipeline state (corpus size, "
            "claim counts, stage completion, file hashes). Select two "
            "snapshots from the list and click \u2018Compare Selected\u2019 to see "
            "what changed between them: new articles added, articles "
            "removed, relevance score shifts, claim count changes, and "
            "GRADE certainty updates."
        ),
        "why": (
            "Sensitivity to analytical choices is a well-known concern in "
            "evidence synthesis. By comparing runs you can assess whether "
            "your conclusions are robust to the specific configuration you "
            "chose. If switching from one LLM to another flips a "
            "component\u2019s alignment from positive to negative, that "
            "component\u2019s evidence base may be too thin or ambiguous to "
            "draw firm conclusions. This is analogous to the \u2018multiverse "
            "analysis\u2019 concept in meta-science.\n\n"
            "Living systematic reviews are an emerging paradigm where the "
            "review is updated continually as new evidence is published. "
            "The Cochrane Living Systematic Review Network recommends "
            "documenting what changed at each update and whether new "
            "evidence altered the conclusions. Snapshots automate this "
            "documentation: each snapshot records the complete pipeline "
            "state, and the diff report shows exactly what shifted, "
            "making it straightforward to write the \u2018What\u2019s new\u2019 section "
            "for an updated review."
        ),
        "outputs": (
            "The comparison table shows per-component alignment scores for "
            "each loaded run, with the delta highlighted in colour. The "
            "article-level diff lists papers that changed relevance "
            "category. A summary row at the bottom shows whether the "
            "overall conclusion (supports / contradicts / insufficient) "
            "changed between runs.\n\n"
            "The snapshot comparison shows a structured diff: new articles "
            "(with titles and DOIs), removed articles, articles whose "
            "relevance scores changed, claim count changes, and GRADE "
            "certainty updates. A markdown report is automatically saved "
            "to the output directory for inclusion in the review\u2019s "
            "supplementary materials."
        ),
        "reading": [
            {
                "text": "Steegen et al. (2016) \u2013 Increasing transparency through a multiverse analysis",
                "url": "https://doi.org/10.1177/1745691616658637",
            },
            {
                "text": "Cochrane Handbook Ch. 10.14: Investigating heterogeneity",
                "url": "https://training.cochrane.org/handbook/current/chapter-10#section-10-14",
            },
            {
                "text": "Page et al. (2021) \u2013 PRISMA 2020 statement: Item 24, sensitivity analyses",
                "url": "https://doi.org/10.1136/bmj.n71",
            },
            {
                "text": "Elliott et al. (2017) \u2013 Living systematic reviews",
                "url": "https://doi.org/10.1371/journal.pmed.1002260",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 12: Report & Export
    # ------------------------------------------------------------------
    "report_export": {
        "title": "Report & Export",
        "icon": "\U0001f4cb",
        "what": (
            "This tab generates a PRISMA 2020 flow diagram and checklist "
            "from the pipeline\u2019s output data. The flow diagram visualises "
            "the number of records identified, screened, assessed for "
            "eligibility, and included in the final synthesis\u2014the standard "
            "reporting format required by most peer-reviewed journals for "
            "systematic reviews. The checklist covers all 27 PRISMA 2020 "
            "items plus four PRISMA-trAIce extension items that document "
            "AI tool usage, roles, human oversight points, and performance "
            "metrics.\n\n"
            "The Export Bibliography section lets you export the entire "
            "corpus in three standard reference-manager formats: BibTeX "
            "(.bib) for LaTeX workflows, RIS (.ris) for Zotero/Mendeley/"
            "EndNote, and EndNote XML (.xml) for direct EndNote import. "
            "Each format captures title, authors, year, journal, DOI, "
            "and abstract where available.\n\n"
            "The Full Report Bundle button performs a one-click export of "
            "every artifact the pipeline has produced\u2014PRISMA diagrams, "
            "GRADE tables, forest and funnel plots, brain coordinates, "
            "bibliography, robustness report, and synthesis abstract\u2014"
            "into a single directory with a README explaining each file."
        ),
        "why": (
            "PRISMA (Preferred Reporting Items for Systematic Reviews and "
            "Meta-Analyses) compliance is a de facto requirement for "
            "publishing systematic reviews. The 2020 update introduced "
            "mandatory reporting of automation tools, making it directly "
            "relevant to AI-assisted pipelines like Sysys. The "
            "PRISMA-trAIce extension goes further by requiring transparent "
            "disclosure of which AI models were used, what decisions they "
            "influenced, where humans intervened, and how AI performance "
            "was validated. Generating the flow diagram and checklist "
            "automatically from pipeline data ensures accuracy\u2014the "
            "counts are pulled directly from each stage\u2019s output files "
            "rather than tallied by hand.\n\n"
            "Bibliography export in standard formats ensures that the "
            "corpus your pipeline processed can be imported into any "
            "reference manager for further annotation, sharing with "
            "collaborators, or inclusion in a manuscript. The report "
            "bundle centralises all outputs so that the complete evidence "
            "package can be archived, submitted as supplementary material, "
            "or handed off to a co-author who was not present during the "
            "pipeline run."
        ),
        "outputs": (
            "The flow diagram shows four tiers\u2014Identification, Screening, "
            "Eligibility, and Included\u2014with record counts and exclusion "
            "reasons at each stage. The checklist table lists each PRISMA "
            "item with its status (auto-filled or manual) and content. "
            "Items that the pipeline can populate automatically are marked "
            "\u2018auto\u2019; remaining items are marked \u2018manual\u2019 with editable "
            "cells so you can complete them before export. Export saves "
            "the flow diagram as PNG, SVG, or PDF and the checklist as "
            "Markdown or CSV.\n\n"
            "The bibliography export generates one file in the selected "
            "format containing every article in the best available corpus. "
            "Citation keys in BibTeX are generated from the first author\u2019s "
            "surname plus year (e.g., smith2023). The report bundle "
            "directory lists all included files in a README.md with the "
            "pipeline configuration that produced them."
        ),
        "reading": [
            {
                "text": "Page et al. (2021) \u2013 PRISMA 2020 statement",
                "url": "https://doi.org/10.1136/bmj.n71",
            },
            {
                "text": "PRISMA-trAIce \u2013 Transparent Reporting of AI in Evidence Synthesis",
                "url": "https://ai.jmir.org/2025/1/e80247",
            },
            {
                "text": "PRISMA 2020 Flow Diagram",
                "url": "http://www.prisma-statement.org/PRISMAStatement/FlowDiagram",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 10: Neuroimaging
    # ------------------------------------------------------------------
    "neuroimaging": {
        "title": "Neuroimaging Analysis",
        "icon": "\U0001f9e0",
        "what": (
            "This tab provides three neuroimaging-specific analysis tools. "
            "(1) Coordinate Extraction scans your full-text articles for "
            "brain activation coordinates reported in MNI or Talairach "
            "space\u2014using regex-based table parsing\u2014and exports them in "
            "NiMARE, GingerALE/Sleuth, or CSV format for downstream "
            "coordinate-based meta-analysis (CBMA). (2) NeuroQuery "
            "Cross-Reference queries the NeuroQuery predictive engine "
            "with each hypothesis component\u2019s keywords and compares the "
            "predicted activation map against the brain regions actually "
            "observed in your extracted claims, producing a convergence "
            "score. (3) Network Visualization renders an interactive "
            "force-directed graph of condition\u2013brain-region co-occurrence "
            "patterns extracted from the evidence corpus."
        ),
        "why": (
            "Text-based systematic reviews capture \u2018what regions are "
            "mentioned\u2019 but lose spatial precision. Coordinate-based "
            "meta-analysis (ALE, MKDA) leverages the exact stereotaxic "
            "locations reported in fMRI/PET studies to test whether "
            "activation peaks converge beyond chance. Exporting your "
            "corpus coordinates into standard CBMA formats bridges the "
            "gap between narrative synthesis and spatial statistics. "
            "The NeuroQuery convergence check adds an independent, "
            "data-driven prior: if the literature consistently implicates "
            "regions that NeuroQuery also predicts for your hypothesis "
            "terms, confidence in the spatial claim increases. The "
            "network graph visualises which configured constructs or "
            "populations share brain-region mentions, revealing hub regions "
            "and potential cross-domain pathway overlaps."
        ),
        "outputs": (
            "Coordinate Extraction displays a table of all extracted "
            "peaks (article, x, y, z, space, region label, statistic, "
            "sample size). Use the export buttons to save as NiMARE "
            "JSON (for Python-based CBMA), GingerALE Sleuth format "
            "(for Java-based ALE), or plain CSV. The NeuroQuery section "
            "shows a convergence table with one row per hypothesis "
            "component: the score (0\u20131) indicates the fraction of "
            "NeuroQuery-predicted regions that were actually observed "
            "in your corpus. Scores above 0.3 suggest meaningful spatial "
            "agreement. Select a row to see the matched, predicted-only, "
            "and observed-only region lists. The network graph colours "
            "condition nodes (circles) and brain-region nodes (structural "
            "= squares, functional = diamonds), with edge thickness "
            "proportional to co-occurrence count. Click a node to "
            "highlight its connections."
        ),
        "reading": [
            {
                "text": "Eickhoff et al. (2012) \u2013 ALE meta-analysis",
                "url": "https://doi.org/10.1016/j.neuroimage.2011.09.017",
            },
            {
                "text": "Dock\u00e8s et al. (2020) \u2013 NeuroQuery",
                "url": "https://doi.org/10.7554/eLife.53385",
            },
            {
                "text": "Salo et al. (2022) \u2013 NiMARE",
                "url": "https://doi.org/10.55458/neurolibre.00007",
            },
        ],
    },

    # ------------------------------------------------------------------
    # Tab 13: Log
    # ------------------------------------------------------------------
    "log": {
        "title": "Stage 11: Execution Log",
        "icon": "\U0001f4dd",
        "what": (
            "The log tab is a real-time, scrollable record of every action "
            "the pipeline performs: API calls, scoring results, extraction "
            "outputs, errors, and timing information. Log entries are "
            "timestamped and colour-coded by severity (INFO, WARNING, "
            "ERROR). You can save the log to a text file for reproducibility "
            "documentation."
        ),
        "why": (
            "A detailed audit trail is essential for reproducibility and "
            "debugging. If a stage fails or produces unexpected results, the "
            "log tells you exactly which API call returned an error, which "
            "article caused a parsing failure, or which LLM response was "
            "malformed. For published systematic reviews, archiving the "
            "execution log alongside your results allows reviewers to "
            "verify the computational steps."
        ),
        "outputs": (
            "Each log line shows a timestamp, severity level, and message. "
            "INFO entries (black text) record normal operations. WARNING "
            "entries (orange) flag non-fatal issues like a rate-limit retry. "
            "ERROR entries (red) indicate failures that may have skipped an "
            "article or aborted a stage. Use the \u2018Save Log\u2019 button to "
            "export the full log as a .txt file."
        ),
        "reading": [
            {
                "text": "Cochrane Handbook Ch. 1.5: The review process",
                "url": "https://training.cochrane.org/handbook/current/chapter-01#section-1-5",
            },
            {
                "text": "Nosek et al. (2015) \u2013 Promoting an open research culture",
                "url": "https://doi.org/10.1126/science.aab2374",
            },
        ],
    },
}


# ── Welcome dialog content ─────────────────────────────────────────────

WELCOME_CONTENT = {
    "title": "Welcome to Sysys",
    "overview": (
        "Sysys (Systematic Evidence Synthesis) is a desktop application "
        "that orchestrates a multi-stage research pipeline for assessing "
        "how well published academic literature supports a specific "
        "hypothesis. It combines automated literature searching, LLM-"
        "assisted relevance scoring, structured data extraction, evidence "
        "auditing, and robustness analysis into a single reproducible "
        "workflow\u2014all wrapped in a classic Windows 98 interface."
    ),
    "pipeline_stages": [
        "Search \u2013 Query multiple databases and de-duplicate results",
        "Score & Filter \u2013 LLM relevance scoring with span verification",
        "Extract & Validate \u2013 Structured claim extraction and study design classification",
        "Audit & Synthesize \u2013 Span similarity audit and narrative synthesis",
        "Fulltext & Interrater \u2013 Full-text retrieval and cross-model reliability",
        "Robustness & Verify \u2013 Bootstrap, split-half, sensitivity, and permutation tests",
        "Text Scaling \u2013 Wordfish ideological scaling and stylometric profiling",
        "Evidence Map \u2013 Article \u00d7 component evidence matrix with trend charts",
        "Results Dashboard \u2013 Final alignment scores per hypothesis component",
        "Neuroimaging \u2013 Coordinate extraction, NeuroQuery convergence, and network visualization",
        "Compare Runs \u2013 Side-by-side comparison of multiple pipeline executions",
        "Report & Export \u2013 PRISMA 2020 flow diagram, checklist, and AI transparency reporting",
        "Log \u2013 Real-time execution log with full audit trail",
    ],
    "getting_started": [
        "Open File \u2192 Settings (Ctrl+,) to configure your API keys and LLM provider.",
        "Edit your hypothesis components in the Search tab, then click \u2018Run Search\u2019.",
        "Work through the tabs left-to-right\u2014each stage builds on the previous one.",
        "Click the \u2018?\u2019 button in any tab to learn what that stage does and why it matters.",
        "Check the templates/ directory for pre-built configurations: "
        "abct_tech_ai_infosheet.yaml (ABCT AI subcommittee info-sheet), "
        "abct_ai_digital_mental_health.yaml (ABCT evidence brief), "
        "lead_developmental_cognition.yaml (developmental cognition starter), "
        "autism_ei_balance.yaml (archived autism/E-I example), "
        "generic_neuroscience.yaml (broader neuroscience), or "
        "blank_template.yaml (start from scratch). Load a template via "
        "File \u2192 Settings and the config file browser.",
    ],
}
