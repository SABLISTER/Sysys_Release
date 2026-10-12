"""Query generation for each academic search provider.

Builds Boolean, fielded, or plain-text queries from the research config's
``search`` block (base_terms + domains), falling back to sensible defaults
when no config is provided.
"""

# ---------------------------------------------------------------------------
# Defaults — used when no research config is supplied
# ---------------------------------------------------------------------------
_DEFAULT_BASE_TERMS: list[str] = [
    "academic research",
    "scholarly literature",
    "peer reviewed study",
    "evidence synthesis",
]

_DEFAULT_DOMAINS: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_base_terms(search_config: dict | None) -> list[str]:
    # Copy before returning: callers treat the result as their own and some
    # mutate it. Returning the module-level default directly let a single
    # caller's ``.append()`` permanently corrupt the defaults for every
    # subsequent query built in the same process.
    if search_config:
        return list(search_config.get("base_terms", _DEFAULT_BASE_TERMS))
    return list(_DEFAULT_BASE_TERMS)


def _get_domains(search_config: dict | None) -> list[dict]:
    if search_config:
        return list(search_config.get("domains", _DEFAULT_DOMAINS))
    return list(_DEFAULT_DOMAINS)


# ---------------------------------------------------------------------------
# Natural-language query (for Elicit / general use)
# ---------------------------------------------------------------------------
def get_natural_language_query(research_config: dict | None = None) -> str:
    """Build a plain-English research question from the config.

    Falls back to a default hypothesis statement if no config is given.
    """
    if research_config:
        relevance = research_config.get("relevance")
        if not isinstance(relevance, dict):
            relevance = {}
        for key, source in (
            ("natural_language_question", research_config),
            ("research_question", relevance),
            ("hypothesis_text", research_config),
            ("hypothesis", research_config),
            ("question", research_config),
            ("description", research_config),
        ):
            value = source.get(key, "")
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
    return "What does the scholarly literature report for this research question?"


# ---------------------------------------------------------------------------
# Semantic Scholar queries (plain text)
# ---------------------------------------------------------------------------
def _s2_base(base_terms: list[str]) -> str:
    return " OR ".join(base_terms)


def _s2_and(terms: list[str]) -> str:
    return " ".join(terms)


def get_s2_queries(search_config: dict | None = None) -> list[str]:
    """Generate Semantic Scholar query strings — one per domain."""
    base_terms = _get_base_terms(search_config)
    domains = _get_domains(search_config)
    base = _s2_base(base_terms)

    queries = []
    for d in domains:
        domain_part = _s2_and(d.get("terms", []))
        if domain_part:
            require_also = d.get("require_also")
            if require_also:
                domain_part += " " + _s2_and(require_also)
            queries.append(f"{base} {domain_part}")
    if not queries:
        queries.append(base)
    return queries


# ---------------------------------------------------------------------------
# PubMed queries (Boolean / E-Utilities syntax)
# ---------------------------------------------------------------------------
def _pm_base(base_terms: list[str]) -> str:
    parts = [f'"{t}"[Title/Abstract]' for t in base_terms]
    return "(" + " OR ".join(parts) + ")"


def _pm_or(terms: list[str], field_tag: str = "[Title/Abstract]") -> str:
    parts = [f'"{t}"{field_tag}' for t in terms]
    return "(" + " OR ".join(parts) + ")"


def get_pubmed_queries(search_config: dict | None = None) -> list[str]:
    """Generate PubMed Boolean query strings — one per domain."""
    base_terms = _get_base_terms(search_config)
    domains = _get_domains(search_config)
    base = _pm_base(base_terms)

    queries = []
    for d in domains:
        terms = d.get("terms", [])
        if not terms:
            continue
        domain_part = _pm_or(terms)
        q = f"{base} AND {domain_part}"
        require_also = d.get("require_also")
        if require_also:
            q += f" AND {_pm_or(require_also)}"
        queries.append(q)
    if not queries:
        queries.append(base)
    return queries


# ---------------------------------------------------------------------------
# Web of Science queries (TS= fielded search)
# ---------------------------------------------------------------------------
def _wos_clause(terms: list[str]) -> str:
    parts = [f'TS="{t}"' for t in terms]
    return "(" + " OR ".join(parts) + ")"


def get_wos_queries(search_config: dict | None = None) -> list[str]:
    """Generate Web of Science query strings — one per domain."""
    base_terms = _get_base_terms(search_config)
    domains = _get_domains(search_config)
    base = _wos_clause(base_terms)

    queries = []
    for d in domains:
        terms = d.get("terms", [])
        if not terms:
            continue
        domain_part = _wos_clause(terms)
        q = f"{base} AND {domain_part}"
        require_also = d.get("require_also")
        if require_also:
            q += f" AND {_wos_clause(require_also)}"
        queries.append(q)
    if not queries:
        queries.append(base)
    return queries


# ---------------------------------------------------------------------------
# Scopus queries (TITLE-ABS-KEY fielded search)
# ---------------------------------------------------------------------------
def _scopus_clause(terms: list[str]) -> str:
    parts = [f'TITLE-ABS-KEY("{t}")' for t in terms]
    return "(" + " OR ".join(parts) + ")"


def get_scopus_queries(search_config: dict | None = None) -> list[str]:
    """Generate Scopus query strings — one per domain."""
    base_terms = _get_base_terms(search_config)
    domains = _get_domains(search_config)
    base = _scopus_clause(base_terms)

    queries = []
    for d in domains:
        terms = d.get("terms", [])
        if not terms:
            continue
        domain_part = _scopus_clause(terms)
        q = f"{base} AND {domain_part}"
        require_also = d.get("require_also")
        if require_also:
            q += f" AND {_scopus_clause(require_also)}"
        queries.append(q)
    if not queries:
        queries.append(base)
    return queries


# ---------------------------------------------------------------------------
# OpenAlex filters
# ---------------------------------------------------------------------------
def get_openalex_filters(search_config: dict | None = None) -> list[dict]:
    """Generate OpenAlex filter dicts — one per domain.

    Each dict has keys: ``search`` (text query) and ``filter`` (API filter string).
    """
    base_terms = _get_base_terms(search_config)
    domains = _get_domains(search_config)
    base = " ".join(base_terms[:2])  # keep it short for OpenAlex text search

    filters = []
    for d in domains:
        terms = d.get("terms", [])
        search_text = f"{base} {' '.join(terms[:3])}" if terms else base
        require_also = d.get("require_also")
        if require_also:
            search_text += " " + " ".join(require_also[:3])
        filters.append({
            "search": search_text,
            "filter": "has_abstract:true",
        })
    if not filters:
        filters.append({"search": base, "filter": "has_abstract:true"})
    return filters

