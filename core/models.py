"""
Data models for the Hypothesis Consensus Analyzer.

Contains enumerations and dataclasses used throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class EvidenceDirection(Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    TANGENTIAL = "tangential"


class SourceType(Enum):
    PUBMED = "PubMed"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    WEB_OF_SCIENCE = "Web of Science"
    SCOPUS = "Scopus"
    ELICIT = "Elicit"
    LOCAL_PDF = "Local PDF"


class SpanStatus(Enum):
    VERIFIED = "verified"
    CLOSE_MATCH = "close_match"
    NOT_FOUND = "not_found"
    UNCHECKED = "unchecked"


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    ANTHROPIC = "anthropic"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class PipelineState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    SCORING = auto()
    FULLTEXT_ACQUIRE = auto()
    EXTRACTING = auto()
    VALIDATING = auto()
    AUDITING = auto()
    SYNTHESIZING = auto()
    FULLTEXT_PROCESS = auto()
    INTERRATING = auto()
    ROBUSTNESS = auto()
    VERIFYING = auto()
    COMPARING = auto()
    ERROR = auto()
    CANCELLED = auto()

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class HypothesisComponent:
    id: str
    label: str
    description: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class Article:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    source: SourceType = SourceType.PUBMED
    citations: int = 0
    url: str = ""
    pmid: str = ""
    keywords: list[str] = field(default_factory=list)
    has_full_text: bool = False
    open_access: bool = False
    full_text_url: str = ""

    @property
    def text_content(self) -> str:
        return self.full_text if self.full_text else self.abstract


@dataclass
class QuoteSpan:
    text: str
    component_id: str = ""
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    relevance_note: str = ""
    status: SpanStatus = SpanStatus.UNCHECKED
    matched_text: str = ""
    match_similarity: float = 0.0
    char_offset_start: int = -1
    char_offset_end: int = -1


@dataclass
class LLMScoringResult:
    article_id: str
    component_id: str
    llm_relevance: float = 0.0
    llm_direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    llm_reasoning: str = ""
    quote_spans: list[QuoteSpan] = field(default_factory=list)
    model_name: str = ""
    provider: LLMProvider = LLMProvider.OPENAI
    spans_verified: int = 0
    spans_close: int = 0
    spans_not_found: int = 0
    spans_total: int = 0
    verification_score: float = 0.0

    @property
    def is_reliable(self) -> bool:
        if self.spans_total == 0:
            return False
        return (self.spans_verified + self.spans_close) / self.spans_total >= 0.6


@dataclass
class EvidenceLink:
    article_id: str
    component_id: str
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    relevance_score: float = 0.0
    confidence: float = 0.0
    key_excerpts: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    llm_result: Optional[LLMScoringResult] = None

    @property
    def combined_score(self) -> float:
        if self.llm_result and self.llm_result.is_reliable:
            return 0.4 * self.relevance_score + 0.6 * self.llm_result.llm_relevance
        return self.relevance_score

    @property
    def combined_direction(self) -> EvidenceDirection:
        if self.llm_result and self.llm_result.is_reliable:
            return self.llm_result.llm_direction
        return self.direction


@dataclass
class AlignmentResult:
    component_id: str
    component_label: str
    support_score: float = 0.0
    evidence_count: int = 0
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_count: int = 0
    confidence: float = 0.0
    top_supporting: list[str] = field(default_factory=list)
    top_contradicting: list[str] = field(default_factory=list)


@dataclass
class TrendPoint:
    year: int
    article_count: int = 0
    avg_support_score: float = 0.0
    cumulative_citations: int = 0
    component_id: str = ""


@dataclass
class AnalysisSession:
    hypothesis_components: list[HypothesisComponent] = field(default_factory=list)
    articles: dict[str, Article] = field(default_factory=dict)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    alignment_results: list[AlignmentResult] = field(default_factory=list)
    trend_data: list[TrendPoint] = field(default_factory=list)
    total_articles_fetched: int = 0
    total_articles_analyzed: int = 0
    llm_results: dict[str, LLMScoringResult] = field(default_factory=dict)
    flagged_spans: list[QuoteSpan] = field(default_factory=list)
    corpus_json_path: str = ""
    fulltext_retrieval_results: list = field(default_factory=list)
    wordfish_model: object = None
    stylometry_profiles: dict = field(default_factory=dict)
    corpus_statistics: dict = field(default_factory=dict)
    pipeline_state: PipelineState = PipelineState.IDLE
    run_id: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    raw_search_counts: dict[str, int] = field(default_factory=dict)
    synthesis_clusters: list[dict[str, Any]] = field(default_factory=list)
    synthesis_abstract: str = ""
