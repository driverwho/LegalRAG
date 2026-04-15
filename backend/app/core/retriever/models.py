"""Data-transfer objects shared across the async RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from backend.app.core.retriever.rag import RetrievalResult


# ── Stage 1 output ────────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """Output of the query preprocessing stage.

    Attributes
    ----------
    original_query : str
        The raw user input before any processing.
    corrected_query : str
        Query after spell-correction / normalisation.
    query_type : str
        Fine-grained type produced by QueryClassifier
        (e.g. ``"simple_law_query"``, ``"case_retrieval"``, ``"legal_consultation"``).
    retrieval_type : str
        Coarse category used by the retrieval pipeline:
        ``"法条"`` | ``"案例"`` | ``"general"``.
    extracted_metadata : dict
        Structured filters (region, year_range, court_level, …)
        produced by MetadataExtractor.
    confidence : float
        0.0 – 1.0 confidence of the classification.
    complexity : str
        ``"simple"`` | ``"medium"`` | ``"complex"`` — guides downstream budget.
    processing_strategy : dict
        Pipeline hints from the classifier (search mode, k, rerank flag, …).
    """

    original_query: str
    corrected_query: str
    query_type: str = "general"
    retrieval_type: str = "general"                  # "法条" | "案例" | "general"
    extracted_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    complexity: str = "simple"
    processing_strategy: Dict[str, Any] = field(default_factory=dict)


# ── Stage 2 output ────────────────────────────────────────────────────────────

@dataclass
class HybridSearchResult:
    """Fused retrieval result with provenance stats."""

    results: List[RetrievalResult]
    vector_count: int
    bm25_count: int
    fusion_method: str
    search_time_ms: float
