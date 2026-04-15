"""Reciprocal Rank Fusion (RRF) for combining ranked result lists."""

from dataclasses import replace
from typing import Dict, List

from backend.app.core.retriever.rag import RetrievalResult


def rrf_fusion(
    vector_results: List[RetrievalResult],
    bm25_results: List[RetrievalResult],
    k: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> List[RetrievalResult]:
    """Weighted Reciprocal Rank Fusion of two ranked lists.

    Formula
    -------
    score(d) = vector_weight / (rrf_k + rank_vector)
             + bm25_weight   / (rrf_k + rank_bm25)

    Equal weights (both 1.0) reproduce standard RRF.
    Increase ``vector_weight`` to favour semantic similarity;
    increase ``bm25_weight`` to favour exact keyword matches.

    Parameters
    ----------
    vector_results : list
        Results from the vector / embedding retriever, ranked by similarity.
    bm25_results : list
        Results from the BM25 keyword retriever, ranked by BM25 score.
    k : int
        Maximum number of fused results to return.
    rrf_k : int
        RRF smoothing constant (default 60, per the original paper).
        Larger values reduce the score gap between top and lower ranks.
    vector_weight : float
        Multiplicative weight for the vector retriever (default 1.0).
    bm25_weight : float
        Multiplicative weight for the BM25 retriever (default 1.0).

    Examples
    --------
    Favour semantic search (2:1 ratio)::

        rrf_fusion(vec, bm25, k=5, vector_weight=2.0, bm25_weight=1.0)

    Keyword-only (disable vector contribution)::

        rrf_fusion(vec, bm25, k=5, vector_weight=0.0, bm25_weight=1.0)
    """
    if not bm25_results:
        return vector_results[:k]
    if not vector_results:
        return bm25_results[:k]

    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, RetrievalResult] = {}

    for rank, r in enumerate(vector_results, 1):
        key = r.content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + vector_weight / (rrf_k + rank)
        doc_map[key] = r

    for rank, r in enumerate(bm25_results, 1):
        key = r.content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (rrf_k + rank)
        if key not in doc_map:
            doc_map[key] = r

    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return [replace(doc_map[key], score=rrf_scores[key]) for key in sorted_keys[:k]]
