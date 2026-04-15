"""Async RAG pipeline — parallel retrieval, streaming progress, zero threads.

Module layout
-------------
models.py   — PreprocessResult, HybridSearchResult  (data classes)
cache.py    — SearchCache  (bounded LRU + TTL)
fusion.py   — rrf_fusion()  (Reciprocal Rank Fusion algorithm)
stages.py   — PipelineStagesMixin  (Stage 1 – 5 implementations)
async_rag.py — AsyncRAGPipeline  (public façade, wires everything together)
"""

from typing import Any, AsyncGenerator, Dict, Optional

from backend.app.core.retriever.cache import SearchCache
from backend.app.core.retriever.models import HybridSearchResult, PreprocessResult
from backend.app.core.retriever.stages import PipelineStagesMixin

# Re-export so callers that currently import from async_rag keep working
__all__ = [
    "AsyncRAGPipeline",
    "PreprocessResult",
    "HybridSearchResult",
]


class AsyncRAGPipeline(PipelineStagesMixin):
    """Fully async RAG pipeline.

    Stage map
    ---------
    1. preprocess_query        — classify / correct / extract metadata
    2. search_hybrid_async     — parallel vector + BM25 → RRF fusion
    3. assemble_context_async  — parent-chunk fetch or sliding-window expand
    4. rerank_async            — optional LLM reranker
    5. deduplicate_async       — hash dedup (CPU, offloaded)

    The public entry-point ``search_stream`` chains all five stages and
    yields SSE-ready dicts so the API layer stays thin.

    Note: this pipeline owns *retrieval only*.  LLM generation lives in
    AsyncContextualChatManager so the two concerns stay independent.
    """

    def __init__(
        self,
        vector_store,
        bm25_retriever=None,
        reranker=None,
        preprocessor=None,
        similarity_threshold: float = 0.5,
        max_results: int = 10,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.preprocessor = preprocessor
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        # RRF fusion weights — increase vector_weight for semantic queries,
        # increase bm25_weight for exact-match / keyword queries.
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Bounded TTL cache (see cache.py)
        self._cache = SearchCache(maxsize=512, ttl=300.0)

    # ── Full pipeline with streaming progress ─────────────────────────────────

    async def search_stream(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        enable_preprocessing: bool = True,
        enable_rerank: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run all five retrieval stages, yielding SSE-ready dicts.

        The API layer iterates this and forwards events straight to the
        client — no business logic in the endpoint.

        Yields
        ------
        {"type": "progress", "stage": str, "text": str}
        {"type": "preprocessing_result", "data": {...}}
        {"type": "sources", "sources": [...], "stats": {...}}
        """
        # ── Stage 1 ──────────────────────────────────────────────────
        strategy: Dict[str, Any] = {}
        if enable_preprocessing and self.preprocessor:
            yield {"type": "progress", "stage": "preprocessing", "text": "查询分析中..."}
            pre = await self.preprocess_query(query, enable_correction=True)
            yield {
                "type": "preprocessing_result",
                "data": {
                    "original": pre.original_query,
                    "corrected": pre.corrected_query,
                    "query_type": pre.query_type,
                    "retrieval_type": pre.retrieval_type,
                    "complexity": pre.complexity,
                    "metadata": pre.extracted_metadata,
                },
            }
            search_query = pre.corrected_query
            query_type = pre.retrieval_type        # coarse: 法条 | 案例 | general
            strategy = pre.processing_strategy
        else:
            search_query = query
            query_type = "general"

        # ── Stage 2 ──────────────────────────────────────────────────
        yield {"type": "progress", "stage": "retrieval", "text": "混合检索中..."}
        k_mult = strategy.get("k_multiplier", 1.0)
        fetch_k = max(k, int(k * 2 * k_mult))     # extra budget for rerank/dedup
        hybrid = await self.search_hybrid_async(
            query=search_query,
            k=fetch_k,
            collection_name=collection_name,
        )

        # ── Stage 3 ──────────────────────────────────────────────────
        yield {"type": "progress", "stage": "assembly", "text": "上下文组装中..."}
        assembled = await self.assemble_context_async(hybrid.results, query_type)

        # ── Stage 4 ──────────────────────────────────────────────────
        should_rerank = enable_rerank or strategy.get("enable_rerank", False)
        if should_rerank and self.reranker:
            yield {"type": "progress", "stage": "reranking", "text": "结果重排序中..."}
            assembled = await self.rerank_async(assembled, search_query, top_k=k)
        else:
            assembled = assembled[:k]

        # ── Stage 5 ──────────────────────────────────────────────────
        final = await self.deduplicate_async(assembled)

        yield {
            "type": "sources",
            "sources": [
                {"content": r.content, "metadata": r.metadata, "score": r.score}
                for r in final
            ],
            "stats": {
                "vector_count": hybrid.vector_count,
                "bm25_count": hybrid.bm25_count,
                "fusion_method": hybrid.fusion_method,
                "search_time_ms": round(hybrid.search_time_ms, 1),
            },
        }
