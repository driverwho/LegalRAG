"""Pipeline stage implementations as a mixin for AsyncRAGPipeline.

Each method corresponds to one numbered stage:

    Stage 1 — preprocess_query
    Stage 2 — search_hybrid_async  (+ private _vector_search_async, _bm25_search_async)
    Stage 3 — assemble_context_async  (+ _fetch_parent_chunks_async, _batch_get_by_ids,
                                         _sliding_window_async)
    Stage 4 — rerank_async
    Stage 5 — deduplicate_async  (+ _dedup_sync)

``PipelineStagesMixin`` is not meant to be instantiated directly; it is mixed
into ``AsyncRAGPipeline`` which provides the instance attributes
(``vector_store``, ``bm25_retriever``, ``reranker``, ``preprocessor``,
``similarity_threshold``, ``_cache``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backend.app.core.retriever.models import HybridSearchResult, PreprocessResult
from backend.app.core.retriever.rag import RetrievalResult
from backend.app.core.retriever.fusion import rrf_fusion

if TYPE_CHECKING:
    from backend.app.core.retriever.cache import SearchCache

logger = logging.getLogger(__name__)


class PipelineStagesMixin:
    """Mixin that provides Stage 1 – 5 logic for AsyncRAGPipeline."""

    # Declared here only for type-checkers; set by AsyncRAGPipeline.__init__
    vector_store: Any
    bm25_retriever: Any
    reranker: Any
    preprocessor: Any
    similarity_threshold: float
    _cache: SearchCache

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────

    async def preprocess_query(
        self,
        query: str,
        enable_correction: bool = False,
    ) -> PreprocessResult:
        """Classify, (optionally) correct, and extract metadata from a query.

        Falls back to a pass-through result if no preprocessor is wired.
        """
        if not self.preprocessor:
            return PreprocessResult(
                original_query=query,
                corrected_query=query,
                query_type="general",
                retrieval_type="general",
            )
        try:
            return await self.preprocessor.process_async(
                query, enable_correction=enable_correction
            )
        except Exception as exc:
            logger.warning("Preprocessing failed (%s) — using original query", exc)
            return PreprocessResult(
                original_query=query,
                corrected_query=query,
                query_type="general",
                retrieval_type="general",
            )

    # ── Stage 2: Hybrid retrieval ─────────────────────────────────────────────

    async def search_hybrid_async(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> HybridSearchResult:
        """Parallel vector + BM25 search fused with RRF.

        Both retrievers run concurrently via ``asyncio.gather``; a failure
        in one branch is logged and treated as an empty result so the
        other branch still contributes.
        """
        if use_cache:
            cached = self._cache.get(query, k, collection_name)
            if cached:
                logger.debug("Cache hit for query: %.50s", query)
                return cached

        t0 = time.perf_counter()

        # Build task list — vector search always present, BM25 optional
        coros = [self._vector_search_async(query, k, collection_name, metadata_filter)]
        has_bm25 = self.bm25_retriever is not None
        if has_bm25:
            coros.append(
                self._bm25_search_async(query, k, collection_name, metadata_filter)
            )

        raw = await asyncio.gather(*coros, return_exceptions=True)

        vector_results: List[RetrievalResult] = (
            raw[0] if not isinstance(raw[0], Exception) else []
        )
        bm25_results: List[RetrievalResult] = (
            raw[1] if has_bm25 and not isinstance(raw[1], Exception) else []
        )

        if isinstance(raw[0], Exception):
            logger.error("Vector search raised: %s", raw[0])
        if has_bm25 and isinstance(raw[1], Exception):
            logger.error("BM25 search raised: %s", raw[1])

        # ── Debug: dump top-5 content preview per retriever ──────────
        if vector_results:
            previews = [
                f"  [{i}] score={r.score:.4f} | {r.content[:50]}"
                for i, r in enumerate(vector_results[:5], 1)
            ]
            logger.debug(
                "Vector top-%d for '%.30s':\n%s",
                min(5, len(vector_results)), query, "\n".join(previews),
            )
        if bm25_results:
            previews = [
                f"  [{i}] score={r.score:.4f} | {r.content[:50]}"
                for i, r in enumerate(bm25_results[:5], 1)
            ]
            logger.debug(
                "BM25 top-%d for '%.30s':\n%s",
                min(5, len(bm25_results)), query, "\n".join(previews),
            )
        # ─────────────────────────────────────────────────────────────

        fused = rrf_fusion(
            vector_results,
            bm25_results,
            k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = HybridSearchResult(
            results=fused,
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            fusion_method="RRF" if bm25_results else "vector-only",
            search_time_ms=elapsed_ms,
        )

        if use_cache:
            self._cache.put(query, k, collection_name, result)

        logger.info(
            "Hybrid search '%.40s': vector=%d bm25=%d → fused=%d (%.0fms)",
            query,
            len(vector_results),
            len(bm25_results),
            len(fused),
            elapsed_ms,
        )
        return result

    async def _vector_search_async(
        self,
        query: str,
        k: int,
        collection_name: Optional[str],
        metadata_filter: Optional[Dict],
    ) -> List[RetrievalResult]:
        """Vector search — async wrapper around the sync ChromaVectorStore."""
        try:
            # ChromaDB is sync; offload to thread pool (one call, not a loop)
            if collection_name:
                raw = await asyncio.to_thread(
                    self.vector_store.search,
                    query, k, metadata_filter, collection_name,
                )
            else:
                raw = await asyncio.to_thread(
                    self.vector_store.search_all_collections,
                    query, k, metadata_filter,
                )
            # ChromaDB returns (Document, distance) tuples.  The distance
            # metric depends on the collection config — L2 distances can be
            # >> 1, so a fixed 0.5 threshold silently drops everything.
            #
            # Strategy: log raw distances for diagnostics, then convert to
            # similarity = 1/(1+distance) (works for any non-negative
            # distance metric) and filter by similarity_threshold.
            if raw:
                dist_preview = ", ".join(
                    f"{d:.4f}" for _, d in raw[:5]
                )
                logger.debug(
                    "Vector raw distances (top %d): [%s]  threshold=%.4f",
                    min(5, len(raw)), dist_preview, self.similarity_threshold,
                )

            results = []
            for doc, distance in raw:
                similarity = 1.0 / (1.0 + distance)
                if similarity >= self.similarity_threshold:
                    results.append(
                        RetrievalResult(
                            content=doc.page_content,
                            score=similarity,
                            metadata=doc.metadata,
                            source=doc.metadata.get("source", ""),
                        )
                    )
            return results
        except Exception as exc:
            logger.error("Vector search error: %s", exc)
            return []

    async def _bm25_search_async(
        self,
        query: str,
        k: int,
        collection_name: Optional[str],
        metadata_filter: Optional[Dict],
    ) -> List[RetrievalResult]:
        """BM25 search — CPU-bound, offloaded to thread pool."""
        try:
            raw = await asyncio.to_thread(
                self.bm25_retriever.search,
                query, k, collection_name, metadata_filter,
            )
            if not raw:
                return []

            # Normalize BM25 scores to (0, 1] so they are on the same
            # scale as vector similarity scores.  Use min-max normalisation
            # when there is a score spread; fall back to 1/(1+s) when all
            # scores are identical or there is only one result.
            raw_scores = [float(s) for _, s in raw]
            max_s = max(raw_scores)
            min_s = min(raw_scores)

            if max_s > min_s:
                # min-max → [0, 1], then shift into (0, 1]
                normed = [(s - min_s) / (max_s - min_s) for s in raw_scores]
            else:
                # All scores equal — map to a fixed similarity via 1/(1+d)
                normed = [1.0 / (1.0 + s) for s in raw_scores]

            return [
                RetrievalResult(
                    content=doc.page_content,
                    score=ns,
                    metadata=doc.metadata,
                    source=doc.metadata.get("source", ""),
                )
                for (doc, _), ns in zip(raw, normed)
            ]
        except Exception as exc:
            logger.error("BM25 search error: %s", exc)
            return []

    # ── Stage 3: Context assembly ─────────────────────────────────────────────

    async def assemble_context_async(
        self,
        results: List[RetrievalResult],
        query_type: str,
        window: int = 1,
    ) -> List[RetrievalResult]:
        """Route context assembly by document type.

        Args:
            results:    Fused retrieval results from Stage 2.
            query_type: ``"法条"`` → parent-chunk expansion;
                        ``"案例"`` → sliding-window expansion (±window chunks).
            window:     Half-width of the sliding window (案例 only, default 1).
        """
        if query_type == "法条":
            return await self._fetch_parent_chunks_async(results)
        if query_type == "案例":
            return await self._sliding_window_async(results, window=window)
        return results

    async def _fetch_parent_chunks_async(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Batch-fetch parent chunks to replace child sub-chunks (法条).

        Supports both ``parent_doc_id`` (current ingest schema) and the legacy
        ``parent_id`` field so existing collections continue to work.
        """
        def _pid(r: RetrievalResult) -> Optional[str]:
            return r.metadata.get("parent_doc_id") or r.metadata.get("parent_id")

        parent_ids = list({_pid(r) for r in results if _pid(r)})
        if not parent_ids:
            return results

        try:
            # Batch query — single thread hop instead of N
            parent_docs: List[Dict[str, Any]] = await asyncio.to_thread(
                self._batch_get_by_ids, parent_ids
            )
            parent_map = {d["id"]: d for d in parent_docs}

            assembled: List[RetrievalResult] = []
            for r in results:
                pid = r.metadata.get("parent_doc_id") or r.metadata.get("parent_id")
                if pid and pid in parent_map:
                    p = parent_map[pid]
                    assembled.append(
                        RetrievalResult(
                            content=p["content"],
                            score=r.score,
                            metadata=p["metadata"],
                            source=r.source,
                        )
                    )
                else:
                    assembled.append(r)
            return assembled
        except Exception as exc:
            logger.error("Parent chunk fetch failed: %s", exc)
            return results

    def _batch_get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Sync helper: bulk-get documents from Chroma by ID list.

        Uses the native Chroma ``collection.get(ids=...)`` batch API
        instead of looping N individual queries.
        """
        try:
            client = self.vector_store._get_client()
            col_name = self.vector_store.collection_name
            if not self.vector_store._collection_exists(col_name):
                return []
            collection = client.get_collection(name=col_name)
            data = collection.get(ids=ids, include=["documents", "metadatas"])
            return [
                {"id": doc_id, "content": doc, "metadata": meta}
                for doc_id, doc, meta in zip(
                    data["ids"], data["documents"], data["metadatas"]
                )
            ]
        except Exception as exc:
            logger.error("Batch get by IDs failed: %s", exc)
            return []

    async def _sliding_window_async(
        self,
        results: List[RetrievalResult],
        window: int = 1,
    ) -> List[RetrievalResult]:
        """Expand each result chunk with ±window adjacent chunks (案例).

        For each retrieved chunk, fetches the *window* chunks immediately
        before and after it (within the same parent document) and inserts
        them into the result list in index order.  The original chunk keeps
        its relevance score; neighbours inherit the same score.

        Requires metadata fields on every chunk:
            ``parent_doc_id`` — ID of the parent document
            ``chunk_index``   — integer position within that parent
        Chunks missing either field are passed through unchanged.
        """
        if not results:
            return results

        # ── Collect which (parent_doc_id, chunk_index) pairs we need ─────────
        # parent_to_need: {parent_doc_id → set of chunk_index values to fetch}
        # origin_map:     {(parent_doc_id, chunk_index) → original RetrievalResult}
        parent_to_need: Dict[str, set] = {}
        passthrough: List[RetrievalResult] = []   # chunks without required fields

        for r in results:
            pid = r.metadata.get("parent_doc_id")
            cidx = r.metadata.get("chunk_index")
            if pid is None or cidx is None:
                passthrough.append(r)
                continue

            if pid not in parent_to_need:
                parent_to_need[pid] = set()
            for delta in range(-window, window + 1):
                parent_to_need[pid].add(cidx + delta)

        if not parent_to_need:
            return results   # nothing has the required fields

        # ── Batch-fetch neighbour chunks from ChromaDB ────────────────────────
        try:
            neighbour_docs: List[Dict[str, Any]] = await asyncio.to_thread(
                self._batch_get_neighbors, parent_to_need
            )
        except Exception as exc:
            logger.error("Sliding window fetch failed: %s", exc)
            return results

        # Build lookup: (parent_doc_id, chunk_index) → doc dict
        neighbour_map: Dict[tuple, Dict[str, Any]] = {}
        for doc in neighbour_docs:
            npid = doc["metadata"].get("parent_doc_id")
            ncidx = doc["metadata"].get("chunk_index")
            if npid is not None and ncidx is not None:
                neighbour_map[(npid, ncidx)] = doc

        # ── Assemble in index order, deduplicate by (pid, cidx) key ──────────
        assembled: List[RetrievalResult] = []
        seen: set = set()

        for r in results:
            pid = r.metadata.get("parent_doc_id")
            cidx = r.metadata.get("chunk_index")
            if pid is None or cidx is None:
                continue   # handled via passthrough below

            for idx in range(cidx - window, cidx + window + 1):
                key = (pid, idx)
                if key in seen:
                    continue
                seen.add(key)

                if idx == cidx:
                    # Original chunk — keep as-is with its own score
                    assembled.append(r)
                elif key in neighbour_map:
                    doc = neighbour_map[key]
                    assembled.append(
                        RetrievalResult(
                            content=doc["content"],
                            score=r.score,          # inherit parent score
                            metadata=doc["metadata"],
                            source=r.source,
                        )
                    )

        assembled.extend(passthrough)

        logger.debug(
            "Sliding window (w=%d): %d → %d chunks",
            window, len(results), len(assembled),
        )
        return assembled

    def _batch_get_neighbors(
        self,
        parent_to_need: Dict[str, set],
    ) -> List[Dict[str, Any]]:
        """Sync: bulk-fetch chunks by ``parent_doc_id`` + ``chunk_index`` set.

        Issues one Chroma ``collection.get(where=...)`` per parent document so
        the total number of round-trips equals the number of unique parents,
        not the number of chunks.
        """
        try:
            client = self.vector_store._get_client()
            col_name = self.vector_store.collection_name
            if not self.vector_store._collection_exists(col_name):
                return []
            collection = client.get_collection(name=col_name)

            docs: List[Dict[str, Any]] = []
            for pid, indices in parent_to_need.items():
                idx_list = sorted(int(i) for i in indices if i >= 0)
                if not idx_list:
                    continue

                # Filter: parent_doc_id == pid AND chunk_index IN idx_list
                if len(idx_list) == 1:
                    where: Dict[str, Any] = {
                        "$and": [
                            {"parent_doc_id": {"$eq": pid}},
                            {"chunk_index": {"$eq": idx_list[0]}},
                        ]
                    }
                else:
                    where = {
                        "$and": [
                            {"parent_doc_id": {"$eq": pid}},
                            {"chunk_index": {"$in": idx_list}},
                        ]
                    }

                data = collection.get(
                    where=where,
                    include=["documents", "metadatas"],
                )
                for doc_id, doc, meta in zip(
                    data["ids"], data["documents"], data["metadatas"]
                ):
                    docs.append({"id": doc_id, "content": doc, "metadata": meta})

            return docs
        except Exception as exc:
            logger.error("Batch get neighbors failed: %s", exc)
            return []

    # ── Stage 4: Reranking ────────────────────────────────────────────────────

    async def rerank_async(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Optional async reranker — skipped when no reranker is wired."""
        if not self.reranker or len(results) <= top_k:
            return results[:top_k]
        try:
            reranked = await self.reranker.rerank_async(
                query=query,
                documents=[r.content for r in results],
                top_k=top_k,
            )
            return [replace(results[idx], score=score) for idx, score in reranked]
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            return results[:top_k]

    # ── Stage 5: Deduplication ────────────────────────────────────────────────

    async def deduplicate_async(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Async dedup — offloads CPU work to thread pool."""
        return await asyncio.to_thread(self._dedup_sync, results)

    @staticmethod
    def _dedup_sync(results: List[RetrievalResult]) -> List[RetrievalResult]:
        seen: set = set()
        unique: List[RetrievalResult] = []
        for r in results:
            h = hashlib.md5(r.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(r)
        return unique
