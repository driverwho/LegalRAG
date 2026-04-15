"""CaseSearchTool — retrieval scoped to judicial case chunks.

Searches the vector store (+ BM25 when available) for 案例-type
documents.  Uses the shared ``AsyncRAGPipeline`` stages so it gets
caching, RRF fusion, sliding-window expansion, and dedup for free.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from backend.app.core.agent.tools.base import AgentTool, ToolResult

logger = logging.getLogger(__name__)

# Must match the value stamped by DocumentClassifier / document_tasks.py
# (backend/app/core/document/classifier.py → ClassificationResult.doc_type)
_DOC_TYPE = "case"


class CaseSearchTool(AgentTool):
    """Search for judicial case chunks (案例)."""

    name = "case_search"
    description = "在司法判例知识库中检索案例、裁判文书，返回相关判例。"

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    async def run_async(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> ToolResult:
        t0 = time.perf_counter()

        # Merge caller filter with doc_type constraint.
        # _DOC_TYPE = "case" matches what DocumentClassifier writes to metadata.
        effective_filter = dict(metadata_filter or {})
        effective_filter.setdefault("doc_type", _DOC_TYPE)

        hybrid = await self._pipeline.search_hybrid_async(
            query=query,
            k=k,
            metadata_filter=effective_filter,
        )

        # Apply sliding-window expansion for case documents
        assembled = await self._pipeline.assemble_context_async(
            hybrid.results, query_type="案例",
        )
        final = await self._pipeline.deduplicate_async(assembled)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "CaseSearchTool '%.40s': %d results (%.0fms)",
            query, len(final), elapsed_ms,
        )
        return ToolResult(
            results=final,
            tool_name=self.name,
            search_time_ms=elapsed_ms,
            metadata={
                "vector_count": hybrid.vector_count,
                "bm25_count": hybrid.bm25_count,
                "fusion_method": hybrid.fusion_method,
            },
        )
