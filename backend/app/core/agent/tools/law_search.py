"""LawSearchTool — retrieval scoped to statute / regulation chunks.

Searches the vector store (+ BM25 when available) for 法条-type
documents.  Uses the shared ``AsyncRAGPipeline`` stages so it gets
caching, RRF fusion, parent-chunk expansion, and dedup for free.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from backend.app.core.agent.tools.base import AgentTool, ToolResult

logger = logging.getLogger(__name__)

# Must match the value stamped by DocumentClassifier / document_tasks.py
# (backend/app/core/document/classifier.py → ClassificationResult.doc_type)
_DOC_TYPE = "law"


class LawSearchTool(AgentTool):
    """Search for statute / regulation chunks (法条)."""

    name = "law_search"
    description = (
        "在法律法规知识库中检索法条、条文，返回相关法律规定。\n"
        "适用场景：查找具体法条、法律定义、构成要件、法律程序规定、"
        "法律概念解释。\n"
        "不适用场景：查找具体案例或判决（应使用 case_search）。\n"
        "参数：query 为搜索关键词（建议包含法律领域名称，如"
        "\"劳动合同法 解除条件\"），k 为返回结果数量（1-20，默认5）。"
    )

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
        # _DOC_TYPE = "law" matches what DocumentClassifier writes to metadata.
        effective_filter = dict(metadata_filter or {})
        effective_filter.setdefault("doc_type", _DOC_TYPE)

        hybrid = await self._pipeline.search_hybrid_async(
            query=query,
            k=k,
            metadata_filter=effective_filter,
        )

        # Apply parent-chunk expansion for law articles
        assembled = await self._pipeline.assemble_context_async(
            hybrid.results, query_type="法条",
        )
        final = await self._pipeline.deduplicate_async(assembled)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "LawSearchTool '%.40s': %d results (%.0fms)",
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
