"""LegalRouterAgent — orchestrates classification → tool selection → generation.

Replaces the previous flat pipeline where every query type ran the same
retrieval + the same system prompt.  Now:

  1. **Classify** — ``QueryPreprocessor`` identifies intent + complexity.
  2. **Route**   — tool-selection table maps ``query_type`` → one or more
                    ``AgentTool`` instances (law / case / both / none).
  3. **Execute** — simple queries run one tool; complex queries dispatch
                    tools in parallel and merge results with RRF.
  4. **Prompt**  — ``PromptRegistry`` provides a per-type system prompt
                    that is injected into the ``ContextManager``.
  5. **Generate** — ``AsyncContextualChatManager`` streams the LLM answer
                    with the chosen prompt and retrieved context.

The agent exposes ``run_stream`` (SSE generator) and ``run`` (single
response), matching the signatures the API layer already expects.

Agent upgrade path
------------------
  Phase 3: Replace the static ``_TOOL_MAP`` with an LLM ``tool_calls``
           decision (ReAct loop).  ``AgentTool.to_tool_def()`` already
           provides the OpenAI function schema.
  Phase 4: Add a Planner stage for ``complexity == "complex"`` that
           breaks the query into sub-tasks before tool dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.app.core.retriever.rag import RetrievalResult
from backend.app.core.retriever.models import PreprocessResult
from backend.app.core.retriever.fusion import rrf_fusion
from backend.app.core.agent.tools.base import AgentTool, ToolResult
from backend.app.core.agent.prompts.registry import PromptRegistry
from backend.app.core.context.context_manager import (
    ContextManager,
    _format_retrieval_block,
)
from backend.app.core.preprocessor.query_preprocessor import QueryPreprocessor
from backend.app.core.llm.async_chat import AsyncContextualChatManager

logger = logging.getLogger(__name__)


# ── Tool routing table ────────────────────────────────────────────────────────

_TOOL_MAP: Dict[str, List[str]] = {
    "simple_law_query":       ["law_search"],
    "concept_explanation":    ["law_search"],
    "procedure_consultation": ["law_search"],
    "case_retrieval":         ["case_search"],
    "case_analysis":          ["case_search"],
    "legal_consultation":     ["law_search", "case_search"],
    "mixed_law_case":         ["law_search", "case_search"],
    "comparative_analysis":   ["law_search", "case_search"],
    "document_generation":    [],                    # no retrieval — template only
}


class LegalRouterAgent:
    """Router Agent that dispatches queries to specialised tools.

    Parameters
    ----------
    preprocessor : QueryPreprocessor
        Classification + correction + metadata extraction.
    chat_manager : AsyncContextualChatManager
        LLM generation (streaming + non-streaming).
    context_manager : ContextManager
        Multi-turn history + token budgeting.
    tools : dict[str, AgentTool]
        Named tools, keyed by ``AgentTool.name``.
    prompt_registry : PromptRegistry | None
        Per-query-type system prompts; a default one is created if omitted.
    """

    def __init__(
        self,
        preprocessor: QueryPreprocessor,
        chat_manager: AsyncContextualChatManager,
        context_manager: ContextManager,
        tools: Dict[str, AgentTool],
        prompt_registry: Optional[PromptRegistry] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.chat_manager = chat_manager
        self.context_manager = context_manager
        self.tools = tools
        self.prompt_registry = prompt_registry or PromptRegistry()

    # ── Streaming entry-point ─────────────────────────────────────────────────

    async def run_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        k: int = 5,
        enable_preprocessing: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Full agent pipeline, yielding SSE-ready dicts.

        Yields
        ------
        {"type": "progress",             "stage": str, "text": str}
        {"type": "preprocessing_result", "data": {...}}
        {"type": "tool_dispatch",        "tools": [...]}
        {"type": "sources",              "sources": [...], "stats": {...}}
        {"type": "chunk",                "text": str}
        {"type": "done"}
        """

        t0 = time.perf_counter()

        # ── Step 1: Preprocessing ─────────────────────────────────────
        if enable_preprocessing and self.preprocessor:
            yield {"type": "progress", "stage": "preprocessing",
                   "text": "查询分析中..."}
            pre = await self.preprocessor.process_async(
                query, enable_correction=True,
            )
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
        else:
            pre = PreprocessResult(
                original_query=query,
                corrected_query=query,
                query_type="legal_consultation",
                retrieval_type="general",
            )

        search_query = pre.corrected_query
        strategy = pre.processing_strategy

        # ── Step 2: Tool selection ────────────────────────────────────
        tool_names = self._select_tools(pre.query_type, pre.complexity)
        selected_tools = [self.tools[n] for n in tool_names if n in self.tools]

        yield {"type": "tool_dispatch",
               "tools": [t.name for t in selected_tools]}

        # ── Step 3: Tool execution ────────────────────────────────────
        k_mult = strategy.get("k_multiplier", 1.0)
        fetch_k = max(k, int(k * k_mult))
        metadata_filter = (
            pre.extracted_metadata.get("filters")
            if pre.extracted_metadata
            else None
        )

        if selected_tools:
            yield {"type": "progress", "stage": "retrieval",
                   "text": "混合检索中..."}
            results, stats = await self._execute_tools(
                selected_tools, search_query, fetch_k, metadata_filter,
                parallel=(pre.complexity in ("medium", "complex")
                          or len(selected_tools) > 1),
            )
        else:
            results, stats = [], {}

        # ── Step 4: Optional rerank (strategy hint) ───────────────────
        if strategy.get("enable_rerank") and len(results) > k:
            results = results[:k]         # placeholder — wire reranker later

        # ── Step 5: Emit sources ──────────────────────────────────────
        final = results[:k]
        sources_data = [
            {"content": r.content, "metadata": r.metadata, "score": r.score}
            for r in final
        ]
        yield {
            "type": "sources",
            "sources": sources_data,
            "stats": stats,
        }

        # ── Step 6: Build prompt + generate ───────────────────────────
        yield {"type": "progress", "stage": "generation",
               "text": "正在生成回答..."}

        retrieval_contexts = [
            {"text": r.content, "source": r.metadata.get("source", "未知")}
            for r in final
        ]

        # Build the per-type system prompt
        retrieval_block = _format_retrieval_block(retrieval_contexts)
        system_prompt = self.prompt_registry.get(
            pre.query_type, retrieval_block,
        )

        # Stream generation with specialised prompt
        answer_parts: list[str] = []
        async for chunk in self._generate_stream(
            question=query,
            retrieval_contexts=retrieval_contexts,
            session_id=session_id,
            system_prompt=system_prompt,
        ):
            answer_parts.append(chunk)
            yield {"type": "chunk", "text": chunk}

        elapsed_ms = (time.perf_counter() - t0) * 1000

        yield {"type": "done", "total_time_ms": round(elapsed_ms, 1)}

        # ── Post-stream: persist conversation ─────────────────────────
        # (returned via yield so the API layer handles DB persistence)

    # ── Non-streaming entry-point ─────────────────────────────────────────────

    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        k: int = 5,
        enable_preprocessing: bool = True,
    ) -> Dict[str, Any]:
        """Non-streaming variant — returns a complete response dict."""
        sources_data: list = []
        answer_parts: list = []
        preprocessing_data: dict = {}

        async for event in self.run_stream(
            query, session_id, k, enable_preprocessing,
        ):
            if event["type"] == "preprocessing_result":
                preprocessing_data = event["data"]
            elif event["type"] == "sources":
                sources_data = event["sources"]
            elif event["type"] == "chunk":
                answer_parts.append(event["text"])

        return {
            "question": query,
            "answer": "".join(answer_parts),
            "sources": sources_data,
            "preprocessing": preprocessing_data,
        }

    # ── Tool selection ────────────────────────────────────────────────────────

    def _select_tools(
        self,
        query_type: str,
        complexity: str,
    ) -> List[str]:
        """Determine which tools to invoke based on classification.

        For complex queries that would normally use a single tool, we
        also add the other tool for richer context.
        """
        tools = list(_TOOL_MAP.get(query_type, ["law_search"]))

        # Complex queries: broaden search to both sources if not already
        if complexity == "complex" and len(tools) == 1:
            complement = "case_search" if tools[0] == "law_search" else "law_search"
            if complement in self.tools:
                tools.append(complement)

        return tools

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _execute_tools(
        self,
        tools: List[AgentTool],
        query: str,
        k: int,
        metadata_filter: Optional[Dict],
        parallel: bool = False,
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """Run tools (sequentially or in parallel) and merge results."""
        if parallel and len(tools) > 1:
            return await self._execute_parallel(tools, query, k, metadata_filter)
        return await self._execute_sequential(tools, query, k, metadata_filter)

    async def _execute_sequential(
        self,
        tools: List[AgentTool],
        query: str,
        k: int,
        metadata_filter: Optional[Dict],
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """Run tools one by one, concatenating results."""
        all_results: List[RetrievalResult] = []
        stats: Dict[str, Any] = {"tools_used": []}
        total_time = 0.0

        for tool in tools:
            try:
                tr = await tool.run_async(query, k=k, metadata_filter=metadata_filter)
                all_results.extend(tr.results)
                total_time += tr.search_time_ms
                stats["tools_used"].append({
                    "name": tr.tool_name,
                    "count": len(tr.results),
                    "time_ms": round(tr.search_time_ms, 1),
                })
            except Exception as exc:
                logger.error("Tool %s failed: %s", tool.name, exc)
                stats["tools_used"].append({
                    "name": tool.name,
                    "error": str(exc),
                })

        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)
        stats["total_search_time_ms"] = round(total_time, 1)
        return all_results, stats

    async def _execute_parallel(
        self,
        tools: List[AgentTool],
        query: str,
        k: int,
        metadata_filter: Optional[Dict],
    ) -> tuple[List[RetrievalResult], Dict[str, Any]]:
        """Run tools concurrently and fuse results."""
        tasks = [
            tool.run_async(query, k=k, metadata_filter=metadata_filter)
            for tool in tools
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: List[RetrievalResult] = []
        stats: Dict[str, Any] = {"tools_used": [], "parallel": True}
        max_time = 0.0

        result_groups: List[List[RetrievalResult]] = []

        for tool, outcome in zip(tools, raw):
            if isinstance(outcome, Exception):
                logger.error("Tool %s failed: %s", tool.name, outcome)
                stats["tools_used"].append({
                    "name": tool.name,
                    "error": str(outcome),
                })
                result_groups.append([])
            else:
                tr: ToolResult = outcome
                result_groups.append(tr.results)
                max_time = max(max_time, tr.search_time_ms)
                stats["tools_used"].append({
                    "name": tr.tool_name,
                    "count": len(tr.results),
                    "time_ms": round(tr.search_time_ms, 1),
                })

        # Fuse results from multiple tools using RRF
        if len(result_groups) == 2:
            all_results = rrf_fusion(
                result_groups[0], result_groups[1], k=k,
            )
        else:
            # Fallback: concatenate + sort
            for group in result_groups:
                all_results.extend(group)
            all_results.sort(key=lambda r: r.score, reverse=True)

        stats["total_search_time_ms"] = round(max_time, 1)
        stats["fusion_method"] = "RRF" if len(result_groups) == 2 else "concat"
        return all_results, stats

    # ── Generation ────────────────────────────────────────────────────────────

    async def _generate_stream(
        self,
        question: str,
        retrieval_contexts: List[Dict[str, Any]],
        session_id: Optional[str],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM generation with a custom system prompt.

        Uses ``ContextManager.build_context`` for multi-turn history,
        then overrides the system prompt with the agent-selected one.
        Delegates to ``chat_manager.stream_messages()`` — no direct
        access to ``chat_manager.client``.
        """
        if self.context_manager and session_id:
            built = self.context_manager.build_context(
                session_id=session_id,
                current_question=question,
                retrieval_contexts=retrieval_contexts,
            )
            # Override the system prompt with the agent-selected one;
            # keep the compressed history and current user message.
            messages = [{"role": "system", "content": system_prompt}] + built.messages
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

        try:
            async for chunk in self.chat_manager.stream_messages(messages):
                yield chunk
        except Exception as exc:
            logger.error("Agent generation streaming failed: %s", exc)
            yield f"\n\n[生成失败: {exc}]"
