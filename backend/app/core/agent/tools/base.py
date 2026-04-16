"""AgentTool — abstract base class for router-agent tools.

Each tool wraps a single, well-scoped retrieval capability:
  - LawSearchTool   → vector + BM25 search filtered to 法条
  - CaseSearchTool  → vector + BM25 search filtered to 案例
  - HybridSearchTool → cross-collection general search

The Router Agent picks one or more tools based on the classifier
output, runs them (possibly in parallel), and merges the results.

LangChain bridge
----------------
  ``agent_tool_to_langchain()`` wraps any ``AgentTool`` subclass into
  a LangChain ``StructuredTool`` for use inside a LangGraph ReAct
  agent.  The same ``AgentTool`` instance works in both the legacy
  static router *and* the new ReAct pipeline — no code duplication.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.app.core.retriever.rag import RetrievalResult

logger = logging.getLogger(__name__)

# ── Max chars per result & per observation (token budget control) ──────────────
_CONTENT_LIMIT = 800
_MAX_RESULTS_IN_OBSERVATION = 8


# ── Pydantic input schema for LangChain tool ──────────────────────────────────

class SearchInput(BaseModel):
    """Input schema shared by all search tools."""

    query: str = Field(description="搜索查询关键词或自然语言问题")
    k: int = Field(default=5, description="返回结果数量，默认为5")


@dataclass
class ToolResult:
    """Unified output of any AgentTool invocation."""

    results: List[RetrievalResult]
    tool_name: str
    search_time_ms: float = 0.0
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentTool(ABC):
    """Abstract base for a retrieval tool usable by the Router Agent.

    Subclasses must implement ``run_async``.  Everything else has
    sensible defaults.
    """

    # ── Identity (set by subclass) ────────────────────────────────────

    name: str = ""
    description: str = ""

    # ── Public API ────────────────────────────────────────────────────

    @abstractmethod
    async def run_async(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> ToolResult:
        """Execute the tool and return results."""
        ...

    # ── OpenAI tool_definition (legacy — kept for backward compat) ────

    def to_tool_def(self) -> Dict[str, Any]:
        """Serialize this tool as an OpenAI-compatible function definition.

        .. deprecated::
           Prefer ``agent_tool_to_langchain()`` for the LangGraph ReAct
           agent.  This method is retained for any code that still uses
           raw OpenAI function-calling dicts.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


# ── LangChain bridge ─────────────────────────────────────────────────────────

def _format_observation(tool_name: str, result: ToolResult) -> str:
    """Serialize a ToolResult into a structured text observation.

    The LangGraph ToolNode expects each tool to return a **string**.
    We format results so the LLM can reference them by number when
    composing its final answer (the ``【n】`` citation convention).
    """
    if not result.results:
        return f"[{tool_name}] 未找到相关结果。建议尝试不同的关键词重新检索。"

    items = result.results[:_MAX_RESULTS_IN_OBSERVATION]
    parts: list[str] = []
    for i, r in enumerate(items, 1):
        source = r.metadata.get("source", "未知来源")
        doc_type = r.metadata.get("doc_type", "")
        score = f"{r.score:.3f}" if r.score else "N/A"
        content = r.content[:_CONTENT_LIMIT]
        if len(r.content) > _CONTENT_LIMIT:
            content += "……"
        parts.append(
            f"[结果{i}] 来源: {source} | 类型: {doc_type} | 相关度: {score}\n"
            f"{content}"
        )

    header = (
        f"[{tool_name}] 共找到 {len(result.results)} 条结果"
        f"（返回前 {len(items)} 条，"
        f"耗时 {result.search_time_ms:.0f}ms）:\n\n"
    )
    return header + "\n\n---\n\n".join(parts)


def agent_tool_to_langchain(tool: AgentTool) -> StructuredTool:
    """Bridge an existing AgentTool to a LangChain StructuredTool.

    The returned tool has:
      - ``name`` / ``description`` from the AgentTool
      - ``args_schema`` = ``SearchInput`` (query + k)
      - An async entrypoint (``coroutine``) that calls ``run_async``
        and formats the result as a citation-friendly text block.

    This preserves backward compatibility — the same ``AgentTool``
    instances work in both the old static router and the new ReAct
    agent without any subclass changes.
    """

    async def _arun(query: str, k: int = 5) -> str:
        try:
            tr = await tool.run_async(query, k=k)
            return _format_observation(tool.name, tr)
        except Exception as exc:
            logger.error("LangChain bridge: tool %s failed: %s", tool.name, exc)
            return f"[{tool.name}] 工具执行失败: {exc}"

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=SearchInput,
        coroutine=_arun,
        func=None,          # no sync fallback — async-only
    )
