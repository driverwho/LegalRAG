"""AgentTool — abstract base class for router-agent tools.

Each tool wraps a single, well-scoped retrieval capability:
  - LawSearchTool   → vector + BM25 search filtered to 法条
  - CaseSearchTool  → vector + BM25 search filtered to 案例
  - HybridSearchTool → cross-collection general search

The Router Agent picks one or more tools based on the classifier
output, runs them (possibly in parallel), and merges the results.

Agent upgrade path
------------------
  When upgrading to an LLM-driven ReAct loop, each tool becomes an
  OpenAI-compatible ``tool_definition`` dict.  The ``to_tool_def()``
  method provides that mapping so the transition is one-line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.app.core.retriever.rag import RetrievalResult


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

    # ── OpenAI tool_definition (ReAct upgrade path) ───────────────────

    def to_tool_def(self) -> Dict[str, Any]:
        """Serialize this tool as an OpenAI-compatible function definition.

        Used when the Router Agent transitions to an LLM-driven ReAct
        loop where the model selects tools via ``tool_calls``.
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
