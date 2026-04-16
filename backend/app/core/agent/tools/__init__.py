"""Agent tool implementations."""

from backend.app.core.agent.tools.base import (
    AgentTool,
    ToolResult,
    agent_tool_to_langchain,
)
from backend.app.core.agent.tools.law_search import LawSearchTool
from backend.app.core.agent.tools.case_search import CaseSearchTool

__all__ = [
    "AgentTool",
    "ToolResult",
    "agent_tool_to_langchain",
    "LawSearchTool",
    "CaseSearchTool",
]
