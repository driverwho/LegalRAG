"""Legal Router Agent — intent-driven query routing and generation.

Public API::

    from backend.app.core.agent import LegalRouterAgent, PromptRegistry

    agent = LegalRouterAgent(
        preprocessor=...,
        chat_manager=...,
        context_manager=...,
        tools={"law_search": law_tool, "case_search": case_tool},
    )
    async for event in agent.run_stream(query, session_id):
        ...
"""

from backend.app.core.agent.router import LegalRouterAgent
from backend.app.core.agent.prompts.registry import PromptRegistry
from backend.app.core.agent.tools.base import AgentTool, ToolResult

__all__ = [
    "LegalRouterAgent",
    "PromptRegistry",
    "AgentTool",
    "ToolResult",
]
