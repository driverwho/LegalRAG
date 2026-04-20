"""Legal Agent — intent-driven query routing and generation.

Two agent implementations are available:

- ``LegalRouterAgent`` (v2): Static ``_TOOL_MAP`` routing.
  Fast, deterministic, single-round retrieval.

- ``LegalReActAgent`` (v3): LangGraph ReAct loop.
  LLM-driven tool selection, multi-round reasoning, self-correction.

Both share the same ``run_stream()`` / ``run()`` interface.  The active
agent is selected via the ``AGENT_VERSION`` setting (default: ``"v3"``).

Public API::

    # v2 — static routing
    from backend.app.core.agent import LegalRouterAgent
    agent = LegalRouterAgent(preprocessor=..., chat_manager=..., ...)

    # v3 — ReAct loop
    from backend.app.core.agent import LegalReActAgent
    agent = LegalReActAgent(preprocessor=..., context_manager=..., ...)

    # Both share the same run_stream / run interface
    async for event in agent.run_stream(query, session_id):
        ...
"""

from backend.app.core.agent.router import LegalRouterAgent
from backend.app.core.agent.react_agent import LegalReActAgent
from backend.app.core.agent.state import AgentState
from backend.app.core.agent.prompts.registry import PromptRegistry
from backend.app.core.agent.tools.base import AgentTool, ToolResult, agent_tool_to_langchain

__all__ = [
    "LegalRouterAgent",
    "LegalReActAgent",
    "AgentState",
    "PromptRegistry",
    "AgentTool",
    "ToolResult",
    "agent_tool_to_langchain",
]
