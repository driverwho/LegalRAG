"""LangGraph Agent State — the shared memory across all graph nodes.

LangGraph requires TypedDict-based state definitions (not dataclasses)
so that the ``add_messages`` reducer can merge message lists automatically.

State fields
------------
messages           Conversation messages (system + human + AI + tool).
                   Uses ``add_messages`` reducer — LangGraph appends
                   rather than replacing.
original_query     The raw user input (preserved for DB persistence).
corrected_query    After spell-correction by QueryPreprocessor.
query_type         Fine-grained intent from QueryClassifier.
complexity         ``"simple"`` | ``"medium"`` | ``"complex"``.
preprocessing_data Full preprocessing payload (forwarded to SSE).
iteration_count    How many reason→act loops have run so far.
max_iterations     Hard cap to prevent infinite loops.
sources            Extracted retrieval sources for the frontend panel.
session_id         Multi-turn session identifier.
k                  Number of retrieval results per tool call.
tool_call_history  List of (tool_name, query) tuples from past iterations
                   — used for deduplication and forced-first-call logic.
total_tokens_est   Running estimate of total tokens consumed by messages
                   (rough: 1 token ≈ 1.5 Chinese characters).
has_called_tool    Whether at least one tool has been called in this run.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through every node in the ReAct graph.

    ``total=False`` makes all fields optional at construction time so
    callers only need to provide what they have.  The graph nodes are
    responsible for populating the rest.
    """

    # ── Core conversation (reducer: append, not replace) ───────────
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Preprocessing metadata ─────────────────────────────────────
    original_query: str
    corrected_query: str
    query_type: str
    complexity: str
    preprocessing_data: Dict[str, Any]

    # ── Tool execution tracking ────────────────────────────────────
    sources: List[Dict[str, Any]]
    tool_call_history: List[Tuple[str, str]]  # [(tool_name, query), ...]
    has_called_tool: bool

    # ── Control flow ───────────────────────────────────────────────
    iteration_count: int
    max_iterations: int
    total_tokens_est: int  # rough token estimate for budget control

    # ── Session context ────────────────────────────────────────────
    session_id: Optional[str]
    k: int
