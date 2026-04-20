"""LangGraph ReAct Agent — LLM-driven tool selection and multi-step reasoning.

Replaces the static ``_TOOL_MAP`` routing in ``LegalRouterAgent`` with a
true ReAct (Reasoning + Acting) loop powered by LangGraph:

  ┌──────────────┐
  │  preprocess   │  QueryPreprocessor (classification, outside the graph)
  └──────┬────────┘
         │
  ┌──────▼────────┐
  │    reason      │◄────────┐  LLM decides: call tool(s) or give final answer
  └──────┬────────┘         │
         │ tool_calls        │ loop back
   ┌─────▼──────┐    ┌──────┴────────┐
   │    act      │───►│   (implicit)  │  LangGraph ToolNode executes tools,
   │ (ToolNode)  │    │   appends     │  adds ToolMessage to state.messages,
   └─────────────┘    │   results     │  and routes back to ``reason``.
                      └───────────────┘
         │ no more tool_calls
  ┌──────▼────────┐
  │     END        │  Final AI message = the answer
  └───────────────┘

Public API
----------
  ``run_stream()``  — AsyncGenerator yielding SSE-ready dicts (same event
                      types as ``LegalRouterAgent`` + new ``thought`` /
                      ``observation`` events).
  ``run()``         — Non-streaming dict response.

Both signatures are **drop-in compatible** with ``LegalRouterAgent`` so
the API layer (``async_search.py``) needs only a dependency swap.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.app.core.agent.state import AgentState
from backend.app.core.agent.tools.base import AgentTool, agent_tool_to_langchain
from backend.app.core.agent.prompts.registry import PromptRegistry
from backend.app.core.agent.prompts.react_prompts import (
    build_react_system_prompt,
    sanitize_user_input,
)
from backend.app.core.context.context_manager import ContextManager
from backend.app.core.preprocessor.query_preprocessor import QueryPreprocessor
from backend.app.core.retriever.models import PreprocessResult

logger = logging.getLogger(__name__)

# ── Token budget constants ────────────────────────────────────────────────────
# Rough estimate: 1 token ≈ 1.5 Chinese characters
_CHARS_PER_TOKEN = 1.5
_DEFAULT_TOKEN_BUDGET = 20000  # conservative budget (model max is 30K+)


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimation from message character counts."""
    total_chars = sum(len(m.content) for m in messages if hasattr(m, "content"))
    return int(total_chars / _CHARS_PER_TOKEN)


class LegalReActAgent:
    """LangGraph-based ReAct agent for legal RAG.

    Drop-in replacement for ``LegalRouterAgent`` with the same
    ``run_stream()`` / ``run()`` interface.

    Parameters
    ----------
    preprocessor : QueryPreprocessor
        Classification + correction + metadata extraction.
    context_manager : ContextManager
        Multi-turn history + token budgeting.
    tools : dict[str, AgentTool]
        Named tools keyed by ``AgentTool.name``.
    prompt_registry : PromptRegistry | None
        Per-query-type system prompts (used for non-ReAct fallback;
        the ReAct loop uses its own prompt from ``react_prompts``).
    api_key, base_url, model
        DashScope / OpenAI-compatible LLM credentials.
    max_iterations : int
        Hard cap on reason→act loops to prevent runaway cost.
    temperature : float
        LLM sampling temperature (low = more deterministic tool calls).
    token_budget : int
        Estimated token budget for messages before compression kicks in.
    """

    def __init__(
        self,
        preprocessor: QueryPreprocessor,
        context_manager: ContextManager,
        tools: Dict[str, AgentTool],
        prompt_registry: Optional[PromptRegistry] = None,
        *,
        api_key: str,
        base_url: str,
        model: str = "qwen-plus",
        max_iterations: int = 5,
        temperature: float = 0.1,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self.preprocessor = preprocessor
        self.context_manager = context_manager
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.max_iterations = max_iterations
        self.token_budget = token_budget

        # ── Convert AgentTool → LangChain StructuredTool ───────────
        self.agent_tools = tools
        self.lc_tools = [
            agent_tool_to_langchain(t) for t in tools.values()
        ]
        self._tool_names = [t.name for t in self.lc_tools]

        # ── LLM with tool-calling capability ───────────────────────
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            streaming=True,
        )
        self.llm_with_tools = self.llm.bind_tools(self.lc_tools)

        # ── Build the LangGraph ────────────────────────────────────
        self.graph = self._build_graph()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Graph construction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_graph(self) -> StateGraph:
        """Construct and compile the ReAct agent graph.

        Nodes
        -----
        reason : LLM decides next action (tool call or final answer).
        act    : ``ToolNode`` executes the selected tools and appends
                 ``ToolMessage`` to ``state["messages"]``.

        Edges
        -----
        reason → act   if ``tool_calls`` present in the AI message
        reason → END   if no ``tool_calls`` (= final answer produced)
        act    → reason   loop back for next reasoning step
        """
        workflow = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", ToolNode(self.lc_tools))

        # ── Edges ──────────────────────────────────────────────────
        workflow.set_entry_point("reason")
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "act": "act",
                "end": END,
            },
        )
        workflow.add_edge("act", "reason")

        return workflow.compile()

    # ── Graph nodes ───────────────────────────────────────────────────────────

    async def _reason_node(self, state: AgentState) -> dict:
        """LLM reasoning node — produces tool_calls or final answer."""
        messages = list(state["messages"])

        # ── Token budget check: compress if approaching limit ────
        token_est = _estimate_tokens(messages)
        if token_est > self.token_budget:
            messages = self._compress_messages(messages)
            logger.info(
                "Compressed messages: %d → %d tokens (est)",
                token_est, _estimate_tokens(messages),
            )

        response = await self.llm_with_tools.ainvoke(messages)

        # Track tool call history for deduplication
        tool_call_history = list(state.get("tool_call_history", []))
        if (
            isinstance(response, AIMessage)
            and hasattr(response, "tool_calls")
            and response.tool_calls
        ):
            for tc in response.tool_calls:
                query_arg = tc.get("args", {}).get("query", "")
                tool_call_history.append((tc["name"], query_arg))

        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "tool_call_history": tool_call_history,
            "has_called_tool": state.get("has_called_tool", False) or bool(
                isinstance(response, AIMessage)
                and hasattr(response, "tool_calls")
                and response.tool_calls
            ),
            "total_tokens_est": _estimate_tokens(messages),
        }

    # ── Edge conditions ───────────────────────────────────────────────────────

    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue acting or finish.

        Returns ``"act"`` if the last AI message contains tool_calls,
        ``"end"`` otherwise (final answer ready).

        Enhanced with:
        - Max iteration guard
        - Tool call deduplication (same tool + similar query → force end)
        - Forced first-call check (must call at least one tool)
        """
        messages = state["messages"]
        last_message = messages[-1]
        iteration = state.get("iteration_count", 0)

        # Guard: max iterations
        if iteration >= self.max_iterations:
            logger.warning(
                "ReAct agent reached max iterations (%d) — forcing end",
                self.max_iterations,
            )
            return "end"

        # If the LLM wants to call tools, check for deduplication
        if (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            # Check for duplicate tool calls
            history = state.get("tool_call_history", [])
            if self._has_duplicate_calls(last_message.tool_calls, history):
                logger.warning(
                    "Detected duplicate tool call at iteration %d — forcing end",
                    iteration,
                )
                return "end"
            return "act"

        # No tool calls — check if we should force at least one tool call
        has_called = state.get("has_called_tool", False)
        if not has_called and iteration <= 1:
            # First iteration with no tool call: this is risky — the LLM
            # is trying to answer from memory. We let it through but log
            # a warning. The prompt should prevent this, but we don't
            # forcefully inject tool calls to avoid breaking the graph flow.
            logger.warning(
                "ReAct agent produced answer without any tool call "
                "(iteration=%d). Prompt may need tuning.",
                iteration,
            )

        return "end"

    def _has_duplicate_calls(
        self,
        new_calls: list[dict],
        history: list[tuple[str, str]],
    ) -> bool:
        """Check if the new tool calls duplicate recent history.

        A call is considered duplicate if the same tool was called with
        a query that has >60% character overlap with a previous call.
        """
        for tc in new_calls:
            name = tc["name"]
            query = tc.get("args", {}).get("query", "")
            for hist_name, hist_query in history:
                if name == hist_name and self._query_similarity(query, hist_query) > 0.6:
                    return True
        return False

    @staticmethod
    def _query_similarity(a: str, b: str) -> float:
        """Jaccard similarity at character level (fast, no tokenizer needed)."""
        if not a or not b:
            return 0.0
        set_a, set_b = set(a), set(b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    # ── Message compression ──────────────────────────────────────────────────

    def _compress_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Compress message list when approaching token budget.

        Strategy: keep system message + last 5 messages, summarise the rest.
        This is a synchronous heuristic compression (no LLM call) to avoid
        adding latency. For LLM-based compression, use ContextManager.
        """
        if len(messages) <= 6:
            return messages

        system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
        tail = messages[-5:]
        middle = messages[1:-5] if system_msg else messages[:-5]

        # Build a summary of the compressed messages
        summary_parts = []
        for msg in middle:
            if isinstance(msg, ToolMessage):
                # Heavily truncate tool results
                content = msg.content[:200] + "…" if len(msg.content) > 200 else msg.content
                summary_parts.append(f"[工具结果] {content}")
            elif isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                tools = [tc["name"] for tc in msg.tool_calls]
                summary_parts.append(f"[AI调用工具: {', '.join(tools)}]")
            elif isinstance(msg, AIMessage):
                content = msg.content[:150] + "…" if len(msg.content) > 150 else msg.content
                summary_parts.append(f"[AI回复] {content}")
            elif isinstance(msg, HumanMessage):
                summary_parts.append(f"[用户] {msg.content[:100]}")

        summary = SystemMessage(
            content="[以下是之前的对话摘要]\n" + "\n".join(summary_parts)
        )

        result = []
        if system_msg:
            result.append(system_msg)
        result.append(summary)
        result.extend(tail)
        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Streaming entry-point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def run_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        k: int = 5,
        enable_preprocessing: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Full ReAct agent pipeline, yielding SSE-ready dicts.

        Event types (superset of ``LegalRouterAgent``)
        -----------------------------------------------
        progress              stage hint (preprocessing / reasoning / ...)
        preprocessing_result  corrected query + classification
        tool_dispatch          which tools the LLM chose this iteration
        observation            summarised tool result (for frontend display)
        sources                extracted retrieval sources + stats
        chunk                  one LLM token (part of the final answer)
        done                   stream finished, with elapsed time
        error                  unrecoverable failure
        """
        t0 = time.perf_counter()

        # ── Step 1: Preprocessing (outside the graph) ──────────────
        pre = await self._preprocess(query, enable_preprocessing)
        yield {
            "type": "progress",
            "stage": "preprocessing",
            "text": "查询分析中...",
        }
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

        # ── Step 2: Build initial messages ─────────────────────────
        # Sanitize user input to prevent prompt injection
        safe_query = sanitize_user_input(query)
        messages = self._build_initial_messages(pre, safe_query, session_id)

        yield {
            "type": "progress",
            "stage": "reasoning",
            "text": "Agent 推理中...",
        }

        # ── Step 3: Run the ReAct graph with streaming events ──────
        initial_state: AgentState = {
            "messages": messages,
            "original_query": query,
            "corrected_query": pre.corrected_query,
            "query_type": pre.query_type,
            "complexity": pre.complexity,
            "preprocessing_data": {},
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "sources": [],
            "session_id": session_id,
            "k": k,
            "tool_call_history": [],
            "has_called_tool": False,
            "total_tokens_est": 0,
        }

        tools_used: list[str] = []
        all_sources: list[dict] = []
        final_answer_started = False

        try:
            async for event in self.graph.astream_events(
                initial_state, version="v2",
            ):
                kind = event["event"]

                # ── LLM finished a full response ───────────────────
                if kind == "on_chat_model_end":
                    output = event["data"].get("output")
                    if not output:
                        continue
                    # Check if this response includes tool_calls
                    if (
                        isinstance(output, AIMessage)
                        and hasattr(output, "tool_calls")
                        and output.tool_calls
                    ):
                        tc_names = [tc["name"] for tc in output.tool_calls]
                        tools_used.extend(tc_names)
                        yield {
                            "type": "tool_dispatch",
                            "tools": tc_names,
                        }
                        # Reset flag: next streaming tokens will be another round
                        final_answer_started = False

                # ── Tool execution completed ───────────────────────
                if kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    tool_name = event.get("name", "unknown_tool")
                    # Collect sources from the tool output
                    parsed = self._parse_sources_from_observation(
                        tool_name, str(output),
                    )
                    all_sources.extend(parsed)
                    # Emit truncated observation for frontend
                    obs_text = str(output)
                    if len(obs_text) > 600:
                        obs_text = obs_text[:600] + "……"
                    yield {
                        "type": "observation",
                        "tool": tool_name,
                        "text": obs_text,
                    }

                # ── LLM streaming tokens ──────────────────────────
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if not chunk:
                        continue

                    # Skip chunks that are tool_call fragments
                    if (
                        isinstance(chunk, AIMessageChunk)
                        and chunk.tool_call_chunks
                    ):
                        continue

                    content = chunk.content if hasattr(chunk, "content") else ""
                    if content:
                        if not final_answer_started:
                            final_answer_started = True
                            yield {
                                "type": "progress",
                                "stage": "generation",
                                "text": "正在生成回答...",
                            }
                        yield {"type": "chunk", "text": content}

        except Exception as exc:
            logger.error("ReAct agent streaming failed: %s", exc, exc_info=True)
            yield {"type": "error", "message": f"Agent 执行异常: {exc}"}
            return

        # ── Step 4: Emit sources ───────────────────────────────────
        yield {
            "type": "sources",
            "sources": all_sources,
            "stats": {
                "tools_used": tools_used,
                "total_iterations": len(set(tools_used)),
                "mode": "react",
            },
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        yield {"type": "done", "total_time_ms": round(elapsed_ms, 1)}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Non-streaming entry-point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
            etype = event["type"]
            if etype == "preprocessing_result":
                preprocessing_data = event["data"]
            elif etype == "sources":
                sources_data = event["sources"]
            elif etype == "chunk":
                answer_parts.append(event["text"])

        return {
            "question": query,
            "answer": "".join(answer_parts),
            "sources": sources_data,
            "preprocessing": preprocessing_data,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Internal helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _preprocess(
        self,
        query: str,
        enable: bool,
    ) -> PreprocessResult:
        """Run the QueryPreprocessor or build a default result."""
        if enable and self.preprocessor:
            return await self.preprocessor.process_async(
                query, enable_correction=True,
            )
        return PreprocessResult(
            original_query=query,
            corrected_query=query,
            query_type="legal_consultation",
            retrieval_type="general",
        )

    def _build_initial_messages(
        self,
        pre: PreprocessResult,
        query: str,
        session_id: Optional[str],
    ) -> list[BaseMessage]:
        """Construct the initial message list for the graph.

        Includes:
          1. ReAct system prompt (with classification hints)
          2. Conversation history from ContextManager (if available)
          3. The current user question
        """
        system_prompt = build_react_system_prompt(
            query_type=pre.query_type,
            complexity=pre.complexity,
        )
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

        # Inject multi-turn history
        if self.context_manager and session_id:
            try:
                history = self.context_manager._load_history(session_id)
                for msg in history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
            except Exception as exc:
                logger.warning("Failed to load history for %s: %s", session_id, exc)

        messages.append(HumanMessage(content=query))
        return messages

    # ── Source extraction from tool observations ──────────────────────────────

    _SOURCE_RE = re.compile(
        r"\[结果(\d+)\]\s*来源:\s*(.+?)\s*\|\s*类型:\s*(\S*)\s*\|\s*相关度:\s*(\S+)\n(.*?)(?=\n\n---|$)",
        re.DOTALL,
    )

    def _parse_sources_from_observation(
        self,
        tool_name: str,
        observation: str,
    ) -> list[dict]:
        """Extract structured source entries from a tool observation string.

        The observation format is defined by ``_format_observation()``
        in ``tools/base.py``.  We reverse-parse it to populate the
        ``sources`` panel in the frontend.
        """
        sources: list[dict] = []
        for m in self._SOURCE_RE.finditer(observation):
            idx, source, doc_type, score_str, content = m.groups()
            try:
                score = float(score_str)
            except (ValueError, TypeError):
                score = 0.0
            sources.append({
                "content": content.strip(),
                "metadata": {
                    "source": source.strip(),
                    "doc_type": doc_type.strip(),
                    "tool": tool_name,
                },
                "score": score,
            })
        return sources
