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
from backend.app.core.agent.prompts.react_prompts import build_react_system_prompt
from backend.app.core.context.context_manager import ContextManager
from backend.app.core.preprocessor.query_preprocessor import QueryPreprocessor
from backend.app.core.retriever.models import PreprocessResult

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.preprocessor = preprocessor
        self.context_manager = context_manager
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.max_iterations = max_iterations

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
        messages = state["messages"]
        response = await self.llm_with_tools.ainvoke(messages)
        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ── Edge conditions ───────────────────────────────────────────────────────

    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue acting or finish.

        Returns ``"act"`` if the last AI message contains tool_calls,
        ``"end"`` otherwise (final answer ready).
        """
        messages = state["messages"]
        last_message = messages[-1]

        # Guard: max iterations
        iteration = state.get("iteration_count", 0)
        if iteration >= self.max_iterations:
            logger.warning(
                "ReAct agent reached max iterations (%d) — forcing end",
                self.max_iterations,
            )
            return "end"

        # If the LLM wants to call tools, continue
        if (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            return "act"

        return "end"

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
        messages = self._build_initial_messages(pre, query, session_id)

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
            yield {"type": "error", "message": str(exc)}
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
