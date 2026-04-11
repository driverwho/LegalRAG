"""Context management with token-aware window control for multi-turn RAG.

Manages a three-part context structure:
  - system:   system prompt (selected by prompt template routing)
  - messages: conversation history + current user message
  - tools:    available tool definitions (reserved for future expansion)

When total tokens exceed the available input budget
(CONTEXT_WINDOW_SIZE − CONTEXT_RESERVED_OUTPUT_TOKENS), the manager
triggers an automatic compression pass that:
  1. Protects the most recent 2 rounds of conversation.
  2. Preserves any already-summarised messages (summary=True).
  3. Compresses the remaining history via a compact LLM prompt.
  4. Replaces the compressed messages with a single summary message
     tagged ``summary: true``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.app.core.database.session_service import SessionService
from backend.app.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rough token estimation
# ---------------------------------------------------------------------------

# Chinese text: ~1.5 tokens/char on average for most LLMs.
# English text: ~4 chars/token.  We use a blended heuristic.
_CHARS_PER_TOKEN = 2


def estimate_tokens(text: str) -> int:
    """Rough token count for a string (Chinese-heavy content)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total token count for a list of message dicts.

    Each message adds ~4 overhead tokens for role/formatting.
    """
    total = 0
    for msg in messages:
        total += 4  # role + structural overhead
        total += estimate_tokens(msg.get("content", ""))
    return total


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Template registry — keyed by template name.
# ``select_system_prompt`` picks one based on the current question +
# whether retrieval contexts are available.

PROMPT_TEMPLATES: Dict[str, str] = {
    "rag": (
        "你是一名专业的中国法律助手，服务于法律 RAG 知识库系统。\n"
        "该知识库涵盖中国全部现行有效的中央法律法规、地方性法规，以及近年来的司法判例。\n"
        "你的职责包括：法律条文精准解答、案例检索与分析、法律概念释义，以及合规建议。\n\n"
        "{retrieval_block}\n\n"
        "回答原则：\n"
        "- 严格依据检索到的法律条文和判例内容作答，不得编造或臆测法律规定；\n"
        "- 引用法条时注明具体法律名称、条款编号，在相关句后用【编号】标注引用来源；\n"
        "- 涉及判例时说明案件要旨、裁判观点及其参考价值；\n"
        "- 当检索内容不足以回答时，明确告知用户并建议咨询专业律师；\n"
        "- 区分法律建议与法律信息：你提供的是法律信息参考，不构成正式法律意见；\n"
        "- 如果用户问题涉及多轮对话上下文，结合历史对话理解问题意图。"
    ),
    "general": (
        "你是一名专业的中国法律助手，服务于法律 RAG 知识库系统。\n"
        "该知识库涵盖中国全部现行有效的中央法律法规、地方性法规，以及近年来的司法判例。\n\n"
        "用户当前的问题在知识库中未检索到直接相关的法条或判例。\n"
        "请基于你的法律知识尝试回答，并友好地提示用户：该回答未引用知识库中的具体文档，"
        "建议进一步核实或咨询专业律师。\n"
        "如果用户问题涉及多轮对话上下文，结合历史对话理解问题意图。"
    ),
}

# The compact prompt used to ask the LLM to summarise old messages.
COMPACT_PROMPT = (
    "请将以下多轮对话历史压缩为一段简洁的摘要。\n"
    "要求：\n"
    "- 保留关键事实、结论和用户意图；\n"
    "- 去除冗余的寒暄和重复内容；\n"
    "- 使用第三人称描述；\n"
    "- 控制在 200 字以内。\n\n"
    "对话历史：\n{conversation}"
)


def _format_retrieval_block(contexts: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a numbered block."""
    if not contexts:
        return ""
    parts = []
    for idx, ctx in enumerate(contexts, 1):
        source = ctx.get("source", "未知")
        text = ctx.get("text", "")
        parts.append(f"[{idx}] 来源: {source}\n内容: {text}")
    return "检索到的文档内容：\n" + "\n\n---\n\n".join(parts)


def select_system_prompt(
    retrieval_contexts: List[Dict[str, Any]],
) -> str:
    """Pick the right system prompt template based on retrieval results."""
    if retrieval_contexts:
        block = _format_retrieval_block(retrieval_contexts)
        return PROMPT_TEMPLATES["rag"].format(retrieval_block=block)
    return PROMPT_TEMPLATES["general"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContextConfig:
    """Configuration — populated from Settings / .env."""

    context_window_size: int = 30000
    reserved_output_tokens: int = 2000
    protected_rounds: int = 2

    # LLM used for the compact summarisation call.
    compact_api_key: str = ""
    compact_base_url: str = ""
    compact_model: str = ""

    @classmethod
    def from_settings(cls) -> "ContextConfig":
        """Build a ContextConfig from the application Settings."""
        s = get_settings()
        return cls(
            context_window_size=s.CONTEXT_WINDOW_SIZE,
            reserved_output_tokens=s.CONTEXT_RESERVED_OUTPUT_TOKENS,
            protected_rounds=s.CONTEXT_PROTECTED_ROUNDS,
            compact_api_key=s.DASHSCOPE_API_KEY,
            compact_base_url=s.LLM_BASE_URL,
            compact_model=s.COMPACT_LLM_MODEL or s.LLM_MODEL,
        )


@dataclass
class BuiltContext:
    """The fully assembled context ready for an LLM call.

    Three sections mirror the OpenAI chat-completion API:
      * system   — a single system message list
      * messages — user/assistant conversation turns
      * tools    — tool definitions (empty list for now)
    """

    system: List[Dict[str, str]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)

    # Diagnostic info
    total_tokens: int = 0
    was_compressed: bool = False

    def to_llm_messages(self) -> List[Dict[str, str]]:
        """Flatten into the ``messages`` list expected by the OpenAI API."""
        return self.system + self.messages


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """Token-aware context window manager.

    Usage::

        ctx = context_manager.build_context(
            session_id="...",
            current_question="它有什么构成要件？",
            retrieval_contexts=[...],
        )
        # ctx.to_llm_messages()  → ready for openai chat.completions.create
    """

    def __init__(
        self,
        session_service: SessionService,
        config: Optional[ContextConfig] = None,
    ):
        self.session_service = session_service
        self.config = config or ContextConfig()
        self._compact_client: Optional[OpenAI] = None

    # ---- public API -------------------------------------------------------

    def build_context(
        self,
        session_id: str,
        current_question: str,
        retrieval_contexts: List[Dict[str, Any]],
    ) -> BuiltContext:
        """Assemble the full context, compressing history when necessary.

        Returns a ``BuiltContext`` with system / messages / tools sections.
        """
        # 1. System prompt
        system_content = select_system_prompt(retrieval_contexts)
        system_msgs: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # 2. History messages from DB
        raw_history = self._load_history(session_id)

        # 3. Current user message
        current_msg: Dict[str, str] = {"role": "user", "content": current_question}

        # 4. Tools (placeholder)
        tools: List[Dict[str, Any]] = []

        # 5. Token budget check
        available = self.config.context_window_size - self.config.reserved_output_tokens
        system_tokens = estimate_messages_tokens(system_msgs)
        current_tokens = estimate_messages_tokens([current_msg])
        tools_tokens = estimate_messages_tokens(tools)

        history_budget = available - system_tokens - current_tokens - tools_tokens
        was_compressed = False

        if history_budget < 0:
            # System + current already overflows — send with no history.
            history_msgs: List[Dict[str, Any]] = []
        else:
            history_tokens = estimate_messages_tokens(raw_history)
            if history_tokens > history_budget:
                raw_history = self._compress(raw_history, history_budget, session_id)
                was_compressed = True
            history_msgs = raw_history

        # Assemble
        all_messages = history_msgs + [current_msg]
        total = system_tokens + estimate_messages_tokens(all_messages) + tools_tokens

        return BuiltContext(
            system=system_msgs,
            messages=all_messages,
            tools=tools,
            total_tokens=total,
            was_compressed=was_compressed,
        )

    # ---- history loading --------------------------------------------------

    def _load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation history from the database.

        Returns a list of message dicts.  Messages that were previously
        compressed carry ``"summary": True``.
        """
        db_messages = self.session_service.get_messages(session_id)
        if not db_messages:
            return []

        formatted: List[Dict[str, Any]] = []
        for msg in db_messages:
            entry: Dict[str, Any] = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            # Propagate the summary flag if it was saved previously.
            if msg.get("summary"):
                entry["summary"] = True
            formatted.append(entry)
        return formatted

    # ---- compression ------------------------------------------------------

    def _compress(
        self, history: List[Dict[str, Any]], budget: int, session_id: str
    ) -> List[Dict[str, Any]]:
        """Compress history to fit within *budget* tokens.

        Strategy:
          1. Split history into *protected* (last N rounds) and *compressible*.
          2. Among compressible messages, skip those already flagged
             ``summary: True`` — they stay as-is.
          3. Summarise the remaining compressible messages via the compact
             LLM prompt.
          4. Persist the summary to DB (replace compressed messages).
          5. Return: [summary_msg, ...kept_summaries, ...protected]
        """
        protected, rest = self._split_protected(history)

        already_summarised: List[Dict[str, Any]] = []
        to_compress: List[Dict[str, Any]] = []
        for msg in rest:
            if msg.get("summary"):
                already_summarised.append(msg)
            else:
                to_compress.append(msg)

        if not to_compress:
            combined = already_summarised + protected
            return self._truncate_to_budget(combined, budget)

        # Summarise the compressible block.
        summary_text = self._call_compact_llm(to_compress)

        # Persist: replace the compressed messages in DB with a summary row.
        compressed_ids = [m["id"] for m in to_compress if m.get("id")]
        if compressed_ids:
            try:
                saved = self.session_service.replace_messages_with_summary(
                    session_id=session_id,
                    message_ids=compressed_ids,
                    summary_content=summary_text,
                )
                summary_msg: Dict[str, Any] = {
                    "id": saved["id"],
                    "role": "assistant",
                    "content": saved["content"],
                    "summary": True,
                }
            except Exception as exc:
                logger.warning("Failed to persist summary to DB: %s", exc)
                summary_msg = {
                    "role": "assistant",
                    "content": f"[历史对话摘要]\n{summary_text}",
                    "summary": True,
                }
        else:
            summary_msg = {
                "role": "assistant",
                "content": f"[历史对话摘要]\n{summary_text}",
                "summary": True,
            }

        result = [summary_msg] + already_summarised + protected
        return self._truncate_to_budget(result, budget)

    def _split_protected(self, history: List[Dict[str, Any]]) -> tuple:
        """Split history into (protected_tail, compressible_head).

        Protected = the last ``config.protected_rounds`` conversation rounds.
        A round = one user message + one assistant message (2 messages).
        """
        n_protected_msgs = self.config.protected_rounds * 2
        if len(history) <= n_protected_msgs:
            # Everything is protected — nothing to compress.
            return history, []

        protected = history[-n_protected_msgs:]
        rest = history[:-n_protected_msgs]
        return protected, rest

    def _truncate_to_budget(
        self, messages: List[Dict[str, Any]], budget: int
    ) -> List[Dict[str, Any]]:
        """Hard-truncate from the front if messages still exceed budget."""
        while messages and estimate_messages_tokens(messages) > budget:
            removed = messages.pop(0)
            logger.debug(
                "Hard-truncated message (role=%s, len=%d) to fit budget",
                removed.get("role"),
                len(removed.get("content", "")),
            )
        return messages

    # ---- compact LLM call -------------------------------------------------

    def _get_compact_client(self) -> OpenAI:
        """Lazy-initialise the OpenAI client used for compaction."""
        if self._compact_client is None:
            self._compact_client = OpenAI(
                api_key=self.config.compact_api_key,
                base_url=self.config.compact_base_url,
            )
        return self._compact_client

    def _call_compact_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Call the LLM with the compact prompt to summarise messages."""
        # Format messages into a readable conversation block.
        conversation_lines = []
        for msg in messages:
            role_label = "用户" if msg["role"] == "user" else "助手"
            content = msg.get("content", "")
            # Truncate extremely long individual messages before sending.
            if len(content) > 2000:
                content = content[:2000] + "..."
            conversation_lines.append(f"{role_label}: {content}")

        conversation_text = "\n".join(conversation_lines)
        prompt = COMPACT_PROMPT.format(conversation=conversation_text)

        try:
            client = self._get_compact_client()
            completion = client.chat.completions.create(
                model=self.config.compact_model,
                messages=[
                    {"role": "system", "content": "你是一个对话摘要助手。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.3,
            )
            summary = completion.choices[0].message.content
            logger.info(
                "Compressed %d messages into summary (%d chars)",
                len(messages),
                len(summary),
            )
            return summary
        except Exception as exc:
            logger.warning("Compact LLM call failed: %s — using fallback", exc)
            return self._fallback_summarise(messages)

    @staticmethod
    def _fallback_summarise(messages: List[Dict[str, Any]]) -> str:
        """CPU-only fallback when the compact LLM is unavailable."""
        topics: List[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                text = msg.get("content", "")[:40]
                topics.append(text)
        if topics:
            return "之前讨论了：" + "；".join(topics[:5]) + "。"
        return "之前有若干轮对话。"
