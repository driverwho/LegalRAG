"""Extended ChatManager with context management support."""

import logging
from typing import List, Dict, Any, Optional, Iterator

from openai import OpenAI

from backend.app.exceptions.handlers import LLMError
from backend.app.core.context.context_manager import ContextManager

logger = logging.getLogger(__name__)


class ContextualChatManager:
    """Chat manager with conversation context support.

    Unlike the base ChatManager which only uses retrieved contexts,
    this version also incorporates conversation history for multi-turn RAG.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        context_manager: Optional[ContextManager] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.context_manager = context_manager

    def generate_rag_response(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate an answer with optional conversation context.

        Args:
            question: User's question
            contexts: Retrieved documents from vector store
            session_id: Optional session ID for loading conversation history

        Returns:
            Generated answer text
        """
        if not contexts:
            return self._generate_fallback(question, session_id)
        return self._generate_with_context(question, contexts, session_id)

    def _generate_with_context(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> str:
        """Generate answer with both retrieval and conversation context."""
        # Build messages using context manager if available
        if self.context_manager and session_id:
            built = self.context_manager.build_context(
                session_id=session_id,
                current_question=question,
                retrieval_contexts=contexts,
            )
            messages = built.to_llm_messages()
        else:
            # Fallback to base implementation (no conversation history)
            messages = self._build_basic_messages(question, contexts)

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            answer = completion.choices[0].message.content
            logger.info("Contextual RAG response generated successfully")
            return answer
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise LLMError(f"Generation failed: {exc}") from exc

    def _generate_fallback(
        self, question: str, session_id: Optional[str] = None
    ) -> str:
        """Generate answer using general knowledge with optional history."""
        logger.info("No contexts available — using general knowledge fallback")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个智能助手。用户的问题在知识库中未找到相关信息，"
                    "请利用你的通用知识尝试回答，并友好地告知用户该信息可能不包含在上传的文档中。"
                ),
            },
        ]

        # Add conversation history if available
        if self.context_manager and session_id:
            history = self.context_manager._load_history(session_id)
            messages.extend(history)

        messages.append({"role": "user", "content": question})

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return completion.choices[0].message.content
        except Exception as exc:
            logger.error("Fallback generation failed: %s", exc)
            raise LLMError(f"Fallback generation failed: {exc}") from exc

    def _build_basic_messages(
        self, question: str, contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Build basic messages without conversation history (base implementation)."""
        context_parts = []
        for idx, ctx in enumerate(contexts, 1):
            source = ctx.get("source", "未知")
            text = ctx.get("text", "")
            context_parts.append(f"[{idx}] 来源: {source}\n内容: {text}")

        combined = "\n\n---\n\n".join(context_parts)

        return [
            {
                "role": "system",
                "content": (
                    "你是一个中文 RAG 助手。\n"
                    "原则：\n"
                    "- 仅依据提供的上下文作答，不得编造；\n"
                    "- 当上下文不足或无关时，明确说明无法回答并指出缺失信息类型；\n"
                    "- 语言简洁准确，尽量提炼关键结论与要点；\n"
                    "- 检索到的上下文和用户问的问题没关系的时候，忽略上下文！"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"上下文（按序编号）：\n{combined}\n\n"
                    "请：\n"
                    "1) 仅基于上述上下文回答；\n"
                    "2) 在相关句后用【编号】注明引用；\n"
                    "3) 若无法回答，说明缺少的上下文类型（例如：定义、步骤、数据）。"
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Streaming variants
    # ------------------------------------------------------------------

    def generate_rag_response_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Streaming version of generate_rag_response.

        Yields text chunks as they arrive from the LLM.
        """
        if not contexts:
            yield from self._generate_fallback_stream(question, session_id)
        else:
            yield from self._generate_with_context_stream(question, contexts, session_id)

    def _generate_with_context_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream answer with retrieval + conversation context."""
        if self.context_manager and session_id:
            built = self.context_manager.build_context(
                session_id=session_id,
                current_question=question,
                retrieval_contexts=contexts,
            )
            messages = built.to_llm_messages()
        else:
            messages = self._build_basic_messages(question, contexts)

        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("LLM streaming generation failed: %s", exc)
            raise LLMError(f"Streaming generation failed: {exc}") from exc

    def _generate_fallback_stream(
        self, question: str, session_id: Optional[str] = None
    ) -> Iterator[str]:
        """Stream answer using general knowledge when no contexts found."""
        logger.info("No contexts available — using general knowledge fallback (stream)")

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是一个智能助手。用户的问题在知识库中未找到相关信息，"
                    "请利用你的通用知识尝试回答，并友好地告知用户该信息可能不包含在上传的文档中。"
                ),
            },
        ]

        if self.context_manager and session_id:
            history = self.context_manager._load_history(session_id)
            messages.extend(history)

        messages.append({"role": "user", "content": question})

        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("Fallback streaming generation failed: %s", exc)
            raise LLMError(f"Fallback streaming generation failed: {exc}") from exc
