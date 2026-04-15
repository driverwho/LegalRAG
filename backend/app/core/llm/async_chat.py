"""Async chat manager using AsyncOpenAI with DashScope backend."""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from openai import AsyncOpenAI

from backend.app.exceptions.handlers import LLMError
from backend.app.core.context.context_manager import ContextManager

logger = logging.getLogger(__name__)

_FALLBACK_SYSTEM_PROMPT = (
    "你是一个智能助手。用户的问题在知识库中未找到相关信息，"
    "请利用你的通用知识尝试回答，并友好地告知用户该信息可能不包含在上传的文档中。"
)


class AsyncContextualChatManager:
    """Fully async chat manager with native AsyncOpenAI streaming.

    Replaces the sync ContextualChatManager's thread-pool bridge pattern:

      Before (sync wrapped):
        def _stream_worker():            # runs in thread
            for chunk in sync_stream():
                queue.put(chunk)         # cross-thread push

      After (true async):
        async for chunk in stream:       # event loop, no threads
            yield chunk.choices[0].delta.content
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        context_manager: Optional[ContextManager] = None,
    ):
        # AsyncOpenAI: shares the same interface as OpenAI but all calls
        # return coroutines / async iterators — no thread needed.
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.context_manager = context_manager

    # ──────────────────────────────────────────────────────────────────
    # Non-streaming
    # ──────────────────────────────────────────────────────────────────

    async def generate_rag_response(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> str:
        """Async RAG answer generation."""
        if not contexts:
            return await self._generate_fallback(question, session_id)
        return await self._generate_with_context(question, contexts, session_id)

    async def _generate_with_context(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> str:
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
            completion = await self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            answer = completion.choices[0].message.content
            logger.info("Async RAG response generated successfully")
            return answer
        except Exception as exc:
            logger.error("Async LLM generation failed: %s", exc)
            raise LLMError(f"Generation failed: {exc}") from exc

    async def _generate_fallback(
        self, question: str, session_id: Optional[str] = None
    ) -> str:
        logger.info("No contexts — using general knowledge fallback")
        messages = self._build_fallback_messages(question, session_id)
        try:
            completion = await self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return completion.choices[0].message.content
        except Exception as exc:
            logger.error("Async fallback generation failed: %s", exc)
            raise LLMError(f"Fallback generation failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────
    # Streaming  ← 核心改进：原生 AsyncIterator，零线程
    # ──────────────────────────────────────────────────────────────────

    async def generate_rag_response_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Native async streaming — no Queue, no thread bridge."""
        if not contexts:
            async for chunk in self._generate_fallback_stream(question, session_id):
                yield chunk
        else:
            async for chunk in self._generate_with_context_stream(
                question, contexts, session_id
            ):
                yield chunk

    async def _generate_with_context_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
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
            stream = await self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("Async LLM streaming failed: %s", exc)
            raise LLMError(f"Streaming generation failed: {exc}") from exc

    async def _generate_fallback_stream(
        self, question: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        logger.info("No contexts — fallback stream")
        messages = self._build_fallback_messages(question, session_id)
        try:
            stream = await self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("Async fallback streaming failed: %s", exc)
            raise LLMError(f"Fallback streaming failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────
    # Low-level helpers (for callers that build their own messages list)
    # ──────────────────────────────────────────────────────────────────

    async def stream_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream LLM tokens from a pre-built messages list.

        Used by callers (e.g. LegalRouterAgent) that construct their own
        system prompt + history instead of relying on the default
        ``build_context`` path.  This keeps ``client`` private and
        provides a stable, testable API surface.
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error("stream_messages failed: %s", exc)
            raise LLMError(f"Streaming generation failed: {exc}") from exc

    # ──────────────────────────────────────────────────────────────────
    # Prompt builder (shared, identical to ContextualChatManager)
    # ──────────────────────────────────────────────────────────────────

    def _build_basic_messages(
        self, question: str, contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
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

    def _build_fallback_messages(
        self, question: str, session_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build the messages list for no-context (fallback) generation.

        Shared by ``_generate_fallback`` (non-streaming) and
        ``_generate_fallback_stream`` (streaming) so the prompt is defined
        in exactly one place.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
        ]
        if self.context_manager and session_id:
            history = self.context_manager._load_history(session_id)
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages
