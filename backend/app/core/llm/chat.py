"""LLM chat completion for RAG response generation."""

import logging
from typing import List, Dict, Any

from openai import OpenAI

from backend.app.exceptions.handlers import LLMError

logger = logging.getLogger(__name__)


class ChatManager:
    """Manages LLM chat completions for RAG responses.

    Uses the OpenAI-compatible API (DashScope, etc.).
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_rag_response(
        self, question: str, contexts: List[Dict[str, Any]]
    ) -> str:
        """Generate an answer based on retrieved contexts.

        If contexts are empty, falls back to general knowledge with a disclaimer.
        """
        if not contexts:
            return self._generate_fallback(question)
        return self._generate_with_context(question, contexts)

    def _generate_with_context(
        self, question: str, contexts: List[Dict[str, Any]]
    ) -> str:
        """Generate answer grounded in provided contexts."""
        context_parts = []
        for idx, ctx in enumerate(contexts, 1):
            source = ctx.get("source", "未知")
            text = ctx.get("text", "")
            context_parts.append(f"[{idx}] 来源: {source}\n内容: {text}")

        combined = "\n\n---\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个中文 RAG 助手。\n"
                    "原则：\n"
                    "- 仅依据提供的上下文作答，不得编造；\n"
                    "- 当上下文不足或无关时，明确说明无法回答并指出缺失信息类型；\n"
                    "- 语言简洁准确，尽量提炼关键结论与要点；检索到的上下文和用户问的问题没关系的时候，忽略上下文！\n"
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

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            answer = completion.choices[0].message.content
            logger.info("RAG response generated successfully")
            return answer
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise LLMError(f"Generation failed: {exc}") from exc

    def _generate_fallback(self, question: str) -> str:
        """Generate answer using general knowledge when no contexts found."""
        logger.info("No contexts available — using general knowledge fallback")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个智能助手。用户的问题在知识库中未找到相关信息，"
                    "请利用你的通用知识尝试回答，并友好地告知用户该信息可能不包含在上传的文档中。"
                ),
            },
            {"role": "user", "content": question},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return completion.choices[0].message.content
        except Exception as exc:
            logger.error("Fallback generation failed: %s", exc)
            raise LLMError(f"Fallback generation failed: {exc}") from exc
