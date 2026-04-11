"""Extended RAG pipeline with conversation context support."""

import logging
from typing import List, Dict, Any, Optional

from backend.app.core.vector_store.base import BaseVectorStore
from backend.app.core.llm.contextual_chat import ContextualChatManager
from backend.app.core.context import ContextManager
from backend.app.core.retriever.rag import RAGPipeline, RetrievalResult, AnswerResult

logger = logging.getLogger(__name__)


class ContextualRAGPipeline(RAGPipeline):
    """RAG pipeline with conversation context support.

    Extends the base RAGPipeline to incorporate conversation history
    into the generation process for multi-turn question answering.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        chat_manager: ContextualChatManager,
        similarity_threshold: float = 0.5,
        max_results: int = 10,
    ):
        # Initialize base class attributes directly
        self.vector_store = vector_store
        self.chat_manager = chat_manager
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

    def answer(
        self,
        question: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AnswerResult:
        """Full RAG pipeline with optional conversation context.

        Args:
            question: User question
            k: Number of context documents to retrieve
            collection_name: Target collection
            session_id: Optional session ID for loading conversation history

        Returns:
            AnswerResult with answer text, confidence, and sources
        """
        # 1. Retrieve similar content
        sources = self.search(query=question, k=k, collection_name=collection_name)

        # 2. Build context dicts for the LLM
        contexts = [{"source": s.source, "text": s.content} for s in sources]

        # 3. Generate response with conversation context
        answer_text = self.chat_manager.generate_rag_response(
            question=question,
            contexts=contexts,
            session_id=session_id,
        )

        # 4. Calculate confidence
        scores = [s.score for s in sources]
        confidence = self._calculate_confidence(scores)

        return AnswerResult(
            answer=answer_text,
            confidence=confidence,
            question_type="general",
            sources=sources,
        )
