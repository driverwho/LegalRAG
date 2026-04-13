"""RAG pipeline orchestrating retrieval and generation."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.app.core.vector_store.base import BaseVectorStore
from backend.app.core.llm.chat import ChatManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class AnswerResult:
    """Complete RAG answer including sources and confidence."""

    answer: str
    confidence: float
    question_type: str
    sources: List[RetrievalResult] = field(default_factory=list)


class RAGPipeline:
    """Orchestrates retrieval → generation for RAG question answering.

    Keeps retrieval (vector store) and generation (LLM) cleanly separated
    while providing a unified interface for the API layer.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        chat_manager: ChatManager,
        similarity_threshold: float = 0.5,
        max_results: int = 10,
    ):
        self.vector_store = vector_store
        self.chat_manager = chat_manager
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

    def search(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Search for similar content, filtering by similarity threshold.

        When collection_name is None, searches across ALL collections and
        returns the globally best results. Pass an explicit name to restrict
        the search to a single collection.
        """
        if collection_name is None:
            raw_results = self.vector_store.search_all_collections(query=query, k=k)
        else:
            raw_results = self.vector_store.search(
                query=query, k=k, collection_name=collection_name
            )

        results = []
        for doc, score in raw_results:
            if score >= self.similarity_threshold:
                results.append(
                    RetrievalResult(
                        content=doc.page_content,
                        score=score,
                        metadata=doc.metadata,
                        source=doc.metadata.get("source", ""),
                    )
                )

        logger.info(
            "Search for '%s' returned %d results (threshold=%.2f, collection=%s)",
            query[:50],
            len(results),
            self.similarity_threshold,
            collection_name or "ALL",
        )
        return results

    def answer(
        self,
        question: str,
        k: int = 5,
        collection_name: Optional[str] = None,
    ) -> AnswerResult:
        """Full RAG pipeline: retrieve → generate → return with metadata.

        Args:
            question: User question.
            k: Number of context documents to retrieve.
            collection_name: Target collection.

        Returns:
            AnswerResult with answer text, confidence, and sources.
        """
        # 1. Retrieve similar content
        sources = self.search(query=question, k=k, collection_name=collection_name)

        # 2. Build context dicts for the LLM
        contexts = [{"source": s.source, "text": s.content} for s in sources]

        # 3. Generate response (ChatManager handles empty-context fallback)
        answer_text = self.chat_manager.generate_rag_response(question, contexts)

        # 4. Calculate confidence
        scores = [s.score for s in sources]
        confidence = self._calculate_confidence(scores)

        return AnswerResult(
            answer=answer_text,
            confidence=confidence,
            question_type="general",
            sources=sources,
        )

    @staticmethod
    def _calculate_confidence(scores: List[float]) -> float:
        """Calculate answer confidence based on similarity scores.

        Formula (preserved from original codebase):
            (max_score * 0.6 + avg_score * 0.4) * min(count / 5.0, 1.0)
        """
        if not scores:
            return 0.0

        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        count_weight = min(len(scores) / 5.0, 1.0)
        confidence = (max_score * 0.6 + avg_score * 0.4) * count_weight

        return min(confidence, 1.0)
