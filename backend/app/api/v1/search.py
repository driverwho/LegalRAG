"""Search and query API endpoints."""

import logging

from fastapi import APIRouter, Depends

from backend.app.models.requests import SessionQueryRequest, SearchRequest
from backend.app.models.responses import QueryResponse, SearchResponse, SourceItem
from backend.app.core.retriever.rag import RAGPipeline
from backend.app.core.database.session_service import SessionService
from backend.app.core.llm.contextual_chat import ContextualChatManager
from backend.app.api.deps import get_rag_pipeline, get_session_service, get_contextual_chat_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: SessionQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    session_service: SessionService = Depends(get_session_service),
    chat_manager: ContextualChatManager = Depends(get_contextual_chat_manager),
):
    """RAG question answering — retrieve context and generate an answer.

    If session_id is provided, conversation history is loaded and saved.
    """
    # Retrieve relevant documents
    search_results = pipeline.search(
        query=body.question,
        k=body.k,
        collection_name=body.collection_name,
    )

    # Build retrieval contexts for prompt assembly
    retrieval_contexts = [
        {"text": r.content, "source": r.metadata.get("source", "未知")}
        for r in search_results
    ]

    # Generate answer with conversation history
    answer = chat_manager.generate_rag_response(
        question=body.question,
        contexts=retrieval_contexts,
        session_id=body.session_id,
    )

    # Determine confidence / question_type from pipeline metadata
    confidence = search_results[0].score if search_results else 0.0
    question_type = "rag" if retrieval_contexts else "general"

    # Save user and assistant messages AFTER context building to avoid
    # the current question appearing twice in conversation history.
    if body.session_id:
        try:
            session_service.add_message(
                session_id=body.session_id,
                role="user",
                content=body.question,
            )
            sources_data = [
                {"content": r.content, "metadata": r.metadata, "score": r.score}
                for r in search_results
            ]
            session_service.add_message(
                session_id=body.session_id,
                role="assistant",
                content=answer,
                sources=sources_data,
            )
        except ValueError as e:
            logger.warning("Failed to save messages: %s", e)

    return QueryResponse(
        success=True,
        question=body.question,
        answer=answer,
        confidence=confidence,
        question_type=question_type,
        sources=[
            SourceItem(
                content=r.content,
                metadata=r.metadata,
                score=r.score,
            )
            for r in search_results
        ],
    )


@router.post("/search", response_model=SearchResponse)
async def search_similar(
    body: SearchRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Similarity search without LLM generation."""
    results = pipeline.search(
        query=body.query,
        k=body.k,
        collection_name=body.collection_name,
    )

    return SearchResponse(
        success=True,
        query=body.query,
        results=[
            SourceItem(
                content=r.content,
                metadata=r.metadata,
                score=r.score,
            )
            for r in results
        ],
    )
