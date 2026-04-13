"""Search and query API endpoints."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

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
    # Retrieve relevant documents — collection_name=None triggers all-collection search
    search_results = pipeline.search(
        query=body.question,
        k=body.k,
        collection_name=None,
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


@router.post("/query/stream")
async def query_documents_stream(
    body: SessionQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    session_service: SessionService = Depends(get_session_service),
    chat_manager: ContextualChatManager = Depends(get_contextual_chat_manager),
):
    """Streaming RAG question answering via Server-Sent Events (SSE).

    The response is a text/event-stream where each event is a JSON object:
      - {"type": "progress", "text": "..."}       — stage hint before data arrives
      - {"type": "sources",  "sources": [...]}    — retrieved reference docs
      - {"type": "chunk",    "text": "..."}        — one per LLM token chunk
      - {"type": "done"}                           — signals stream completion
      - {"type": "error",    "message": "..."}     — on failure
    """
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def event_generator():
        # ── Stage 1: vector retrieval ──────────────────────────────────
        yield _sse({"type": "progress", "text": "检索向量数据库中..."})

        try:
            search_results = await asyncio.to_thread(
                pipeline.search, body.question, body.k
                # collection_name defaults to None → searches all collections
            )
        except Exception as exc:
            logger.error("Vector search error: %s", exc)
            yield _sse({"type": "error", "message": f"检索失败: {exc}"})
            return

        retrieval_contexts = [
            {"text": r.content, "source": r.metadata.get("source", "未知")}
            for r in search_results
        ]
        sources_data = [
            {"content": r.content, "metadata": r.metadata, "score": r.score}
            for r in search_results
        ]

        # Send retrieved sources to the frontend
        yield _sse({"type": "sources", "sources": sources_data})

        # ── Stage 2: LLM generation ────────────────────────────────────
        yield _sse({"type": "progress", "text": "正在生成回答..."})

        full_answer_parts = []
        try:
            # Use asyncio.Queue to bridge sync iterator to async generator
            queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _stream_worker():
                """Worker thread that consumes sync iterator and feeds queue."""
                try:
                    for chunk_text in chat_manager.generate_rag_response_stream(
                        question=body.question,
                        contexts=retrieval_contexts,
                        session_id=body.session_id,
                    ):
                        # Put chunk into queue (thread-safe)
                        asyncio.run_coroutine_threadsafe(
                            queue.put(chunk_text), loop
                        )
                except Exception as exc:
                    # Signal error to main coroutine
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("__error__", exc)), loop
                    )
                finally:
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(
                        queue.put(None), loop
                    )

            # Start worker in thread pool
            asyncio.create_task(asyncio.to_thread(_stream_worker))

            # Consume from queue without blocking event loop
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, tuple) and chunk[0] == "__error__":
                    raise chunk[1]
                full_answer_parts.append(chunk)
                yield _sse({"type": "chunk", "text": chunk})

        except Exception as exc:
            logger.error("Streaming generation error: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})
            return

        # Signal completion
        yield _sse({"type": "done"})

        # Persist messages to DB after streaming completes
        if body.session_id:
            full_answer = "".join(full_answer_parts)
            try:
                session_service.add_message(
                    session_id=body.session_id,
                    role="user",
                    content=body.question,
                )
                session_service.add_message(
                    session_id=body.session_id,
                    role="assistant",
                    content=full_answer,
                    sources=sources_data,
                )
            except ValueError as e:
                logger.warning("Failed to save streamed messages: %s", e)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
        },
    )


@router.post("/search", response_model=SearchResponse)
async def search_similar(
    body: SearchRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Similarity search without LLM generation — searches all collections."""
    results = pipeline.search(
        query=body.query,
        k=body.k,
        collection_name=None,
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

