"""Async search endpoints (v2) — agent-routed, true async/await.

v2 endpoints now delegate to ``LegalRouterAgent`` which:
  1. Classifies the query intent
  2. Selects and dispatches retrieval tools (law / case / both)
  3. Chooses a per-type system prompt
  4. Streams the LLM answer

The original ``AsyncRAGPipeline`` + ``AsyncContextualChatManager``
search-only endpoint remains unchanged (no agent needed for raw search).
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.app.models.requests import SessionQueryRequest, SearchRequest
from backend.app.models.responses import QueryResponse, SearchResponse, SourceItem
from backend.app.core.retriever.async_rag import AsyncRAGPipeline
from backend.app.core.database.session_service import SessionService
from backend.app.core.agent import LegalRouterAgent
from backend.app.api.async_deps import (
    get_async_rag_pipeline,
    get_session_service,
    get_legal_router_agent,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    """Encode a dict as a Server-Sent Event frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── Streaming endpoint (v2) ───────────────────────────────────────────────────

@router.post("/query/stream/v2")
async def query_stream_v2(
    body: SessionQueryRequest,
    agent: LegalRouterAgent = Depends(get_legal_router_agent),
    session_service: SessionService = Depends(get_session_service),
):
    """Fully async streaming RAG (v2) — agent-routed.

    The agent handles the full pipeline:
      preprocessing → tool dispatch → retrieval → generation

    SSE event types
    ---------------
    ``progress``              stage hint before data arrives
    ``preprocessing_result``  corrected query + query type
    ``tool_dispatch``         which retrieval tools were selected
    ``sources``               retrieved + fused + deduped documents + stats
    ``chunk``                 one LLM token chunk
    ``done``                  stream finished
    ``error``                 unrecoverable failure
    """

    async def event_generator():
        sources_data: list = []
        answer_parts: list = []

        try:
            async for event in agent.run_stream(
                query=body.question,
                session_id=body.session_id,
                k=body.k,
            ):
                if event["type"] == "sources":
                    sources_data = event["sources"]
                elif event["type"] == "chunk":
                    answer_parts.append(event["text"])
                yield _sse(event)

        except Exception as exc:
            logger.error("Streaming pipeline error: %s", exc, exc_info=True)
            yield _sse({"type": "error", "message": str(exc)})
            return

        # Persist to DB after stream completes (sync SQLAlchemy → thread pool)
        if body.session_id:
            full_answer = "".join(answer_parts)

            def _persist():
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

            try:
                await asyncio.to_thread(_persist)
            except ValueError as exc:
                logger.warning("Failed to persist streamed messages: %s", exc)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable Nginx buffering
        },
    )


# ── Non-streaming endpoint (v2) ───────────────────────────────────────────────

@router.post("/query/v2", response_model=QueryResponse)
async def query_v2(
    body: SessionQueryRequest,
    agent: LegalRouterAgent = Depends(get_legal_router_agent),
    session_service: SessionService = Depends(get_session_service),
):
    """Fully async non-streaming RAG (v2) — agent-routed."""
    result = await agent.run(
        query=body.question,
        session_id=body.session_id,
        k=body.k,
    )

    if body.session_id:
        def _persist():
            session_service.add_message(
                session_id=body.session_id, role="user", content=body.question
            )
            session_service.add_message(
                session_id=body.session_id,
                role="assistant",
                content=result["answer"],
                sources=result["sources"],
            )
        try:
            await asyncio.to_thread(_persist)
        except ValueError as exc:
            logger.warning("Failed to persist messages: %s", exc)

    sources_data = result["sources"]
    return QueryResponse(
        success=True,
        question=body.question,
        answer=result["answer"],
        confidence=sources_data[0]["score"] if sources_data else 0.0,
        question_type=result.get("preprocessing", {}).get("query_type", "general"),
        sources=[
            SourceItem(
                content=s["content"],
                metadata=s["metadata"],
                score=s["score"],
            )
            for s in sources_data
        ],
    )


# ── Search-only endpoint (v2) ─────────────────────────────────────────────────

@router.post("/search/v2", response_model=SearchResponse)
async def search_v2(
    body: SearchRequest,
    pipeline: AsyncRAGPipeline = Depends(get_async_rag_pipeline),
):
    """Async hybrid similarity search without LLM generation (v2)."""
    hybrid = await pipeline.search_hybrid_async(
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
            for r in hybrid.results
        ],
    )
