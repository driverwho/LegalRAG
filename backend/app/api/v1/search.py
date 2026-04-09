"""Search and query API endpoints."""

from fastapi import APIRouter, Depends

from backend.app.models.requests import QueryRequest, SearchRequest
from backend.app.models.responses import QueryResponse, SearchResponse, SourceItem
from backend.app.core.retriever.rag import RAGPipeline
from backend.app.api.deps import get_rag_pipeline

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """RAG question answering — retrieve context and generate an answer."""
    result = pipeline.answer(
        question=body.question,
        k=body.k,
        collection_name=body.collection_name,
    )

    return QueryResponse(
        success=True,
        question=body.question,
        answer=result.answer,
        confidence=result.confidence,
        question_type=result.question_type,
        sources=[
            SourceItem(
                content=src.content,
                metadata=src.metadata,
                score=src.score,
            )
            for src in result.sources
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
