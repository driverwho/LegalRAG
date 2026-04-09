"""Collection management API endpoints."""

from fastapi import APIRouter, Depends, Query

from backend.app.models.requests import ClearCollectionRequest
from backend.app.models.responses import CollectionInfoResponse, UploadResponse
from backend.app.core.vector_store.chroma import ChromaVectorStore
from backend.app.api.deps import get_vector_store

router = APIRouter()


@router.get("/collection_info", response_model=CollectionInfoResponse)
async def get_collection_info(
    collection_name: str = Query(..., description="Collection name to inspect"),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Get metadata and statistics for a collection."""
    db_info = vector_store.get_collection_info(collection_name=collection_name)
    return CollectionInfoResponse(success=True, database_info=db_info)


@router.post("/clear_collection", response_model=UploadResponse)
async def clear_collection(
    body: ClearCollectionRequest,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
):
    """Delete all documents in a collection."""
    vector_store.clear_collection(collection_name=body.collection_name)
    return UploadResponse(
        success=True,
        message=f"Collection '{body.collection_name}' cleared",
    )
