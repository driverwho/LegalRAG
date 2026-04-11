"""Knowledge base management API endpoints."""

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
import chromadb

from backend.app.models.requests import DocumentDeleteRequest, DocumentUpdateRequest
from backend.app.models.responses import (
    CollectionListResponse,
    CollectionItem,
    DocumentListResponse,
    DocumentItem,
    DocumentDetailResponse,
    DeleteResponse,
)
from backend.app.core.vector_store.chroma import ChromaVectorStore
from backend.app.api.deps import get_chroma_client, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Helper functions using the lightweight chromadb client directly
# (no embedding model initialization required)
# ------------------------------------------------------------------


def _list_collections(client: chromadb.ClientAPI) -> List[Dict[str, Any]]:
    result = []
    for col in client.list_collections():
        try:
            count = client.get_collection(name=col.name).count()
        except Exception:
            count = 0
        result.append({"name": col.name, "document_count": count})
    return result


def _get_documents(
    client: chromadb.ClientAPI,
    collection_name: str,
    offset: int = 0,
    limit: int = 20,
    keyword: Optional[str] = None,
) -> Dict[str, Any]:
    empty = {"documents": [], "total": 0, "offset": offset, "limit": limit}

    names = [c.name for c in client.list_collections()]
    if collection_name not in names:
        return empty

    collection = client.get_collection(name=collection_name)
    total = collection.count()
    if total == 0:
        return empty

    if keyword:
        data = collection.get(
            where_document={"$contains": keyword},
            include=["documents", "metadatas"],
        )
        all_ids = data.get("ids", [])
        all_docs = data.get("documents", [])
        all_metas = data.get("metadatas", [])
        filtered_total = len(all_ids)

        page_ids = all_ids[offset : offset + limit]
        page_docs = all_docs[offset : offset + limit]
        page_metas = all_metas[offset : offset + limit]

        documents = []
        for i, doc_id in enumerate(page_ids):
            documents.append({
                "id": doc_id,
                "content": page_docs[i] if page_docs else "",
                "metadata": page_metas[i] if page_metas else {},
            })
        return {
            "documents": documents,
            "total": filtered_total,
            "offset": offset,
            "limit": limit,
        }

    data = collection.get(
        include=["documents", "metadatas"],
        offset=offset,
        limit=limit,
    )
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    documents = []
    for i, doc_id in enumerate(ids):
        documents.append({
            "id": doc_id,
            "content": docs[i] if docs else "",
            "metadata": metas[i] if metas else {},
        })

    return {
        "documents": documents,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def _get_document(
    client: chromadb.ClientAPI, collection_name: str, doc_id: str
) -> Optional[Dict[str, Any]]:
    names = [c.name for c in client.list_collections()]
    if collection_name not in names:
        return None

    collection = client.get_collection(name=collection_name)
    data = collection.get(ids=[doc_id], include=["documents", "metadatas"])

    if not data["ids"]:
        return None

    return {
        "id": data["ids"][0],
        "content": data["documents"][0] if data["documents"] else "",
        "metadata": data["metadatas"][0] if data["metadatas"] else {},
    }


def _delete_documents(
    client: chromadb.ClientAPI, collection_name: str, ids: List[str]
) -> int:
    names = [c.name for c in client.list_collections()]
    if collection_name not in names:
        return 0

    collection = client.get_collection(name=collection_name)
    existing = collection.get(ids=ids, include=[])
    found_ids = existing["ids"]
    if not found_ids:
        return 0

    collection.delete(ids=found_ids)
    logger.info("Deleted %d documents from '%s'", len(found_ids), collection_name)
    return len(found_ids)


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections(
    client: chromadb.ClientAPI = Depends(get_chroma_client),
):
    """List all collections with document counts."""
    try:
        raw = _list_collections(client)
        collections = [CollectionItem(**c) for c in raw]
        return CollectionListResponse(success=True, collections=collections)
    except Exception as exc:
        logger.error("Failed to list collections: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/knowledge/{collection_name}/documents",
    response_model=DocumentListResponse,
)
async def list_documents(
    collection_name: str,
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Page size"),
    keyword: Optional[str] = Query(default=None, description="Content keyword filter"),
    client: chromadb.ClientAPI = Depends(get_chroma_client),
):
    """Paginated document listing for a collection, with optional keyword filter."""
    try:
        result = _get_documents(client, collection_name, offset, limit, keyword)
        documents = [DocumentItem(**d) for d in result["documents"]]
        return DocumentListResponse(
            success=True,
            documents=documents,
            total=result["total"],
            offset=result["offset"],
            limit=result["limit"],
        )
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/knowledge/{collection_name}/documents/{doc_id}",
    response_model=DocumentDetailResponse,
)
async def get_document(
    collection_name: str,
    doc_id: str,
    client: chromadb.ClientAPI = Depends(get_chroma_client),
):
    """Get a single document with full content and metadata."""
    doc = _get_document(client, collection_name, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in collection '{collection_name}'",
        )
    return DocumentDetailResponse(success=True, document=DocumentItem(**doc))


@router.delete(
    "/knowledge/{collection_name}/documents",
    response_model=DeleteResponse,
)
async def delete_documents(
    collection_name: str,
    body: DocumentDeleteRequest,
    client: chromadb.ClientAPI = Depends(get_chroma_client),
):
    """Batch delete documents by IDs."""
    count = _delete_documents(client, collection_name, body.ids)
    return DeleteResponse(
        success=True,
        message=f"Deleted {count} document(s) from '{collection_name}'",
    )


@router.put(
    "/knowledge/{collection_name}/documents/{doc_id}",
    response_model=DocumentDetailResponse,
)
async def update_document(
    collection_name: str,
    doc_id: str,
    body: DocumentUpdateRequest,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    client: chromadb.ClientAPI = Depends(get_chroma_client),
):
    """Update a document's content and/or metadata.

    Uses the full vector store (with embeddings) only when content changes.
    """
    if body.content is None and body.metadata is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'content' or 'metadata' must be provided",
        )

    if body.content is not None:
        # Content changed — need to re-embed via the full vector store
        ok = vector_store.update_document(
            doc_id=doc_id,
            content=body.content,
            metadata=body.metadata,
            collection_name=collection_name,
        )
    else:
        # Metadata-only update — use lightweight client directly
        names = [c.name for c in client.list_collections()]
        if collection_name not in names:
            ok = False
        else:
            collection = client.get_collection(name=collection_name)
            existing = collection.get(ids=[doc_id], include=[])
            if not existing["ids"]:
                ok = False
            else:
                collection.update(ids=[doc_id], metadatas=[body.metadata])
                ok = True

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in collection '{collection_name}'",
        )

    updated = _get_document(client, collection_name, doc_id)
    return DocumentDetailResponse(success=True, document=DocumentItem(**updated))
