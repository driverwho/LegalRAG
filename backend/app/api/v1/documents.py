"""Document upload API endpoints."""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from werkzeug.utils import secure_filename

from backend.app.config.settings import get_settings, Settings
from backend.app.models.requests import DocumentUploadRequest
from backend.app.models.responses import UploadResponse
from backend.app.core.vector_store.chroma import ChromaVectorStore
from backend.app.core.document.loader import DocumentLoader
from backend.app.core.document.splitter import DocumentSplitter
from backend.app.api.deps import (
    get_vector_store,
    get_document_loader,
    get_document_splitter,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _process_and_store(
    file_path: str,
    collection_name: str,
    vector_store: ChromaVectorStore,
    loader: DocumentLoader,
    splitter: DocumentSplitter,
) -> dict:
    """Load → split → store a document. Returns collection info on success."""
    documents = loader.load_single_file(file_path)
    if not documents:
        raise HTTPException(
            status_code=400, detail=f"Failed to load document: {file_path}"
        )

    chunks = splitter.split(documents)
    vector_store.add_documents(chunks, collection_name=collection_name)

    return vector_store.get_collection_info(collection_name=collection_name)


@router.post("/upload_document", response_model=UploadResponse)
async def upload_document(
    body: DocumentUploadRequest,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    loader: DocumentLoader = Depends(get_document_loader),
    splitter: DocumentSplitter = Depends(get_document_splitter),
):
    """Upload a document by server-side file path."""
    if not os.path.exists(body.file_path):
        raise HTTPException(
            status_code=400, detail=f"File does not exist: {body.file_path}"
        )

    try:
        db_info = _process_and_store(
            body.file_path, body.collection_name, vector_store, loader, splitter
        )
        return UploadResponse(
            success=True,
            message=f"Document processed successfully: {body.file_path}",
            database_info=db_info,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document upload failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {exc}"
        )


@router.post("/upload_file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(default="agent_rag"),
    settings: Settings = Depends(get_settings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    loader: DocumentLoader = Depends(get_document_loader),
    splitter: DocumentSplitter = Depends(get_document_splitter),
):
    """Upload a file via multipart form data."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Save to temp directory
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    safe_name = secure_filename(file.filename)
    file_path = os.path.join(settings.UPLOAD_FOLDER, safe_name)

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        db_info = _process_and_store(
            file_path, collection_name, vector_store, loader, splitter
        )
        return UploadResponse(
            success=True,
            message=f"File uploaded and processed: {safe_name}",
            database_info=db_info,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
