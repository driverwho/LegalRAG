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
from backend.app.core.document.preprocessor import DocumentPreprocessor
from backend.app.core.quality.checker import DocumentChecker
from backend.app.api.deps import (
    get_vector_store,
    get_document_loader,
    get_document_splitter,
    get_document_preprocessor,
    get_document_checker,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _process_and_store(
    file_path: str,
    collection_name: str,
    vector_store: ChromaVectorStore,
    loader: DocumentLoader,
    splitter: DocumentSplitter,
    preprocessor: DocumentPreprocessor,
    checker: DocumentChecker,
) -> dict:
    """Load → preprocess (with quality check) → split → store a document.

    Returns collection info and quality report on success.
    """
    logger.info(f"Processing document: {file_path}")

    file_size = os.path.getsize(file_path)
    file_ext = os.path.splitext(file_path)[1]
    logger.info(f"File info: size={file_size} bytes, extension='{file_ext}'")

    documents = loader.load_single_file(file_path)

    # DEBUG: 将加载后的文档内容保存到txt文件
    debug_dir = os.path.join(os.path.dirname(file_path), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(
        debug_dir, Path(file_path).stem + "_loaded.txt"
    )
    with open(debug_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            f.write(f"=== 第 {i+1} 段 ===\n")
            f.write(doc.page_content)
            f.write("\n\n")
    logger.info(f"[DEBUG] 文档加载内容已保存至: {debug_path}")

    if not documents:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load document: {file_path}. File type may not be supported or file may be corrupted.",
        )

    logger.info(f"Document loaded successfully: {len(documents)} pages/sections")

    # Preprocess documents (NLTK + Qwen)
    logger.info("Running document preprocessing pipeline...")
    processed_documents = preprocessor.preprocess(documents)
    logger.info(f"Preprocessing complete: {len(processed_documents)} documents")

    # Quality check: compare before vs after preprocessing
    logger.info("Running quality comparison (before vs after preprocessing)...")
    quality_report = checker.compare_before_after(documents, processed_documents)
    before_errors = quality_report["before"]["total_errors"]
    after_errors = quality_report["after"]["total_errors"]
    errors_reduced = quality_report["improvement"]["errors_reduced"]
    reduction_rate = quality_report["improvement"]["reduction_rate"]
    logger.info(
        f"Quality comparison: {before_errors} errors (before) → {after_errors} errors (after), "
        f"reduced {errors_reduced} errors ({reduction_rate}% improvement)"
    )

    chunks = splitter.split(processed_documents)
    logger.info(f"Document split into {len(chunks)} chunks")
    vector_store.add_documents(chunks, collection_name=collection_name)

    db_info = vector_store.get_collection_info(collection_name=collection_name)
    db_info["quality_report"] = quality_report

    return db_info


@router.post("/upload_document", response_model=UploadResponse)
async def upload_document(
    body: DocumentUploadRequest,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    loader: DocumentLoader = Depends(get_document_loader),
    splitter: DocumentSplitter = Depends(get_document_splitter),
    preprocessor: DocumentPreprocessor = Depends(get_document_preprocessor),
    checker: DocumentChecker = Depends(get_document_checker),
):
    """Upload a document by server-side file path."""
    if not os.path.exists(body.file_path):
        raise HTTPException(
            status_code=400, detail=f"File does not exist: {body.file_path}"
        )

    try:
        db_info = _process_and_store(
            body.file_path,
            body.collection_name,
            vector_store,
            loader,
            splitter,
            preprocessor,
            checker,
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
    use_ocr: bool = Form(default=False),
    settings: Settings = Depends(get_settings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    loader: DocumentLoader = Depends(get_document_loader),
    splitter: DocumentSplitter = Depends(get_document_splitter),
    preprocessor: DocumentPreprocessor = Depends(get_document_preprocessor),
    checker: DocumentChecker = Depends(get_document_checker),
):
    """Upload a file via multipart form data."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    logger.info(
        f"Received file upload: filename='{file.filename}', content_type='{file.content_type}'"
    )

    # Save to temp directory
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    safe_name = secure_filename(file.filename)

    # Validate filename has proper extension
    if not safe_name or "." not in safe_name:
        logger.error(
            f"Invalid filename after sanitization: '{safe_name}' (original: '{file.filename}')"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename: '{file.filename}'. File must have a valid extension (e.g., .pdf, .docx)",
        )

    file_path = os.path.join(settings.UPLOAD_FOLDER, safe_name)
    logger.info(f"Saving file to: {file_path}")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved successfully: {file_path} ({len(content)} bytes)")

        # Use OCR-enabled loader if requested and file type supports it
        if use_ocr:
            from backend.app.core.document.loader import DocumentLoader as OCRLoader

            loader = OCRLoader(use_ocr=True)
            logger.info(f"Using OCR-enabled loader for file: {safe_name}")

        db_info = _process_and_store(
            file_path,
            collection_name,
            vector_store,
            loader,
            splitter,
            preprocessor,
            checker,
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
