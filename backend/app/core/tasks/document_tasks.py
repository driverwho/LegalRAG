"""Celery tasks for async document processing pipeline."""

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from langchain_core.documents import Document

from backend.celery_app import celery_app
from backend.app.config.settings import get_settings
from backend.app.core.tasks.task_state import (
    TaskStage,
    update_task_progress,
    build_result,
)

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for deduplication.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest of the file's SHA256 hash.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_preprocess_services():
    """Get preprocessing-related services (loader, preprocessor, checker).

    Does NOT include embedding/vector store - those are loaded separately
    in chunk_and_store to avoid loading embeddings during preprocessing.

    Returns:
        Tuple of (loader, preprocessor, checker).
    """
    from backend.app.core.document.loader import DocumentLoader
    from backend.app.core.document.preprocessor import DocumentPreprocessor, LLMProvider
    from backend.app.core.quality.checker import DocumentChecker
    from openai import OpenAI as OpenAIClient

    settings = get_settings()

    loader = DocumentLoader(use_ocr=False)

    # Build LLM fallback chain from configuration
    fallback_providers = []
    for provider_cfg in settings.get_fallback_chain():
        if provider_cfg.get("api_key"):
            fallback_providers.append(
                LLMProvider(
                    name=provider_cfg["name"],
                    client=OpenAIClient(
                        api_key=provider_cfg["api_key"],
                        base_url=provider_cfg["base_url"],
                    ),
                    model=provider_cfg["model"],
                )
            )

    preprocessor = DocumentPreprocessor(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        enable_llm_preprocessing=settings.ENABLE_LLM_PREPROCESSING,
        fallback_providers=fallback_providers,
        max_retries=settings.LLM_FALLBACK_MAX_RETRIES,
        retry_delay=settings.LLM_FALLBACK_RETRY_DELAY,
    )

    checker = DocumentChecker(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        enable_llm_check=settings.ENABLE_LLM_QUALITY_CHECK,
    )

    return loader, preprocessor, checker


def _get_vector_services():
    """Get vector storage services (splitter, vector_store).

    Loads embedding model - call this ONLY when ready to store documents.

    Returns:
        Tuple of (splitter, vector_store).
    """
    from backend.app.core.document.splitter import DocumentSplitter
    from backend.app.core.vector_store.chroma import ChromaVectorStore
    from backend.app.core.llm.embedding import EmbeddingManager

    settings = get_settings()

    splitter = DocumentSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # Load embedding model ONLY when needed (after preprocessing)
    embedding_manager = EmbeddingManager(
        embedding_model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    vector_store = ChromaVectorStore(
        collection_name=settings.COLLECTION_NAME,
        embeddings=embedding_manager.embeddings,
        persist_directory=settings.CHROMADB_PERSIST_DIR,
    )

    return splitter, vector_store


T = TypeVar("T")


def _retry_step(
    func: Callable[[], T],
    retry_exceptions: Tuple[type, ...],
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> T:
    """Execute *func* with retries on specified exceptions.

    Uses exponential backoff: delay × 2^(attempt-1).

    Args:
        func: Zero-argument callable to execute.
        retry_exceptions: Exception types that trigger a retry.
        max_retries: Maximum number of retry attempts.
        retry_delay: Base delay in seconds between retries.

    Returns:
        The return value of *func* on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_exceptions as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "Step failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Step failed after %d attempts: %s", max_retries + 1, exc,
                )
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pipeline step functions
# ---------------------------------------------------------------------------
# Each step receives the Celery *task* instance so that progress updates
# are written to the SAME task ID that the frontend is polling.
# Documents are passed as Python objects (no serialization needed).
# ---------------------------------------------------------------------------


def _step_validate_and_extract(
    task: Any,
    file_path: str,
    use_ocr: bool,
    original_filename: Optional[str],
) -> Tuple[List[Document], str]:
    """Step 1: Validate file and extract document content.

    Returns:
        ``(documents, file_hash)`` tuple.
    """
    from backend.app.core.document.loader import DocumentLoader

    update_task_progress(task, TaskStage.VALIDATING, 10, "Validating file...")

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    # Validate file extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Compute file hash for deduplication
    file_hash = _compute_file_hash(file_path)

    update_task_progress(task, TaskStage.EXTRACTING, 30, "Extracting text...")

    # Load document
    loader = DocumentLoader(use_ocr=use_ocr)
    documents = loader.load_single_file(file_path)

    if not documents:
        raise ValueError(f"Failed to load document: {file_path}")

    update_task_progress(
        task, TaskStage.EXTRACTING, 50, f"Extracted {len(documents)} pages"
    )

    # Overwrite metadata with original filename when available
    # (temp files use UUID names, losing the original CJK filename)
    if original_filename:
        for doc in documents:
            doc.metadata["original_filename"] = original_filename
            doc.metadata["file_name"] = original_filename
            doc.metadata["source"] = original_filename

    return documents, file_hash


def _step_preprocess_and_check(
    task: Any,
    documents: List[Document],
) -> Tuple[List[Document], int]:
    """Step 2: Preprocess documents and run quality check.

    Returns:
        ``(processed_documents, pending_count)`` tuple.
    """
    # Only load preprocessing services (no embedding to save time/memory)
    _loader, preprocessor, _checker = _get_preprocess_services()

    update_task_progress(
        task, TaskStage.PREPROCESSING, 55, "Preprocessing documents..."
    )

    # Run preprocessing
    processed_documents = preprocessor.preprocess(documents)

    # Check for documents that fell through to "pending" state
    pending_count = sum(
        1 for d in processed_documents if d.metadata.get("pending_preprocessing")
    )
    if pending_count:
        update_task_progress(
            task,
            TaskStage.PREPROCESSING_DEGRADED,
            60,
            f"LLM fallback chain exhausted for {pending_count} document(s) — marked pending",
        )

    update_task_progress(
        task, TaskStage.QUALITY_CHECKING, 70, "Running quality check..."
    )

    # Quality check placeholder (currently disabled)
    # quality_report = checker.compare_before_after(documents, processed_documents)

    return processed_documents, pending_count


def _step_chunk_and_store(
    task: Any,
    documents: List[Document],
    collection_name: str,
) -> Dict[str, Any]:
    """Step 3: Classify, split and store documents in vector database.

    Returns:
        Dict with db_info, chunk_count, doc_type, classification_confidence.
    """
    from backend.app.core.document.classifier import DocumentClassifier
    from backend.app.core.document.legal_splitter import LegalParentChildSplitter

    # Load vector services (includes embedding model) — AFTER preprocessing
    splitter, vector_store = _get_vector_services()

    # ---- Classify document type ----
    update_task_progress(
        task, TaskStage.CHUNKING, 75, "Classifying document type..."
    )

    classifier = DocumentClassifier()
    classification = classifier.classify(documents)
    doc_type = classification.doc_type

    # Stamp doc_type on source documents so it propagates to chunks
    for doc in documents:
        doc.metadata["doc_type"] = doc_type

    # ---- Route to appropriate splitter ----
    if doc_type == "law":
        update_task_progress(
            task, TaskStage.CHUNKING, 78,
            "Splitting legal document (parent-child)...",
        )

        # Derive law_name from filename metadata (best-effort)
        law_name = None
        for doc in documents:
            fname = (
                doc.metadata.get("original_filename")
                or doc.metadata.get("file_name")
                or ""
            )
            if fname:
                law_name = os.path.splitext(fname)[0]
                break

        legal_splitter = LegalParentChildSplitter(law_name=law_name)
        parents, children = legal_splitter.split(documents)

        if children:
            # Parent-child split succeeded — store both parents and children.
            # Parents provide context retrieval; children are for vector search.
            chunks = parents + children
        else:
            # No legal structure found — fall back to generic splitter
            logger.warning(
                "Legal splitter produced no children — "
                "falling back to recursive splitter"
            )
            chunks = splitter.split(documents)
    else:
        update_task_progress(
            task, TaskStage.CHUNKING, 78, "Splitting case document..."
        )
        chunks = splitter.split(documents)

    # Stamp doc_type and processed_at on every chunk
    processed_at = datetime.now().isoformat()
    for chunk in chunks:
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["processed_at"] = processed_at

    update_task_progress(
        task, TaskStage.VECTORIZING, 85, f"Vectorizing {len(chunks)} chunks..."
    )

    # Add chunks to vector store
    vector_store.add_documents(chunks, collection_name=collection_name)

    update_task_progress(task, TaskStage.COMPLETED, 100, "Processing complete")

    # Get collection info
    db_info = vector_store.get_collection_info(collection_name=collection_name)

    return {
        "db_info": db_info,
        "chunk_count": len(chunks),
        "doc_type": doc_type,
        "classification_confidence": classification.confidence,
    }


# ---------------------------------------------------------------------------
# Unified Celery task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="tasks.process_document",
    max_retries=0,       # retries are handled per-step inside the task
)
def process_document(
    self,
    file_path: str,
    collection_name: str,
    use_ocr: bool = False,
    original_filename: Optional[str] = None,
    cleanup_after: bool = False,
) -> Dict[str, Any]:
    """Unified document processing pipeline.

    Runs **all** processing steps inside a single Celery task so that
    every ``update_task_progress`` call writes to the **same** task ID.
    The frontend can therefore observe real-time progress from 0 % → 100 %.

    Pipeline::

        validate → extract → preprocess → quality check
            → classify → chunk → vectorize → store

    Per-step retry logic:

    - Step 1 (validate / extract): retries on ``IOError``, ``OSError``
    - Step 2 (preprocess / check): retries on ``ConnectionError``
    - Step 3 (chunk / store):      retries on ``ConnectionError``

    Args:
        file_path: Path to the document file.
        collection_name: Target vector-store collection.
        use_ocr: Whether to use OCR for extraction.
        original_filename: Original filename (preserves CJK characters).
        cleanup_after: Whether to delete the temp file after processing.
    """
    try:
        # ---- Step 1: Validate & Extract (retry on I/O errors) ----
        documents, file_hash = _retry_step(
            lambda: _step_validate_and_extract(
                self, file_path, use_ocr, original_filename,
            ),
            retry_exceptions=(IOError, OSError),
            max_retries=2,
        )

        # ---- Step 2: Preprocess & Quality Check (retry on connection errors) ----
        processed_documents, pending_count = _retry_step(
            lambda: _step_preprocess_and_check(self, documents),
            retry_exceptions=(ConnectionError,),
            max_retries=2,
        )

        # ---- Step 3: Classify, Chunk & Store (retry on connection errors) ----
        result_details = _retry_step(
            lambda: _step_chunk_and_store(
                self, processed_documents, collection_name,
            ),
            retry_exceptions=(ConnectionError,),
            max_retries=3,
            retry_delay=2.0,
        )

        result_details["file_hash"] = file_hash

        return build_result(
            TaskStage.COMPLETED,
            "Document processed successfully",
            result_details,
        )

    except Exception as exc:
        logger.error("Document processing pipeline failed: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 0, str(exc))
        raise

    finally:
        # Clean up temp file if requested (uploaded files, not server-side paths)
        if cleanup_after and file_path:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info("Cleaned up temp file: %s", file_path)
            except OSError as e:
                logger.warning(
                    "Failed to clean up temp file %s: %s", file_path, e,
                )


def submit_document_task(
    file_path: str,
    collection_name: str,
    use_ocr: bool = False,
    original_filename: Optional[str] = None,
    cleanup_after: bool = False,
) -> str:
    """Submit a document processing task.

    Unlike the previous ``chain()``-based approach, this now submits a
    **single** Celery task.  The returned task ID is the same ID used for
    all progress updates, so the frontend can poll it and observe the
    full 0 %–100 % lifecycle.

    Args:
        file_path: Path to the document file.
        collection_name: Target collection name.
        use_ocr: Whether to use OCR for extraction.
        original_filename: Original filename before sanitization.
        cleanup_after: Whether to delete the file after processing.

    Returns:
        The task ID (usable for ``GET /tasks/{task_id}``).
    """
    result = process_document.apply_async(
        args=[file_path, collection_name],
        kwargs={
            "use_ocr": use_ocr,
            "original_filename": original_filename,
            "cleanup_after": cleanup_after,
        },
    )
    logger.info("Submitted document processing task: %s", result.id)
    return result.id
