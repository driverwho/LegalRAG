"""Celery tasks for async document processing pipeline."""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from celery import chain
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


@celery_app.task(
    bind=True,
    name="tasks.validate_and_extract",
    max_retries=2,
    autoretry_for=(IOError, OSError),
    retry_backoff=True,
)
def validate_and_extract(
    self, file_path: str, collection_name: str, use_ocr: bool = False
) -> Dict[str, Any]:
    """Validate file and extract document content.

    Args:
        file_path: Path to the document file.
        collection_name: Target collection name.
        use_ocr: Whether to use OCR for extraction.

    Returns:
        Dict containing file_path, collection_name, file_hash, documents (serialized), and document_count.
    """
    try:
        from backend.app.core.document.loader import DocumentLoader

        update_task_progress(self, TaskStage.VALIDATING, 10, "Validating file...")

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Validate file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Compute file hash for deduplication
        file_hash = _compute_file_hash(file_path)

        update_task_progress(self, TaskStage.EXTRACTING, 30, "Extracting text...")

        # Load document
        loader = DocumentLoader(use_ocr=use_ocr)
        documents = loader.load_single_file(file_path)

        if not documents:
            raise ValueError(f"Failed to load document: {file_path}")

        update_task_progress(
            self, TaskStage.EXTRACTING, 50, f"Extracted {len(documents)} pages"
        )

        # Serialize documents for passing to next task
        serialized_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]

        return {
            "file_path": file_path,
            "collection_name": collection_name,
            "file_hash": file_hash,
            "documents": serialized_docs,
            "document_count": len(documents),
        }

    except (IOError, OSError) as exc:
        logger.error("File I/O error in validate_and_extract: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 10, str(exc))
        raise self.retry(exc=exc)
    except Exception as exc:
        logger.error("Validation/extraction failed: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 10, str(exc))
        raise


@celery_app.task(
    bind=True,
    name="tasks.preprocess_and_check",
    max_retries=2,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
)
def preprocess_and_check(self, extract_result: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess documents and run quality check.

    Args:
        extract_result: Result from validate_and_extract task.

    Returns:
        Dict containing all extract_result fields plus quality_report, with processed documents.
    """
    try:
        # Only load preprocessing services (no embedding to save time/memory)
        loader, preprocessor, checker = _get_preprocess_services()

        # Deserialize documents
        documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in extract_result["documents"]
        ]

        update_task_progress(
            self, TaskStage.PREPROCESSING, 55, "Preprocessing documents..."
        )

        # Run preprocessing
        processed_documents = preprocessor.preprocess(documents)

        # Check for documents that fell through to "pending" state
        pending_count = sum(
            1 for d in processed_documents if d.metadata.get("pending_preprocessing")
        )
        if pending_count:
            update_task_progress(
                self,
                TaskStage.PREPROCESSING_DEGRADED,
                60,
                f"LLM fallback chain exhausted for {pending_count} document(s) — marked pending",
            )

        update_task_progress(
            self, TaskStage.QUALITY_CHECKING, 70, "Running quality check..."
        )

        # Run quality check
        quality_report = checker.compare_before_after(documents, processed_documents)

        # Log quality report summary
        before_errors = quality_report["before"]["total_errors"]
        after_errors = quality_report["after"]["total_errors"]
        errors_reduced = quality_report["improvement"]["errors_reduced"]
        reduction_rate = quality_report["improvement"]["reduction_rate"]
        logger.info(
            "Quality comparison: %d errors (before) → %d errors (after), "
            "reduced %d errors (%.1f%% improvement)",
            before_errors,
            after_errors,
            errors_reduced,
            reduction_rate,
        )

        # Serialize processed documents
        serialized_processed_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in processed_documents
        ]

        # Return with processed documents (overwrites original documents)
        return {
            **extract_result,
            "documents": serialized_processed_docs,
            "quality_report": quality_report,
            "pending_preprocessing_count": pending_count,
        }

    except ConnectionError as exc:
        logger.error("Connection error in preprocess_and_check: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 55, str(exc))
        raise self.retry(exc=exc)
    except Exception as exc:
        logger.error("Preprocessing/quality check failed: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 55, str(exc))
        raise


@celery_app.task(
    bind=True,
    name="tasks.chunk_and_store",
    max_retries=3,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_backoff_max=60,
)
def chunk_and_store(self, preprocess_result: Dict[str, Any]) -> Dict[str, Any]:
    """Split documents into chunks and store in vector database.

    Args:
        preprocess_result: Result from preprocess_and_check task.

    Returns:
        Dict with final result including db_info, quality_report, file_hash, and chunk_count.
    """
    try:
        # Load vector services (includes embedding model) - AFTER preprocessing
        splitter, vector_store = _get_vector_services()

        # Deserialize processed documents
        documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in preprocess_result["documents"]
        ]

        update_task_progress(self, TaskStage.CHUNKING, 75, "Splitting documents...")

        # Split documents into chunks
        chunks = splitter.split(documents)

        update_task_progress(
            self, TaskStage.VECTORIZING, 85, f"Vectorizing {len(chunks)} chunks..."
        )

        # Add chunks to vector store
        vector_store.add_documents(
            chunks, collection_name=preprocess_result["collection_name"]
        )

        update_task_progress(self, TaskStage.COMPLETED, 100, "Processing complete")

        # Get collection info
        db_info = vector_store.get_collection_info(
            collection_name=preprocess_result["collection_name"]
        )

        return build_result(
            TaskStage.COMPLETED,
            "Document processed successfully",
            {
                "db_info": db_info,
                "quality_report": preprocess_result.get("quality_report"),
                "file_hash": preprocess_result.get("file_hash"),
                "chunk_count": len(chunks),
            },
        )

    except ConnectionError as exc:
        logger.error("Connection error in chunk_and_store: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 85, str(exc))
        raise self.retry(exc=exc)
    except Exception as exc:
        logger.error("Chunking/storage failed: %s", exc)
        update_task_progress(self, TaskStage.FAILED, 85, str(exc))
        raise


def submit_document_task(
    file_path: str, collection_name: str, use_ocr: bool = False
) -> str:
    """Submit a document processing task chain.

    Args:
        file_path: Path to the document file.
        collection_name: Target collection name.
        use_ocr: Whether to use OCR for extraction.

    Returns:
        The chain's task ID.
    """
    task_chain = chain(
        validate_and_extract.s(file_path, collection_name, use_ocr),
        preprocess_and_check.s(),
        chunk_and_store.s(),
    )
    result = task_chain.apply_async()
    logger.info("Submitted document task chain: %s", result.id)
    return result.id
