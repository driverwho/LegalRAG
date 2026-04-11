"""FastAPI dependency injection — singleton services wired from Settings."""

from functools import lru_cache

from backend.app.config.settings import get_settings
from backend.app.core.vector_store.chroma import ChromaVectorStore
from backend.app.core.llm.embedding import EmbeddingManager
from backend.app.core.llm.chat import ChatManager
from backend.app.core.llm.contextual_chat import ContextualChatManager
from backend.app.core.retriever.rag import RAGPipeline
from backend.app.core.document.loader import DocumentLoader
from backend.app.core.document.splitter import DocumentSplitter
from backend.app.core.document.preprocessor import DocumentPreprocessor
from backend.app.core.quality.checker import DocumentChecker
from backend.app.core.database.session_service import SessionService
from backend.app.core.context import ContextManager, ContextConfig


@lru_cache()
def get_context_manager() -> ContextManager:
    """Create a ContextManager wired from Settings."""
    session_service = get_session_service()
    config = ContextConfig.from_settings()
    return ContextManager(session_service, config)


@lru_cache()
def get_contextual_chat_manager() -> ContextualChatManager:
    """Create a ContextualChatManager with context management support."""
    settings = get_settings()
    return ContextualChatManager(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        context_manager=get_context_manager(),
    )


@lru_cache()
def get_embedding_manager() -> EmbeddingManager:
    settings = get_settings()
    return EmbeddingManager(
        embedding_model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


@lru_cache()
def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    em = get_embedding_manager()
    return ChromaVectorStore(
        collection_name=settings.COLLECTION_NAME,
        embeddings=em.embeddings,
        persist_directory=settings.CHROMADB_PERSIST_DIR,
    )


@lru_cache()
def get_chat_manager() -> ChatManager:
    settings = get_settings()
    return ChatManager(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
    )


@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(
        vector_store=get_vector_store(),
        chat_manager=get_chat_manager(),
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        max_results=settings.MAX_RESULTS,
    )


def get_document_loader() -> DocumentLoader:
    return DocumentLoader(use_ocr=False)


def get_document_splitter() -> DocumentSplitter:
    settings = get_settings()
    return DocumentSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def get_document_preprocessor() -> DocumentPreprocessor:
    """Create a DocumentPreprocessor wired from Settings."""
    settings = get_settings()
    return DocumentPreprocessor(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        enable_llm_preprocessing=settings.ENABLE_LLM_PREPROCESSING,
    )


def get_document_checker() -> DocumentChecker:
    """Create a DocumentChecker wired from Settings."""
    settings = get_settings()
    return DocumentChecker(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        enable_llm_check=settings.ENABLE_LLM_QUALITY_CHECK,
    )


def get_session_service() -> SessionService:
    """Create a SessionService for chat history management."""
    return SessionService()
