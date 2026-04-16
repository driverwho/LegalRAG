"""Dependency injection for async components — wired from real Settings."""

from functools import lru_cache

from openai import AsyncOpenAI

from backend.app.config.settings import get_settings
from backend.app.core.llm.embedding import EmbeddingManager
from backend.app.core.vector_store.chroma import ChromaVectorStore
from backend.app.core.retriever.async_rag import AsyncRAGPipeline
from backend.app.core.retriever.bm25 import HybridBM25Retriever
from backend.app.core.llm.async_chat import AsyncContextualChatManager
from backend.app.core.context import ContextManager, ContextConfig
from backend.app.core.database.session_service import SessionService
from backend.app.core.preprocessor.query_preprocessor import QueryPreprocessor
from backend.app.core.agent import LegalRouterAgent
from backend.app.core.agent.react_agent import LegalReActAgent
from backend.app.core.agent.tools.law_search import LawSearchTool
from backend.app.core.agent.tools.case_search import CaseSearchTool


# ── shared singletons (reuse existing ones where possible) ────────────────────

@lru_cache()
def get_session_service() -> SessionService:
    """Stateless DB wrapper — cached to keep a single instance
    consistent with the one captured by ContextManager."""
    return SessionService()


@lru_cache()
def get_async_embedding_manager() -> EmbeddingManager:
    """EmbeddingManager singleton — identical to sync version."""
    settings = get_settings()
    return EmbeddingManager(
        embedding_model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


@lru_cache()
def get_async_vector_store() -> ChromaVectorStore:
    """ChromaVectorStore singleton — same store as sync pipeline."""
    settings = get_settings()
    em = get_async_embedding_manager()
    return ChromaVectorStore(
        collection_name=settings.COLLECTION_NAME,
        embeddings=em.embeddings,
        persist_directory=settings.CHROMADB_PERSIST_DIR,
    )


@lru_cache()
def get_bm25_retriever() -> HybridBM25Retriever:
    """BM25 retriever that mirrors the vector store corpus.

    Built once at startup (index construction is CPU-bound).
    Returns None-equivalent stub if corpus is empty — the pipeline
    gracefully falls back to vector-only search in that case.
    """
    vector_store = get_async_vector_store()
    return HybridBM25Retriever(vector_store=vector_store)


@lru_cache()
def get_async_context_manager() -> ContextManager:
    """ContextManager singleton — shared with sync pipeline."""
    config = ContextConfig.from_settings()
    return ContextManager(get_session_service(), config)


@lru_cache()
def get_query_preprocessor() -> QueryPreprocessor:
    """QueryPreprocessor singleton.

    Uses a dedicated AsyncOpenAI client for spell-correction so it
    doesn't share connection state with the chat manager.
    """
    settings = get_settings()
    llm_client = AsyncOpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
    return QueryPreprocessor(llm_client=llm_client, model=settings.LLM_MODEL)


@lru_cache()
def get_async_rag_pipeline() -> AsyncRAGPipeline:
    """AsyncRAGPipeline singleton.

    Components wired:
      - vector_store  : ChromaVectorStore (same DB as sync pipeline)
      - bm25          : HybridBM25Retriever (corpus synced from Chroma)
      - preprocessor  : QueryPreprocessor (classification + correction + metadata)
      - reranker      : None for now (wire LLMReranker when ready)
    """
    settings = get_settings()
    return AsyncRAGPipeline(
        vector_store=get_async_vector_store(),
        bm25_retriever=get_bm25_retriever(),
        reranker=None,
        preprocessor=get_query_preprocessor(),
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        max_results=settings.MAX_RESULTS,
        vector_weight=settings.VECTOR_WEIGHT,
        bm25_weight=settings.BM25_WEIGHT,
    )


@lru_cache()
def get_async_chat_manager() -> AsyncContextualChatManager:
    """AsyncContextualChatManager singleton.

    Uses AsyncOpenAI (DashScope-compatible) — no thread pool needed.
    """
    settings = get_settings()
    return AsyncContextualChatManager(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        context_manager=get_async_context_manager(),
    )


@lru_cache()
def get_legal_router_agent() -> LegalRouterAgent:
    """LegalRouterAgent singleton (v2 — static routing).

    Wires together:
      - preprocessor  : query classification + correction + metadata
      - chat_manager  : async LLM generation (streaming)
      - context_manager: multi-turn history + token budgeting
      - tools         : law_search + case_search (backed by the shared pipeline)
    """
    pipeline = get_async_rag_pipeline()
    return LegalRouterAgent(
        preprocessor=get_query_preprocessor(),
        chat_manager=get_async_chat_manager(),
        context_manager=get_async_context_manager(),
        tools={
            "law_search": LawSearchTool(pipeline),
            "case_search": CaseSearchTool(pipeline),
        },
    )


@lru_cache()
def get_legal_react_agent() -> LegalReActAgent:
    """LegalReActAgent singleton (v3 — LangGraph ReAct loop).

    Wires together:
      - preprocessor    : query classification + correction
      - context_manager : multi-turn history + token budgeting
      - tools           : law_search + case_search (same AgentTool instances)
      - LLM             : DashScope Qwen via ChatOpenAI (with tool_calls)

    Uses the same underlying ``AsyncRAGPipeline`` as the v2 router agent,
    so both can coexist and share caches / vector store state.
    """
    settings = get_settings()
    pipeline = get_async_rag_pipeline()

    return LegalReActAgent(
        preprocessor=get_query_preprocessor(),
        context_manager=get_async_context_manager(),
        tools={
            "law_search": LawSearchTool(pipeline),
            "case_search": CaseSearchTool(pipeline),
        },
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        max_iterations=5,
    )
