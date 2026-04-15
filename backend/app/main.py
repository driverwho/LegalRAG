"""FastAPI application factory for the RAG Backend Service."""

import asyncio
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Any, Union

# Add the project root directory to sys.path to enable absolute imports
# This ensures that 'backend' module can be found when running from any directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── Patch Starlette's multipart form parser BEFORE any request is processed ──
# Starlette 0.36+ introduced a 1 MB per-part limit (max_part_size) to defend
# against multipart-bomb DoS attacks.  That default is too restrictive for
# legal document ingestion which routinely handles large PDF / MD files.
# We raise the limit to sys.maxsize (≈ 9 EB on 64-bit) which is effectively
# unlimited without disabling the security check entirely.
import starlette.requests as _starlette_requests

_orig_request_form = _starlette_requests.Request.form


async def _unlimited_form(
    self: Any,
    *,
    max_files: Union[int, float] = 1000,
    max_fields: Union[int, float] = 1000,
    max_part_size: int = sys.maxsize,
) -> Any:
    """Delegate to the original ``Request.form`` with an unlimited part size."""
    return await _orig_request_form(
        self,
        max_files=max_files,
        max_fields=max_fields,
        max_part_size=max_part_size,
    )


_starlette_requests.Request.form = _unlimited_form  # type: ignore[method-assign]
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config.settings import get_settings
from backend.app.api.router import api_router
from backend.app.exceptions.handlers import RAGException, rag_exception_handler
from backend.app.core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Enable DEBUG for retrieval stages to see per-result content previews
logging.getLogger("backend.app.core.retriever.stages").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — run async startup / shutdown tasks.

    BM25 index construction is CPU-heavy and synchronous; we offload it
    to a thread so it never blocks the event loop during startup.
    """
    # ── startup ───────────────────────────────────────────────────────
    logger.info("Running lifespan startup tasks")

    async def _warmup_bm25():
        """Build BM25 index in a background thread (non-blocking)."""
        try:
            from backend.app.api.async_deps import get_bm25_retriever
            retriever = await asyncio.to_thread(get_bm25_retriever)
            logger.info(
                "BM25 index ready (%d documents)",
                retriever.get_corpus_size(),
            )
        except Exception as exc:
            # Non-fatal — pipeline falls back to vector-only search
            logger.warning("BM25 warmup failed (vector-only fallback): %s", exc)

    asyncio.create_task(_warmup_bm25())

    yield

    # ── shutdown ──────────────────────────────────────────────────────
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Backend Service",
        description="Enterprise-grade RAG API with vector search and LLM generation",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS — allow frontend cross-origin access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Centralized exception handling
    app.add_exception_handler(RAGException, rag_exception_handler)

    # Mount API routes
    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {"message": "RAG Backend Service is Running!"}

    # Ensure required directories exist
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.CHROMADB_PERSIST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.SQLITE_DB_PATH), exist_ok=True)

    # Initialize database tables
    init_db()

    logger.info("RAG Backend Service initialized")
    logger.info(
        "API docs available at http://%s:%d/docs", settings.APP_HOST, settings.APP_PORT
    )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    logger.info(
        "Starting RAG Backend at http://%s:%d",
        "localhost" if settings.APP_HOST == "0.0.0.0" else settings.APP_HOST,
        settings.APP_PORT,
    )
    uvicorn.run(
        "backend.app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
