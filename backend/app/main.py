"""FastAPI application factory for the RAG Backend Service."""

import os
import sys
import logging

# Add the project root directory to sys.path to enable absolute imports
# This ensures that 'backend' module can be found when running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config.settings import get_settings
from backend.app.api.router import api_router
from backend.app.exceptions.handlers import RAGException, rag_exception_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Backend Service",
        description="Enterprise-grade RAG API with vector search and LLM generation",
        version="2.0.0",
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

