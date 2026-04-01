"""Centralized exception handling for the RAG application."""

from fastapi import Request
from fastapi.responses import JSONResponse


class RAGException(Exception):
    """Base exception for RAG application."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DocumentProcessingError(RAGException):
    """Raised when document loading or processing fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class CollectionNotFoundError(RAGException):
    """Raised when a requested collection does not exist."""

    def __init__(self, collection_name: str):
        super().__init__(f"Collection '{collection_name}' not found", status_code=404)


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class LLMError(RAGException):
    """Raised when LLM inference fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=502)


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """Global exception handler for RAGException and subclasses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.message},
    )
