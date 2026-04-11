"""Pydantic request models for API validation."""

from typing import Optional
from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    """Request body for document upload by file path."""

    file_path: str
    collection_name: str


class QueryRequest(BaseModel):
    """Request body for RAG question answering."""

    question: str
    collection_name: str
    k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")


class SearchRequest(BaseModel):
    """Request body for similarity search."""

    query: str
    collection_name: str
    k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class ClearCollectionRequest(BaseModel):
    """Request body for clearing a collection."""

    collection_name: str


class CreateSessionRequest(BaseModel):
    """Request body for creating a new session."""

    title: Optional[str] = Field(default=None, description="Session title")


class UpdateSessionRequest(BaseModel):
    """Request body for updating a session."""

    title: str = Field(..., description="New session title")


class SessionQueryRequest(BaseModel):
    """Request body for RAG query within a session."""

    question: str
    collection_name: str
    session_id: Optional[str] = Field(
        default=None, description="Session ID to save conversation to"
    )
    k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
