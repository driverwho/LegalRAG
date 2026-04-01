"""Pydantic request models for API validation."""

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
