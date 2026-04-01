"""Pydantic response models for API serialization."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    """A single retrieval source."""

    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float


class QueryResponse(BaseModel):
    """Response for RAG question answering."""

    success: bool
    question: str
    answer: str
    confidence: float
    question_type: str
    sources: List[SourceItem]


class UploadResponse(BaseModel):
    """Response for document upload operations."""

    success: bool
    message: str
    database_info: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response for similarity search."""

    success: bool
    query: str
    results: List[SourceItem]


class CollectionInfoResponse(BaseModel):
    """Response for collection info queries."""

    success: bool
    database_info: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    message: str
