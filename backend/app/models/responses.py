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


class TaskSubmitResponse(BaseModel):
    """Response when a document processing task is submitted."""

    success: bool
    message: str
    task_id: str


class TaskStatusResponse(BaseModel):
    """Response for task status query."""

    task_id: str
    status: str
    stage: Optional[str] = None
    progress: Optional[int] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response for session creation or update."""

    success: bool
    session: Dict[str, Any]


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    success: bool
    sessions: List[Dict[str, Any]]


class SessionDetailResponse(BaseModel):
    """Response for session detail with messages."""

    success: bool
    session: Dict[str, Any]
    messages: List[Dict[str, Any]]


class DeleteResponse(BaseModel):
    """Response for delete operations."""

    success: bool
    message: str


class CollectionItem(BaseModel):
    """A single collection summary."""

    name: str
    document_count: int


class CollectionListResponse(BaseModel):
    """Response for listing all collections."""

    success: bool
    collections: List[CollectionItem]


class DocumentItem(BaseModel):
    """A single document in a list."""

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """Paginated response for document listing."""

    success: bool
    documents: List[DocumentItem]
    total: int
    offset: int
    limit: int


class DocumentDetailResponse(BaseModel):
    """Response for a single document detail."""

    success: bool
    document: DocumentItem
