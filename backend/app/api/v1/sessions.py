"""Session management API endpoints for conversation persistence."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.models.requests import CreateSessionRequest, UpdateSessionRequest
from backend.app.models.responses import (
    SessionResponse,
    SessionListResponse,
    SessionDetailResponse,
    DeleteResponse,
)
from backend.app.api.deps import get_session_service
from backend.app.core.database.session_service import SessionService

router = APIRouter()


@router.post(
    "/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(
    body: CreateSessionRequest,
    session_service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Create a new conversation session with an optional title."""
    session = session_service.create_session(title=body.title)
    return SessionResponse(success=True, session=session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    session_service: SessionService = Depends(get_session_service),
) -> SessionListResponse:
    """List all sessions sorted by updated_at in descending order."""
    sessions = session_service.list_sessions()
    return SessionListResponse(success=True, sessions=sessions)


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
) -> SessionDetailResponse:
    """Get session details including all messages."""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    messages = session_service.get_messages(session_id)
    return SessionDetailResponse(
        success=True,
        session=session,
        messages=messages,
    )


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
) -> DeleteResponse:
    """Delete a session and all its messages."""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    session_service.delete_session(session_id)
    return DeleteResponse(
        success=True,
        message=f"Session {session_id} deleted successfully",
    )


@router.put("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    body: UpdateSessionRequest,
    session_service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Update the title of a session."""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    updated_session = session_service.update_session_title(session_id, body.title)
    return SessionResponse(success=True, session=updated_session)
