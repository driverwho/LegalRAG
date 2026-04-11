"""Database module for SQLite-based chat history persistence."""

from backend.app.core.database.engine import init_db, SessionLocal
from backend.app.core.database.models import ChatSession, ChatMessage
from backend.app.core.database.session_service import SessionService

__all__ = [
    "init_db",
    "SessionLocal",
    "ChatSession",
    "ChatMessage",
    "SessionService",
]
