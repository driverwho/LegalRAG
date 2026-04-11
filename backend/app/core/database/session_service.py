"""CRUD service for chat sessions and messages."""

import json
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.core.database.engine import SessionLocal
from backend.app.core.database.models import ChatSession, ChatMessage


class SessionService:
    """Service for managing chat sessions and messages."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize with optional database session.

        Args:
            db: Optional SQLAlchemy session. If not provided, a new session
                will be created for each operation.
        """
        self._db = db

    def _get_db(self) -> Session:
        """Get database session."""
        if self._db is not None:
            return self._db
        return SessionLocal()

    def create_session(self, title: Optional[str] = None) -> dict:
        """Create a new chat session.

        Args:
            title: Optional session title. Defaults to "新对话".

        Returns:
            Dictionary containing session data.
        """
        db = self._get_db()
        try:
            session = ChatSession(
                id=str(uuid.uuid4()),
                title=title or "新对话",
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.to_dict()
        finally:
            if self._db is None:
                db.close()

    def list_sessions(self) -> list[dict]:
        """List all sessions sorted by updated_at DESC.

        Returns:
            List of session dictionaries.
        """
        db = self._get_db()
        try:
            sessions = (
                db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
            )
            return [s.to_dict() for s in sessions]
        finally:
            if self._db is None:
                db.close()

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID.

        Args:
            session_id: The session UUID.

        Returns:
            Session dictionary or None if not found.
        """
        db = self._get_db()
        try:
            session = db.query(ChatSession).filter_by(id=session_id).first()
            return session.to_dict() if session else None
        finally:
            if self._db is None:
                db.close()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session UUID.

        Returns:
            True if deleted, False if not found.
        """
        db = self._get_db()
        try:
            session = db.query(ChatSession).filter_by(id=session_id).first()
            if session is None:
                return False
            db.delete(session)
            db.commit()
            return True
        finally:
            if self._db is None:
                db.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
        summary: bool = False,
    ) -> dict:
        """Add a message to a session.

        Args:
            session_id: The session UUID.
            role: "user" or "assistant".
            content: Message content.
            sources: Optional list of RAG sources (stored as JSON).
            summary: Mark this message as a compressed summary.

        Returns:
            Dictionary containing message data.

        Raises:
            ValueError: If session not found.
        """
        db = self._get_db()
        try:
            session = db.query(ChatSession).filter_by(id=session_id).first()
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")

            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                sources=json.dumps(sources, ensure_ascii=False) if sources else None,
                summary=summary,
            )
            db.add(message)
            db.commit()
            db.refresh(message)

            # Update session's updated_at timestamp
            session.updated_at = message.created_at
            db.commit()

            return message.to_dict()
        finally:
            if self._db is None:
                db.close()

    def replace_messages_with_summary(
        self,
        session_id: str,
        message_ids: list[int],
        summary_content: str,
    ) -> dict:
        """Replace a set of messages with a single summary message.

        Deletes the specified messages and inserts a new assistant message
        tagged summary=True in their place.

        Args:
            session_id: The session UUID.
            message_ids: IDs of the messages to replace.
            summary_content: The compressed summary text.

        Returns:
            The newly created summary message dict.
        """
        db = self._get_db()
        try:
            # Delete the messages to be replaced
            db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id,
                ChatMessage.id.in_(message_ids),
            ).delete(synchronize_session=False)

            summary_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=f"[历史对话摘要]\n{summary_content}",
                summary=True,
            )
            db.add(summary_msg)
            db.commit()
            db.refresh(summary_msg)
            return summary_msg.to_dict()
        finally:
            if self._db is None:
                db.close()

    def get_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a session.

        Args:
            session_id: The session UUID.

        Returns:
            List of message dictionaries with sources deserialized.
        """
        db = self._get_db()
        try:
            messages = (
                db.query(ChatMessage)
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.created_at)
                .all()
            )
            result = []
            for m in messages:
                msg_dict = m.to_dict()
                if msg_dict.get("sources"):
                    try:
                        msg_dict["sources"] = json.loads(msg_dict["sources"])
                    except json.JSONDecodeError:
                        msg_dict["sources"] = None
                result.append(msg_dict)
            return result
        finally:
            if self._db is None:
                db.close()

    def update_session_title(self, session_id: str, title: str) -> Optional[dict]:
        """Update a session's title.

        Args:
            session_id: The session UUID.
            title: New title.

        Returns:
            Updated session dictionary or None if not found.
        """
        db = self._get_db()
        try:
            session = db.query(ChatSession).filter_by(id=session_id).first()
            if session is None:
                return None
            session.title = title
            db.commit()
            db.refresh(session)
            return session.to_dict()
        finally:
            if self._db is None:
                db.close()
