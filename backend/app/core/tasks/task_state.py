"""Task state management for document processing pipeline."""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStage(str, Enum):
    """Stages of the document processing pipeline."""

    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    EXTRACTING = "EXTRACTING"
    PREPROCESSING = "PREPROCESSING"
    PREPROCESSING_DEGRADED = "PREPROCESSING_DEGRADED"
    QUALITY_CHECKING = "QUALITY_CHECKING"
    CHUNKING = "CHUNKING"
    VECTORIZING = "VECTORIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def update_task_progress(
    task: Any,
    stage: TaskStage,
    progress: int,
    message: str = "",
    details: dict | None = None,
) -> None:
    """Update the task state with progress information.

    Args:
        task: The Celery task instance.
        stage: Current processing stage.
        progress: Progress percentage (0-100).
        message: Human-readable status message.
        details: Optional additional details dict.
    """
    meta = {
        "stage": stage.value,
        "progress": progress,
        "message": message,
        "details": details or {},
    }
    task.update_state(state="PROGRESS", meta=meta)
    logger.debug(
        "Task %s progress: stage=%s, progress=%d%%, message=%s",
        task.request.id,
        stage.value,
        progress,
        message,
    )


def build_result(
    stage: TaskStage,
    message: str,
    details: dict | None = None,
) -> dict:
    """Build the final result dictionary for task completion.

    Args:
        stage: Final stage reached.
        message: Human-readable result message.
        details: Optional additional details dict.

    Returns:
        Dict containing stage, message, details, and completion timestamp.
    """
    return {
        "stage": stage.value,
        "message": message,
        "details": details or {},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
