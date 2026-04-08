"""Task status query API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from backend.celery_app import celery_app
from backend.app.models.responses import TaskStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Query the status of a document processing task."""
    try:
        result = AsyncResult(task_id, app=celery_app)

        if result.state == "PENDING":
            return TaskStatusResponse(
                task_id=task_id,
                status="PENDING",
                progress=0,
                message="Task is queued",
            )
        elif result.state == "STARTED":
            return TaskStatusResponse(
                task_id=task_id,
                status="STARTED",
                progress=5,
                message="Task started",
            )
        elif result.state == "PROGRESS":
            # result.info contains: {"stage": str, "progress": int, "message": str, "details": dict}
            meta = result.info if result.info else {}
            return TaskStatusResponse(
                task_id=task_id,
                status="PROGRESS",
                stage=meta.get("stage"),
                progress=meta.get("progress"),
                message=meta.get("message"),
                details=meta.get("details"),
            )
        elif result.state == "SUCCESS":
            # result.result contains: {"stage": str, "message": str, "details": dict, "completed_at": str}
            task_result = result.result if result.result else {}
            return TaskStatusResponse(
                task_id=task_id,
                status="SUCCESS",
                progress=100,
                message=task_result.get("message", "Task completed successfully"),
                result=task_result,
            )
        elif result.state == "FAILURE":
            error_msg = str(result.info) if result.info else "Unknown error"
            return TaskStatusResponse(
                task_id=task_id,
                status="FAILED",
                message=error_msg,
            )
        else:
            # Handle any other states
            return TaskStatusResponse(
                task_id=task_id,
                status=result.state,
                message=f"Task is in {result.state} state",
            )
    except Exception as exc:
        logger.error(f"Error querying task status for {task_id}: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to query task status: {exc}"
        )
