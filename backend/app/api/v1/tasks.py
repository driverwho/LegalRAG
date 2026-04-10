"""Task status query and control API endpoints."""

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


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running or pending document processing task.

    This will revoke the task and terminate it if it's currently running.
    Note: Tasks that have already completed cannot be cancelled.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)

        # Check if task exists
        if not result.id:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        # Check current state
        current_state = result.state
        if current_state == "SUCCESS":
            return {
                "success": False,
                "message": "Task already completed, cannot cancel",
                "task_id": task_id,
                "status": current_state,
            }

        if current_state == "FAILURE":
            return {
                "success": False,
                "message": "Task already failed, cannot cancel",
                "task_id": task_id,
                "status": current_state,
            }

        if current_state == "REVOKED":
            return {
                "success": False,
                "message": "Task already cancelled",
                "task_id": task_id,
                "status": current_state,
            }

        # Revoke the task
        # terminate=True: send SIGTERM to worker process if task is running
        # wait=True: wait for the task to be revoked
        result.revoke(terminate=True, wait=False)

        logger.info(f"Task {task_id} cancelled (state was: {current_state})")

        return {
            "success": True,
            "message": f"Task cancelled successfully (was {current_state})",
            "task_id": task_id,
            "previous_status": current_state,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error cancelling task {task_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {exc}")
