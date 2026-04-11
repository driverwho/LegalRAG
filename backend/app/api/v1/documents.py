"""Document upload API endpoints."""

import os
import logging
import uuid

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from backend.app.config.settings import get_settings, Settings
from backend.app.models.requests import DocumentUploadRequest
from backend.app.models.responses import TaskSubmitResponse
from backend.app.core.tasks.document_tasks import submit_document_task

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload_document", response_model=TaskSubmitResponse)
async def upload_document(body: DocumentUploadRequest):
    """Upload a document by server-side file path. Returns task_id for tracking."""
    if not os.path.exists(body.file_path):
        raise HTTPException(
            status_code=400, detail=f"File does not exist: {body.file_path}"
        )

    task_id = submit_document_task(body.file_path, body.collection_name)
    return TaskSubmitResponse(
        success=True,
        message="Document processing task submitted",
        task_id=task_id,
    )


@router.post("/upload_file", response_model=TaskSubmitResponse)
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(default="agent_rag"),
    use_ocr: bool = Form(default=False),
    settings: Settings = Depends(get_settings),
):
    """Upload a file. Saves to temp directory and submits async processing task."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Preserve the original filename (may contain CJK characters)
    original_filename = file.filename

    # Extract extension for file type detection
    _, ext = os.path.splitext(file.filename)
    if not ext:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename: '{file.filename}' (missing extension)",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    # Save to upload directory with UUID name to avoid collision
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    temp_name = f"{uuid.uuid4().hex}{ext.lower()}"
    file_path = os.path.join(settings.UPLOAD_FOLDER, temp_name)

    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(
        "File saved: %s (%d bytes, original: %s)",
        file_path, len(content), original_filename,
    )

    # Submit async task with original filename for metadata
    task_id = submit_document_task(
        file_path,
        collection_name,
        use_ocr=use_ocr,
        original_filename=original_filename,
        cleanup_after=True,
    )
    return TaskSubmitResponse(
        success=True,
        message=f"File '{original_filename}' uploaded. Processing task submitted.",
        task_id=task_id,
    )
