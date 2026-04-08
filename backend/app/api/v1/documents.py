"""Document upload API endpoints."""

import os
import logging

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from werkzeug.utils import secure_filename

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
    """Upload a file. Saves to disk and submits async processing task."""
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Save file to temp directory
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    safe_name = secure_filename(file.filename)
    if not safe_name or "." not in safe_name:
        raise HTTPException(
            status_code=400, detail=f"Invalid filename: '{file.filename}'"
        )

    file_path = os.path.join(settings.UPLOAD_FOLDER, safe_name)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(f"File saved: {file_path} ({len(content)} bytes)")

    # Submit async task
    task_id = submit_document_task(file_path, collection_name, use_ocr=use_ocr)
    return TaskSubmitResponse(
        success=True,
        message=f"File '{safe_name}' uploaded. Processing task submitted.",
        task_id=task_id,
    )
