"""Celery application instance for async document processing."""

import logging
from celery import Celery
from backend.app.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery("rag_worker")

# Configure Celery from settings
celery_app.conf.broker_url = settings.CELERY_BROKER_URL
celery_app.conf.result_backend = settings.CELERY_RESULT_BACKEND
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.timezone = "Asia/Shanghai"
celery_app.conf.enable_utc = True
celery_app.conf.task_track_started = True
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.result_expires = 3600

# Windows compatibility: use solo pool to avoid billiard multiprocessing issues
import sys

if sys.platform == "win32":
    celery_app.conf.worker_pool = "solo"
    logger.info("Windows detected: using solo pool for compatibility")

# Autodiscover tasks from backend.app.core.tasks
celery_app.autodiscover_tasks(["backend.app.core.tasks"])

logger.debug("Celery app initialized with broker: %s", settings.CELERY_BROKER_URL)
