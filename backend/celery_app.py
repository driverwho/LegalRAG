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

# 关键配置：禁用任务持久化（开发环境推荐）
# task_acks_late = False 确保任务在执行前就被确认，避免重启时重新执行
celery_app.conf.task_acks_late = False
# worker_prefetch_multiplier = 1 确保worker一次只取一个任务
celery_app.conf.worker_prefetch_multiplier = 1

# 开发环境可设置结果不持久化
celery_app.conf.result_expires = 60  # 1分钟后过期

# Windows compatibility: use solo pool to avoid billiard multiprocessing issues
import sys

if sys.platform == "win32":
    celery_app.conf.worker_pool = "solo"
    logger.info("Windows detected: using solo pool for compatibility")

# Autodiscover tasks from backend.app.core.tasks
celery_app.autodiscover_tasks(["backend.app.core.tasks"])

logger.debug("Celery app initialized with broker: %s", settings.CELERY_BROKER_URL)