"""API router aggregating all v1 endpoints under /api/vector."""

from fastapi import APIRouter

from backend.app.api.v1 import documents, search, collections, tasks, sessions, knowledge

api_router = APIRouter(prefix="/api/vector")
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(collections.router, tags=["collections"])
api_router.include_router(tasks.router, tags=["tasks"])
api_router.include_router(sessions.router, tags=["sessions"])
api_router.include_router(knowledge.router, tags=["knowledge"])
