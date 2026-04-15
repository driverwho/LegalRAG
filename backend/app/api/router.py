"""API router aggregating all v1 endpoints under /api/vector."""

from fastapi import APIRouter

from backend.app.api.v1 import documents, search, collections, tasks, sessions, knowledge
from backend.app.api.v1 import async_search

api_router = APIRouter(prefix="/api/vector")

# ── v1 endpoints (sync / thread-bridge) ──────────────────────────────────────
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(collections.router, tags=["collections"])
api_router.include_router(tasks.router, tags=["tasks"])
api_router.include_router(sessions.router, tags=["sessions"])
api_router.include_router(knowledge.router, tags=["knowledge"])

# ── v2 endpoints (true async — parallel hybrid retrieval) ────────────────────
# Exposes:
#   POST /api/vector/query/v2          non-streaming
#   POST /api/vector/query/stream/v2   streaming SSE
#   POST /api/vector/search/v2         retrieval only
api_router.include_router(async_search.router, tags=["search-v2"])
