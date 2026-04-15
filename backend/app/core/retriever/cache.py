"""Bounded LRU + TTL cache for hybrid search results."""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional, Tuple

from backend.app.core.retriever.models import HybridSearchResult


class SearchCache:
    """Thread-unsafe LRU cache with per-entry TTL.

    Designed for single-process async use — no locking needed because the
    asyncio event loop is single-threaded.  Background tasks that call
    ``asyncio.to_thread`` do NOT touch the cache.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries before the oldest is evicted (LRU).
    ttl : float
        Time-to-live in seconds.  Stale entries are lazily evicted on read.
    """

    def __init__(self, maxsize: int = 512, ttl: float = 300.0) -> None:
        self._store: OrderedDict[str, Tuple[HybridSearchResult, float]] = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self, query: str, k: int, collection: Optional[str]
    ) -> Optional[HybridSearchResult]:
        """Return cached result or ``None`` if missing / expired."""
        key = self._make_key(query, k, collection)
        entry = self._store.get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts < self.ttl:
            self._store.move_to_end(key)           # refresh LRU position
            return result
        del self._store[key]                        # lazy expiry
        return None

    def put(
        self,
        query: str,
        k: int,
        collection: Optional[str],
        result: HybridSearchResult,
    ) -> None:
        """Insert or refresh an entry, evicting LRU overflow."""
        key = self._make_key(query, k, collection)
        self._store[key] = (result, time.time())
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(query: str, k: int, collection: Optional[str]) -> str:
        raw = json.dumps({"q": query, "k": k, "c": collection}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()
