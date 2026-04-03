"""Embedding model management with DashScope primary and HuggingFace fallback."""

import os
import logging
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding model lifecycle.

    Tries DashScope first; falls back to a local HuggingFace model
    if the API key is missing or the remote call fails.
    """

    def __init__(
        self,
        embedding_model: str = None,
        dashscope_api_key: str = None,
    ):
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-v3"
        )
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.embeddings: Embeddings = self._init_embeddings()

    def _init_embeddings(self) -> Embeddings:
        """Initialize embedding model with automatic fallback."""
        try:
            # Ensure we have an API key
            if not self.dashscope_api_key or self.dashscope_api_key.strip() == "":
                logger.warning("No DashScope API key found in environment")
                raise ValueError("DashScope API key is required")

            embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key.strip(),
            )
            # Smoke test
            embeddings.embed_query("test")
            logger.info("DashScope embedding model loaded: %s", self.embedding_model)
            return embeddings

        except Exception as exc:
            logger.error("DashScope embedding init failed: %s", exc)
            logger.warning("Falling back to HuggingFace sentence-transformers")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as exc:
            logger.error("Document embedding failed: %s", exc)
            return []

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        try:
            return self.embeddings.embed_query(query)
        except Exception as exc:
            logger.error("Query embedding failed: %s", exc)
            return []