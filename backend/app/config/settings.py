"""Centralized configuration using Pydantic BaseSettings."""

import json
import os
from functools import lru_cache
from typing import List, Tuple
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    DASHSCOPE_API_KEY: str = Field(default="", description="DashScope API key")

    # Embedding
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-v3", description="DashScope embedding model"
    )

    # LLM
    LLM_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="LLM API base URL (OpenAI compatible)",
    )
    LLM_MODEL: str = Field(default="qwen-plus", description="LLM model name")
    LLM_MODEL_MAX: str = Field(
        default="qwen3-max", description="LLM model name (max tier)"
    )

    # Moonshot / Kimi
    MOONSHOT_API_KEY: str = Field(default="", description="Moonshot API key")
    MOONSHOT_BASE_URL: str = Field(
        default="https://api.moonshot.cn/v1",
        description="Moonshot API base URL (OpenAI compatible)",
    )
    MOONSHOT_MODEL: str = Field(default="kimi-k2.5", description="Moonshot model name")

    # LLM preprocessing fallback chain
    LLM_FALLBACK_MAX_RETRIES: int = Field(
        default=3,
        description="Max retries per LLM provider before degrading to the next",
    )
    LLM_FALLBACK_RETRY_DELAY: float = Field(
        default=2.0,
        description="Base delay in seconds between retries (exponential backoff)",
    )
    LLM_FALLBACK_CHAIN: str = Field(
        default="",
        description=(
            "JSON array defining the LLM fallback providers (after the primary DashScope). "
            'Each element: {"name": "...", "api_key_env": "...", "base_url": "...", "model": "..."}. '
            "api_key_env references another env var name that holds the actual key. "
            'Example: [{"name":"kimi","api_key_env":"MOONSHOT_API_KEY","base_url":"https://api.moonshot.cn/v1","model":"kimi-k2.5"}]'
        ),
    )

    def get_fallback_chain(self) -> list:
        """Parse LLM_FALLBACK_CHAIN into a list of provider dicts.

        Each dict contains: name, api_key, base_url, model.
        Falls back to MOONSHOT_* settings for backward compatibility when
        LLM_FALLBACK_CHAIN is empty but MOONSHOT_API_KEY is set.
        """
        if self.LLM_FALLBACK_CHAIN.strip():
            try:
                chain = json.loads(self.LLM_FALLBACK_CHAIN)
                resolved = []
                for entry in chain:
                    resolved.append(
                        {
                            "name": entry["name"],
                            "api_key": os.getenv(entry["api_key_env"], ""),
                            "base_url": entry["base_url"],
                            "model": entry["model"],
                        }
                    )
                return resolved
            except (json.JSONDecodeError, KeyError) as exc:
                import logging

                logging.getLogger(__name__).error(
                    "Failed to parse LLM_FALLBACK_CHAIN: %s — falling back to defaults",
                    exc,
                )

        # Backward compatibility: auto-build from legacy MOONSHOT_* fields
        if self.MOONSHOT_API_KEY:
            return [
                {
                    "name": "kimi",
                    "api_key": self.MOONSHOT_API_KEY,
                    "base_url": self.MOONSHOT_BASE_URL,
                    "model": self.MOONSHOT_MODEL,
                }
            ]

        return []

    # Vector Store
    COLLECTION_NAME: str = Field(
        default="agent_rag", description="Default collection name"
    )
    CHROMADB_PERSIST_DIR: str = Field(
        default="./chroma_db", description="ChromaDB persistence directory"
    )

    # SQLite Database
    SQLITE_DB_PATH: str = Field(
        default="./data/chat_history.db",
        description="SQLite database path for chat history",
    )

    # File Upload
    UPLOAD_FOLDER: str = Field(
        default="./temp/vector_uploads", description="Temporary upload directory"
    )

    # Server
    APP_HOST: str = Field(default="0.0.0.0", description="Application host")
    APP_PORT: int = Field(default=5000, description="Application port")
    DEBUG: bool = Field(default=True, description="Debug mode")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")

    # Document Processing
    CHUNK_SIZE: int = Field(default=500, description="Document chunk size")
    CHUNK_OVERLAP: int = Field(default=50, description="Document chunk overlap")

    # PDF OCR Settings - images smaller than these ratios will be skipped
    PDF_OCR_THRESHOLD: Tuple[float, float] = Field(
        default=(0.01, 0.01),
        description="Minimum image size ratio (width_ratio, height_ratio) to trigger OCR",
    )

    # Document Preprocessing
    ENABLE_PREPROCESSING: bool = Field(
        default=True, description="Enable document preprocessing pipeline"
    )
    ENABLE_LLM_PREPROCESSING: bool = Field(
        default=True, description="Enable Qwen LLM processing stage in preprocessing"
    )
    PREPROCESSING_LLM_CHUNK_SIZE: int = Field(
        default=6000,
        description="Max characters per chunk for LLM preprocessing",
    )

    # Document Quality Check
    ENABLE_QUALITY_CHECK: bool = Field(
        default=True, description="Enable document quality checking"
    )
    ENABLE_LLM_QUALITY_CHECK: bool = Field(
        default=True, description="Enable Qwen LLM analysis in quality checker"
    )
    QUALITY_CHECK_LLM_CHUNK_SIZE: int = Field(
        default=4000,
        description="Max characters to send to LLM for quality check",
    )

    # Celery / Redis
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis)",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL (Redis)",
    )

    # Context Window
    CONTEXT_WINDOW_SIZE: int = Field(
        default=30000,
        description="Total context window size of the target LLM (in tokens)",
    )
    CONTEXT_RESERVED_OUTPUT_TOKENS: int = Field(
        default=2000,
        description="Tokens reserved for the model's response generation",
    )
    CONTEXT_PROTECTED_ROUNDS: int = Field(
        default=2,
        description="Number of most-recent conversation rounds never compressed",
    )
    # Model used for context compression (defaults to primary LLM)
    COMPACT_LLM_MODEL: str = Field(
        default="",
        description="Model for context compression; defaults to LLM_MODEL if empty",
    )

    # Retrieval
    SIMILARITY_THRESHOLD: float = Field(
        default=0.5, description="Maximum ChromaDB distance to keep (lower = stricter)"
    )
    MAX_RESULTS: int = Field(default=10, description="Maximum search results")
    VECTOR_WEIGHT: float = Field(
        default=1.0,
        description="RRF weight for vector (semantic) retriever. "
                    "Increase to favour semantic similarity over keyword match.",
    )
    BM25_WEIGHT: float = Field(
        default=1.0,
        description="RRF weight for BM25 (keyword) retriever. "
                    "Increase to favour exact keyword matches over semantic similarity.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    def __init__(self, **kwargs):
        # Try multiple locations for .env file
        env_paths = [
            ".env",  # Current directory
            "../../.env",  # From backend/app/config/ to backend/
            "backend/.env",  # Project root
        ]
        for path in env_paths:
            if os.path.exists(path):
                self.model_config["env_file"] = path
                break
        super().__init__(**kwargs)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()
