"""Centralized configuration using Pydantic BaseSettings."""

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

    # Vector Store
    COLLECTION_NAME: str = Field(
        default="agent_rag", description="Default collection name"
    )
    CHROMADB_PERSIST_DIR: str = Field(
        default="./chroma_db", description="ChromaDB persistence directory"
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

    # Retrieval
    SIMILARITY_THRESHOLD: float = Field(
        default=0.5, description="Minimum similarity score"
    )
    MAX_RESULTS: int = Field(default=10, description="Maximum search results")

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
