"""Centralized configuration using Pydantic BaseSettings."""

import os
from functools import lru_cache
from typing import List
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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()
