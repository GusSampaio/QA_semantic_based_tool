"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    mongodb_uri: str | None = os.getenv("MONGODB_URI")
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    hf_token: str | None = os.getenv("HF_TOKEN")

    mongodb_database: str = os.getenv("MONGODB_DATABASE", "book_rag_db")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION", "chapters")
    mongodb_vector_index: str = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    chat_model: str = os.getenv("CHAT_MODEL", "mistral-small-latest")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "8"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))


def get_settings() -> AppSettings:
    """Return settings read from the current process environment."""
    return AppSettings()


def missing_required_env(settings: AppSettings) -> list[str]:
    """Required variables for MongoDB + embedding + LLM operations."""
    missing: list[str] = []
    if not settings.mongodb_uri:
        missing.append("MONGODB_URI")
    if not settings.mistral_api_key:
        missing.append("MISTRAL_API_KEY")
    if not settings.hf_token:
        missing.append("HF_TOKEN")
    return missing
