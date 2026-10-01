import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class AppConfig:
    """Production RAG configuration."""
    APP_NAME: str = "sec-10k-rag"

    # Qdrant Database Settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "sec_10k_filings")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None) or None

    # Model Specifications
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    # Local PDF HTTP Server Settings
    PDF_SERVER_URL: str = os.getenv("PDF_SERVER_URL", "http://localhost:8000")
    DATA_DIR: str = os.getenv("DATA_DIR", "data/raw_pdfs/pdfs")

    # Timeouts & Retrieval Tuning
    CLIENT_TIMEOUT: float = 60.0
    LLM_TIMEOUT: float = 360.0
    DEFAULT_TOP_K: int = 15
    DEFAULT_TOP_N: int = 4
    MAX_CHUNK_CHARS: int = 1400


config = AppConfig()