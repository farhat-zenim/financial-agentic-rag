import warnings
from typing import Optional

warnings.filterwarnings("ignore")

from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.config import config

# Global Embedding Model
try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    Settings.embed_model = FastEmbedEmbedding(model_name=config.EMBEDDING_MODEL)
except Exception:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)

# Cross-Encoder Model Singleton
_RERANKER_INSTANCE: Optional[SentenceTransformerRerank] = None


def get_qdrant_client() -> QdrantClient:
    """Returns a Qdrant client explicitly bound to IPv4 localhost."""
    return QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        api_key=config.QDRANT_API_KEY,
        timeout=config.CLIENT_TIMEOUT,
        prefer_grpc=False
    )


def get_reranker(top_n: int = config.DEFAULT_TOP_N) -> SentenceTransformerRerank:
    """Singleton getter for the Cross-Encoder Reranker model."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        print("⚡ [System Warmup] Loading Cross-Encoder Reranker into RAM...")
        _RERANKER_INSTANCE = SentenceTransformerRerank(
            model=config.RERANKER_MODEL,
            top_n=top_n
        )
    return _RERANKER_INSTANCE


def preload_models():
    """Pre-warms local models into memory on system boot."""
    get_reranker()
    print("✅ [System Warmup] Models initialized and ready for queries.")


def get_hybrid_retriever(
    collection_name: str = config.QDRANT_COLLECTION,
    top_k: int = config.DEFAULT_TOP_K,
    filters: Optional[MetadataFilters] = None
):
    """Initializes Qdrant vector store and returns hybrid retriever instance."""
    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=True,
        batch_size=32
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_retriever(similarity_top_k=top_k, filters=filters)