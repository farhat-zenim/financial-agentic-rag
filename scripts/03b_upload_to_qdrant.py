import pickle
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

def main():
    cache_path = Path("data/processed/embedded_nodes.pkl")
    collection_name = "sec_10k_filings"

    if not cache_path.exists():
        print(f"Error: {cache_path} not found. Run 03a_generate_embeddings.py first.")
        return

    print(f"Loading pre-computed nodes from {cache_path}...")
    with open(cache_path, "rb") as f:
        nodes = pickle.load(f)
    print(f"Loaded {len(nodes)} fully embedded nodes.")

    print("Connecting to Qdrant (preferring gRPC, 300s timeout)...")
    client = QdrantClient(
        host="localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=True,
        timeout=300.0
    )

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            # REQUIRED FIX: Explicitly allocate the sparse layer during manual creation
            sparse_vectors_config={
                "text-sparse-new": models.SparseVectorParams()
            },
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0)
        )
        print(f"Collection '{collection_name}' created with hybrid config and indexing disabled.")
    else:
        print(f"Collection '{collection_name}' already active.")

    batch_size = 32
    print(f"Pushing data to Qdrant container (Batch size = {batch_size})...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        batch_size=batch_size,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    mock_embed = MockEmbedding(embed_dim=1024)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=mock_embed,
        insert_batch_size=batch_size,
        show_progress=True,
    )

    print("Optimizing and finalizing collection index...")
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    )

    print("\n--> UPLOAD SUCCESS: Vectors and metadata are successfully indexed in Qdrant.")

if __name__ == "__main__":
    main()