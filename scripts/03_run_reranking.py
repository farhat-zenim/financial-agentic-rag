import sys
import os

# Ensure the src module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid import get_hybrid_retriever
from llama_index.core import QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank

def main():
    print("Initializing Hybrid Retriever (Semantic + BM25 Keyword)...")
    # Retrieve wide net of top 15 chunks from Qdrant[cite: 5]
    retriever = get_hybrid_retriever(collection_name="sec_10k_filings", top_k=15)

    print("\nLoading Cross-Encoder Reranker ('BAAI/bge-reranker-v2-m3') onto GPU...")
    # BAAI/bge-reranker-v2-m3 consumes ~2.2GB VRAM and performs full cross-attention scoring[cite: 5]
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=3
    )

    test_query = "What are the primary risk factors detailed in Item 1A regarding supply chain disruptions?"
    query_bundle = QueryBundle(query_str=test_query)

    print(f"\nExecuting Hybrid Query: '{test_query}'")
    # 1. Broad Retrieval (Fast hybrid search in Qdrant)
    initial_nodes = retriever.retrieve(test_query)
    print(f"Retrieved {len(initial_nodes)} initial blended chunks from Qdrant.")

    # 2. Deep Reranking (Cross-encoder deep contextual scoring)
    print("Re-scoring chunks through Cross-Encoder...")
    reranked_nodes = reranker.postprocess_nodes(
        nodes=initial_nodes,
        query_bundle=query_bundle
    )

    print(f"\nSuccessfully compressed context to top {len(reranked_nodes)} chunks.")

    # Display final filtered results
    for i, node in enumerate(reranked_nodes, 1):
        print(f"\n--- Final Rank {i} (Cross-Encoder Score: {node.score:.4f}) ---")
        print(f"{node.text[:250]}...")
        print(f"Metadata: {node.node.metadata}")

if __name__ == "__main__":
    main()