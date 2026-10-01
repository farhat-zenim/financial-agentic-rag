import sys
import os

# Ensure the src module is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid import get_hybrid_retriever

def main():
    print("Initializing Hybrid Retriever (Semantic + BM25 Keyword)...")

    # Retrieve top 15 chunks as dictated by the system blueprint for the upcoming cross-encoder[cite: 3]
    retriever = get_hybrid_retriever(collection_name="sec_10k_filings", top_k=15)

    test_query = "What are the primary risk factors detailed in Item 1A regarding supply chain disruptions?"
    print(f"\nExecuting Query: '{test_query}'")

    results = retriever.retrieve(test_query)

    print(f"\nSuccessfully retrieved {len(results)} blended chunks.")

    # Display the top 3 blended results
    for i, node in enumerate(results[:3], 1):
        print(f"\n--- Rank {i} (Combined RRF Score: {node.score:.4f}) ---")
        print(f"{node.text[:250]}...")
        print(f"Metadata: {node.metadata}")

if __name__ == "__main__":
    main()