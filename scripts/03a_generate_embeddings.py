import json
import pickle
import uuid
from pathlib import Path

from llama_index.core.schema import TextNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding

def load_nodes_from_jsonl(file_path: Path) -> list[TextNode]:
    nodes = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)

            text_content = chunk.get("text_for_embedding") or chunk.get("text", "")
            chunk_id = str(chunk.get("chunk_id", ""))
            qdrant_compliant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "company": chunk.get("company"),
                "fiscal_year": chunk.get("fiscal_year"),
                "section": chunk.get("section"),
                "page_no": chunk.get("page_no"),
                "raw_text": chunk.get("text"),
            }

            nodes.append(TextNode(text=text_content, id_=qdrant_compliant_id, metadata=metadata))
    return nodes

def main():
    # Define paths relative to the project root
    chunks_dir = Path("parsed_docs/chunks")
    cache_path = Path("data/processed/embedded_nodes.pkl")
    viz_path = Path("data/processed/embedding_sample.json")

    jsonl_files = list(chunks_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        print(f"Error: No JSONL files found in {chunks_dir}")
        return

    print("Loading raw chunks from disk...")
    all_nodes = []
    for file_path in jsonl_files:
        all_nodes.extend(load_nodes_from_jsonl(file_path))

    print(f"Loaded {len(all_nodes)} nodes. Initializing FastEmbed (BAAI/bge-large-en-v1.5)...")
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5", max_length=512, batch_size=16)

    texts = [node.text for node in all_nodes]
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    for node, embedding in zip(all_nodes, embeddings):
        node.embedding = embedding

    # Save complete Python objects for the next stage
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(all_nodes, f)

    # Save a human-readable JSON sample for verification
    sample_data = {
        "id": all_nodes[0].id_,
        "metadata": all_nodes[0].metadata,
        "vector_preview": all_nodes[0].embedding[:10]
    }
    with open(viz_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4)

    print(f"\n--> SUCCESS: {len(all_nodes)} nodes embedded.")
    print(f"--> Binary Cache: {cache_path}")
    print(f"--> Visual Sample: {viz_path}")

if __name__ == "__main__":
    main()