from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle


def rerank_nodes(query_str: str, nodes: list, top_n: int = 4):
    """
    Reranks retrieved node candidates using BAAI/bge-reranker-v2-m3.
    """
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=top_n
    )

    query_bundle = QueryBundle(query_str=query_str)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle=query_bundle)

    return reranked_nodes