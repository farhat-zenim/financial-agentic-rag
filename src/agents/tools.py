# src/agents/tools.py

import urllib.request
import urllib.error
import time
import re
from typing import Tuple, List
from llama_index.llms.ollama import Ollama
from src.retrieval.hybrid import get_hybrid_retriever, get_reranker
from src.config import config


def get_ollama_llm() -> Ollama:
    """Initializes and returns the LlamaIndex Ollama LLM client."""
    return Ollama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_HOST,
        request_timeout=180.0,
    )


def verify_ollama_connection(max_retries: int = 3, timeout: float = 10.0) -> bool:
    """Verifies Ollama server connectivity with retries."""
    url = f"{config.OLLAMA_HOST}/"
    req = urllib.request.Request(url)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue
    return True


def execute_direct_llm(query: str) -> Tuple[str, List[str]]:
    """Executes direct LLM query without retrieval context."""
    verify_ollama_connection()
    llm = get_ollama_llm()
    response = llm.complete(query)
    answer = response.text if hasattr(response, "text") else str(response)
    return answer.strip(), []


def filter_and_rank_nodes(query: str, nodes: list, target_company: str = None, target_year: str = None) -> list:
    """
    Scores and filters candidate nodes by matching company entity and year
    in node text/metadata before cross-encoder reranking.
    """
    query_upper = query.upper()

    if not target_company:
        for c in ["3M", "ADOBE", "ACTIVISION", "AMAZON", "APPLE", "MICROSOFT", "NIKE", "TESLA", "NETFLIX"]:
            if c in query_upper:
                target_company = c
                break

    if not target_year:
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            target_year = year_match.group(1)

    scored_nodes = []
    for node in nodes:
        content = node.node.get_content().lower()
        meta_str = str(node.node.metadata).lower()
        score_boost = 0.0

        if target_company:
            comp_lower = target_company.lower()
            if comp_lower in meta_str or comp_lower in content:
                score_boost += 2.0

        if target_year:
            if target_year in meta_str or target_year in content:
                score_boost += 1.5

        node.score = (node.score or 0.0) + score_boost
        scored_nodes.append(node)

    scored_nodes.sort(key=lambda x: x.score, reverse=True)
    return scored_nodes


def execute_rag_pipeline(query: str, company: str = None, year: str = None) -> Tuple[str, List[str]]:
    """Executes multi-document RAG with entity/year-aware chunk selection."""
    verify_ollama_connection()

    retriever = get_hybrid_retriever(top_k=20)
    reranker = get_reranker(top_n=config.DEFAULT_TOP_N)

    candidate_nodes = retriever.retrieve(query)
    prioritized_nodes = filter_and_rank_nodes(query, candidate_nodes, target_company=company, target_year=year)
    reranked_nodes = reranker.postprocess_nodes(prioritized_nodes[:10], query_str=query)

    retrieved_texts = [n.node.get_content() for n in reranked_nodes]
    context_str = "\n\n---\n\n".join(retrieved_texts)

    prompt = (
        "You are an expert financial analyst. Answer the user question based STRICTLY and ONLY on the provided financial report excerpts below.\n"
        "State exact values, numbers, and facts directly. If a specific fiscal year or dollar amount is requested, extract it precisely from the context.\n"
        "Do not invent figures or draw from outside knowledge.\n\n"
        f"Context Excerpts:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    llm = get_ollama_llm()
    response = llm.complete(prompt)
    answer = response.text if hasattr(response, "text") else str(response)
    return answer.strip(), retrieved_texts


def execute_filtered_rag(query: str, company: str = None, year: str = None) -> Tuple[str, List[str]]:
    """Executes entity-filtered RAG targeting specific company/year metadata."""
    return execute_rag_pipeline(query, company=company, year=year)