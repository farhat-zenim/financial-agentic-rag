# src/agents/router.py

import re
from typing import Tuple, List
from src.agents.tools import execute_direct_llm, execute_rag_pipeline, execute_filtered_rag


def route_and_execute(query: str) -> Tuple[str, str, List[str]]:
    """
    Classifies intent, extracts entity metadata, and executes the optimal retrieval path.
    Returns: (route_name, answer, context_texts)
    """
    query_upper = query.upper()

    company = None
    for c in ["3M", "ADOBE", "ACTIVISION", "AMAZON", "APPLE", "MICROSOFT", "NIKE", "TESLA", "NETFLIX"]:
        if c in query_upper:
            company = c
            break

    year_match = re.search(r'\b(20\d{2})\b', query)
    year = year_match.group(1) if year_match else None

    if company or year:
        answer, contexts = execute_filtered_rag(query, company=company, year=year)
        return "FILTERED_RAG", answer, contexts

    if any(term in query_upper for term in ["CAPITAL EXPENDITURE", "MARGIN", "SEGMENT", "REVENUE", "FY20", "BALANCE SHEET"]):
        answer, contexts = execute_rag_pipeline(query)
        return "GLOBAL_RAG", answer, contexts

    answer, contexts = execute_direct_llm(query)
    return "DIRECT_LLM", answer, contexts