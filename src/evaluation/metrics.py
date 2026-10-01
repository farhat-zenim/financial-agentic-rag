# src/evaluation/metrics.py

from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Use ChatOllama for native JSON mode enforcement
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_community.embeddings import FastEmbedEmbeddings
from src.config import config


def run_ragas_evaluation(dataset_dict: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Executes RAGAs evaluation sequentially using local Ollama in strict JSON mode.
    """
    eval_dataset = Dataset.from_dict(dataset_dict)

    # Force ChatOllama into strict JSON mode and deterministic temperature
    raw_llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_HOST,
        format="json",
        temperature=0.0,
    )
    eval_llm = LangchainLLMWrapper(raw_llm)

    raw_embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    eval_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Assign wrapped models to RAGAs metric objects
    for metric in metrics:
        metric.llm = eval_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = eval_embeddings

    # Enforce sequential execution (max_workers=1) to eliminate GPU memory throttling
    run_config = RunConfig(
        max_workers=1,
        timeout=240,
        max_retries=3,
    )

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=run_config,
    )

    # Robust output extraction across RAGAs versions
    if hasattr(results, "to_pandas"):
        df = results.to_pandas()
        metric_cols = [c for c in df.columns if c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
        return {col: float(df[col].mean()) for col in metric_cols}

    return dict(results)