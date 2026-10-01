# scripts/05_evaluate_ragas.py

import sys
import types
from pathlib import Path

# Add project root to Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# --- LangChain Compatibility Patch ---
try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    m.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = m

try:
    import langchain_community.llms.vertexai
except ModuleNotFoundError:
    m = types.ModuleType("langchain_community.llms.vertexai")
    m.VertexAI = type("VertexAI", (object,), {})
    sys.modules["langchain_community.llms.vertexai"] = m
# --------------------------------------

import json
from src.agents.router import route_and_execute
from src.evaluation.metrics import run_ragas_evaluation

BENCHMARK_PATH = ROOT_DIR / "data" / "benchmarks" / "jsons" / "financebench_open_source.jsonl"


def load_financebench_samples(limit: int = 3):
    """Loads benchmark evaluation samples from FinanceBench."""
    samples = []
    if not BENCHMARK_PATH.exists():
        print(f"❌ Benchmark file not found at: {BENCHMARK_PATH}")
        return samples

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def run_evaluation():
    print("🚀 Starting RAGAs Evaluation (Synchronized Context Mode)...")
    samples = load_financebench_samples(limit=3)

    if not samples:
        print("❌ No evaluation samples loaded. Aborting evaluation.")
        return

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for idx, sample in enumerate(samples, 1):
        query = sample.get("question") or sample.get("query", "")
        ground_truth = sample.get("answer") or sample.get("ground_truth", "")

        print(f"\n[{idx}/{len(samples)}] Processing query: '{query[:60]}...'")

        # Route query and extract generated answer AND retrieved contexts simultaneously
        route_name, generated_answer, retrieved_contexts = route_and_execute(query)
        print(f"   🔀 Route Selected: {route_name} | Retracted Context Chunks: {len(retrieved_contexts)}")

        questions.append(query)
        answers.append(generated_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    eval_payload = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    print("\n📊 Computing RAGAs Metrics (Faithfulness, Relevancy, Precision, Recall)...")
    results = run_ragas_evaluation(eval_payload)

    print("\n=================== RAGAS RESULTS ===================")
    if isinstance(results, dict):
        for metric_name, score in results.items():
            print(f" • {metric_name:22s}: {score:.4f}")
    else:
        print(results)
    print("=====================================================")


if __name__ == "__main__":
    run_evaluation()