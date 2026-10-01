import sys
import time
import warnings
from pathlib import Path

# Resolve project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(override=True)

from src.retrieval.hybrid import preload_models
from src.agents.router import route_and_execute


def main():
    print("🚀 Initializing Enterprise RAG Pipeline Execution Suite...")

    # Model pre-warming to avoid runtime delays during test execution
    preload_models()

    test_suite = [
        "What is the definition of gross margin?",
        "What were Adobe's major risk factors in 2022?",
        "Compare revenue growth across all available tech filings."
    ]

    print("\nStarting Test Pipeline Runs...")

    for i, query in enumerate(test_suite, 1):
        print(f"\n================ TEST RUN {i} ================")
        print(f"Query: '{query}'")

        start_time = time.time()
        try:
            output = route_and_execute(query)
            elapsed = time.time() - start_time
            print(f"\nResponse (Completed in {elapsed:.2f}s):\n\n{output}\n")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Test Run {i} Failed after {elapsed:.2f}s: {str(e)}")


if __name__ == "__main__":
    main()