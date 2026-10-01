import os
import urllib.request
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini


def is_ollama_running(url: str = "http://localhost:11434/api/tags") -> bool:
    """Checks if local Ollama server is running on localhost."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def get_llm():
    """
    Returns local Ollama using qwen2.5-coder:7b (0 Quota / Unlimited Calls).
    Falls back to Gemini if Ollama is offline.
    """
    if is_ollama_running():
        print("💻 [LLM Engine] Active Local Ollama detected! Using 'qwen2.5-coder:7b' (0 Quotas / Unlimited).")
        return Ollama(
            model="qwen2.5-coder:7b",
            request_timeout=120.0
        )

    print("☁️ [LLM Engine] Ollama not detected. Falling back to Gemini API...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set in .env")

    return Gemini(model_name="models/gemini-2.0-flash", api_key=api_key)


def generate_answer(query_str: str, nodes: list) -> str:
    """Generates an answer using retrieved context nodes."""
    context_str = "\n\n".join([f"--- Chunk {idx+1} ---\n{node.text}" for idx, node in enumerate(nodes)])

    prompt = (
        "You are an expert financial analyst. Answer the user question concisely and accurately based strictly on the provided SEC 10-K filing context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query_str}\n"
        "Answer:"
    )

    llm = get_llm()
    response = llm.complete(prompt)
    return str(response)