# src/api/main.py

from fastapi import FastAPI, Body, HTTPException
from src.agents.router import route_and_execute

app = FastAPI(
    title="Financial 10-K RAG API",
    description="Backend API for hybrid RAG retrieval and evaluation."
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "RAG Backend Engine"}

@app.post("/query")
def process_query(payload: dict = Body(...)):
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query string is required in JSON payload.")

    try:
        route_name, answer, retrieved_contexts = route_and_execute(query)
        return {
            "route": route_name,
            "answer": answer,
            "contexts": retrieved_contexts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))