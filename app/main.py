import logging

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pathlib import Path

from app.graph import run_pipeline

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Self-Healing Wikipedia RAG")

# Allow CORS for browser JS fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query_rag(request: Request):
    data = await request.json()
    topic = (data.get("topic") or "").strip()
    question = (data.get("question") or "").strip()

    if not topic or not question:
        return {"error": "Both 'topic' and 'question' are required."}

    # The pipeline (embedding + generation calls) is synchronous/CPU-bound;
    # run it off the event loop so one slow request doesn't block others.
    final_state = await run_in_threadpool(run_pipeline, topic, question)

    status = final_state.get("status")

    if status == "blocked":
        return {"error": final_state.get("reason", "Request blocked by guardrails.")}
    if status == "error":
        return {"error": final_state.get("reason", "Unexpected error.")}
    if status == "insufficient_info":
        return {
            "answer": final_state.get("reason"),
            "insufficient_info": True,
            "resolved_title": final_state.get("resolved_title"),
            "attempts": final_state.get("attempts"),
            "trace": final_state.get("trace", []),
        }

    return {
        "answer": final_state.get("answer"),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "resolved_title": final_state.get("resolved_title"),
        "attempts": final_state.get("attempts"),
        "critique": final_state.get("critique"),
        "trace": final_state.get("trace", []),
    }
