from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.wiki_loader import get_wikipedia_content, split_text
from app.retrieval import create_faiss_index, search_index
from app.qa import answer_question
from app.config import TOP_K

app = FastAPI()

# Allow CORS for browser JS fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")

# Serve index.html at root
@app.get("/")
def root():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")

@app.post("/query")
async def query_rag(request: Request):
    data = await request.json()
    topic = data.get("topic")
    question = data.get("question")

    result = get_wikipedia_content(topic)

    if result["status"] == "not_found":
        return {"error": f"Could not find any Wikipedia page for '{topic}'"}
    elif result["status"] == "ambiguous":
        return {"error": f"Topic is ambiguous. Options: {result['options']}"}
    elif result["status"] == "error":
        return {"error": f"Error: {result['message']}"}
    elif result["status"] == "ok":
        document = result["content"]
        resolved_title = result.get("title", topic)
    else:
        return {"error": "Unexpected response from Wikipedia fetch."}

    chunks = split_text(document)
    index, _ = create_faiss_index(chunks)
    retrieved_chunks = search_index(question, index, chunks, k=TOP_K)
    context = " ".join(retrieved_chunks)
    answer = answer_question(question, context)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "resolved_title": resolved_title
    }
