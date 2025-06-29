# 🧠 Smart Query: RAG + LLM Wikipedia QA

A lightweight Retrieval-Augmented Generation (RAG) pipeline using FastAPI, Wikipedia, FAISS, Sentence Transformers, and a generative LLM (FLAN-T5) for intelligent question-answering.

## 🚀 Features

- 🔍 Fetches real-time Wikipedia content
- 🔗 Chunks and embeds text using `sentence-transformers`
- 🔎 Retrieves relevant context with FAISS
- 🤖 Answers questions using `google/flan-t5-large`
- 🌐 Minimal frontend with TailwindCSS and Dark Mode toggle


## 🧪 Local Setup

```bash
# Clone the repo
git clone https://github.com/your-username/rag-pipeline.git
cd rag-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app.main:app --reload
Then open http://localhost:8000 in your browser.
```

## ✨ Example Query
- Topic: Apple Inc.

- Question: Who is the current CEO of Apple?

## 📦 Models Used

- Embedding: sentence-transformers/all-mpnet-base-v2

- QA Model: google/flan-t5-large
