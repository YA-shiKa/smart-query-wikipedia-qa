# 🧠 Smart Query: RAG + LLM Wikipedia QA

A lightweight Retrieval-Augmented Generation (RAG) pipeline using FastAPI, Wikipedia, FAISS, Sentence Transformers, and a generative LLM (FLAN-T5) for intelligent question-answering.

## 🚀 Features

- 🔍 Fetches real-time Wikipedia content
- 🔗 Chunks and embeds text using `sentence-transformers`
- 🔎 Retrieves relevant context with FAISS
- 🤖 Answers questions using `google/flan-t5-large`
- 🌐 Minimal frontend with TailwindCSS and Dark Mode toggle

## 🖼️ Preview



https://github.com/user-attachments/assets/2d817d20-e4db-4f96-bd7d-5aa1f8f91bfe



## 🧪 Local Setup

```bash
# Clone the repo
git clone https://github.com/YA-shiKa/smart-query-wikipedia-qa.git
cd smart-query-wikipedia-qa

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
