from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.config import EMBEDDING_MODEL

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def search_index(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]
