from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.config import EMBEDDING_MODEL, INDEX_CACHE_SIZE

_embedding_model = None


def get_embedding_model():
    """Lazily load the sentence-transformer so importing this module (and
    anything that depends on it, like app.graph) doesn't force a network
    call / model download until an embedding is actually needed. Makes
    startup faster and lets the routing/guardrail logic be unit-tested
    without a live model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts):
    """Shared embedding helper so retrieval and the critic use the exact
    same vector space when scoring groundedness."""
    return get_embedding_model().encode(texts)


def create_faiss_index(chunks):
    embeddings = embed_texts(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings


def search_index(query, index, chunks, k=3):
    query_embedding = embed_texts([query])
    k = min(k, len(chunks))
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]


class IndexCache:
    """Tiny in-memory LRU cache mapping a resolved Wikipedia title to its
    already-built (chunks, index) pair.

    Previously the app rebuilt embeddings for the *entire* article on every
    single question, even when re-asking about the same page (or when the
    self-healing loop re-retrieves for a reformulated query). That made every
    request pay the full embedding cost repeatedly. Caching per-topic keeps
    the retry loop fast and cheap.
    """

    def __init__(self, max_size=INDEX_CACHE_SIZE):
        self.max_size = max_size
        self._store = OrderedDict()

    def get(self, title):
        key = title.lower().strip()
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, title, chunks, index):
        key = title.lower().strip()
        self._store[key] = (chunks, index)
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)


index_cache = IndexCache()


def get_or_build_index(title, document, split_fn):
    """Return (chunks, index) for a resolved article title, building and
    caching it on first use."""
    cached = index_cache.get(title)
    if cached is not None:
        return cached
    chunks = split_fn(document)
    index, _ = create_faiss_index(chunks)
    index_cache.set(title, chunks, index)
    return chunks, index
