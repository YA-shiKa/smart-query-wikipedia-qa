import os
from pathlib import Path

# --- Retrieval / chunking ---
WIKI_CHUNK_SIZE = int(os.getenv("WIKI_CHUNK_SIZE", 256))
WIKI_CHUNK_OVERLAP = int(os.getenv("WIKI_CHUNK_OVERLAP", 20))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = int(os.getenv("TOP_K", 3))

# --- Generation ---
QA_MODEL = os.getenv("QA_MODEL", "google/flan-t5-large")

# --- Self-healing loop ---
# Fallback signal used by the critic when the answer isn't a verbatim
# substring of the retrieved context (see app/qa.py:critique_answer, which
# also checks lexical containment first). This is now sentence-level cosine
# similarity, not whole-chunk similarity -- whole ~256-token chunks cover
# many sentences/topics and dilute the similarity of one specific fact.
HALLUCINATION_SIMILARITY_THRESHOLD = float(os.getenv("HALLUCINATION_SIMILARITY_THRESHOLD", 0.4))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 2))

# --- Index cache ---
# How many (topic -> FAISS index) entries to keep in memory so repeated
# questions about the same page don't re-embed the whole article every time.
INDEX_CACHE_SIZE = int(os.getenv("INDEX_CACHE_SIZE", 32))

# --- Guardrails ---
POLICY_PATH = Path(os.getenv("POLICY_PATH", Path(__file__).parent.parent / "policy.yaml")) 