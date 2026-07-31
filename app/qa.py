"""
Generation + self-critique for the Self-Healing RAG pipeline.

Three pieces live here:
  * answer_question   - generate an answer from retrieved context
  * critique_answer    - score how grounded that answer is in the context
                         (the "is this a hallucination?" check)
  * reformulate_query  - produce a different retrieval query when the critic
                         rejects an answer, so the retry actually looks
                         somewhere new instead of repeating the same search
"""

import re

import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from app.config import QA_MODEL, HALLUCINATION_SIMILARITY_THRESHOLD
from app.retrieval import embed_texts

_qa_pipeline = None


def get_qa_pipeline():
    """Lazily load the generation model. See retrieval.get_embedding_model
    for why this is lazy rather than eager at import time."""
    global _qa_pipeline
    if _qa_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)
        _qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return _qa_pipeline


def answer_question(question, context):
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, reply exactly with "
        "\"NOT_IN_CONTEXT\".\n\n"
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = get_qa_pipeline()(prompt, max_length=150, do_sample=False)[0]["generated_text"]
    return answer.strip()


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text):
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences


def critique_answer(answer, retrieved_chunks, threshold=None):
    """Score whether `answer` is actually grounded in `retrieved_chunks`.

    This is the "critic agent" step: rather than asking the same weak
    generation model to grade its own homework in free text (which is noisy
    and easy to hallucinate a "yes" on), we use two cheap, deterministic
    signals instead:

    1. Lexical containment -- does the answer text itself appear verbatim in
       the retrieved context? Short factual answers ("1879", "Guido van
       Rossum") from an extractive-style QA model are very often near-exact
       spans of the source text, and this catches them directly.
    2. Sentence-level embedding similarity -- compare the answer against
       each *individual sentence* within the retrieved chunks, not whole
       chunks. A whole ~256-token chunk covers many sentences/topics, so its
       pooled embedding dilutes the one specific fact the answer is about,
       which previously made even correct short answers score far below any
       reasonable threshold (observed: ~0.10-0.43 for verifiably correct
       answers). Comparing against sentences fixes that dilution.

    The answer is considered grounded if either signal fires.

    Returns a dict: {grounded: bool, best_score: float, reason: str}
    """
    threshold = HALLUCINATION_SIMILARITY_THRESHOLD if threshold is None else threshold

    if not answer or answer.strip().upper() == "NOT_IN_CONTEXT":
        return {
            "grounded": False,
            "best_score": 0.0,
            "reason": "Model explicitly signaled the context didn't contain the answer.",
        }

    if not retrieved_chunks:
        return {"grounded": False, "best_score": 0.0, "reason": "No retrieved chunks to check against."}

    answer_clean = answer.strip()
    context_lower = " ".join(retrieved_chunks).lower()
    lexical_hit = len(answer_clean) >= 2 and answer_clean.lower() in context_lower

    sentences = []
    for chunk in retrieved_chunks:
        sentences.extend(_split_into_sentences(chunk))
    if not sentences:
        sentences = retrieved_chunks

    answer_embedding = embed_texts([answer_clean])[0]
    sentence_embeddings = embed_texts(sentences)
    scores = [_cosine_similarity(answer_embedding, se) for se in sentence_embeddings]
    best_score = max(scores) if scores else 0.0

    similarity_hit = best_score >= threshold
    grounded = lexical_hit or similarity_hit
    reason = (
        f"Lexical match in context: {lexical_hit}. "
        f"Best sentence-level similarity {best_score:.2f} "
        f"({'>=' if similarity_hit else '<'} threshold {threshold:.2f})."
    )
    return {"grounded": grounded, "best_score": best_score, "reason": reason}


def reformulate_query(original_question, previous_query, failed_chunks):
    """Ask the model for a differently-worded retrieval query so a retry
    actually explores a different part of the vector store instead of
    re-running the exact same search and getting the exact same (wrong)
    chunks back."""
    seen_snippet = " ".join(failed_chunks)[:400]
    prompt = (
        "The following search query did not retrieve text that answers the question. "
        "Rewrite the query using different keywords or a different phrasing so a "
        "search engine might find better matching content. Reply with only the new "
        "query, nothing else.\n\n"
        f"Original question: {original_question}\n"
        f"Previous query: {previous_query}\n"
        f"Unhelpful retrieved text (for context, do not repeat it): {seen_snippet}\n\n"
        "New query:"
    )
    new_query = get_qa_pipeline()(prompt, max_length=64, do_sample=False)[0]["generated_text"].strip()
    # Guard against the model returning an empty string or literally echoing back
    # the same query, which would make the retry pointless.
    if not new_query or new_query.lower() == previous_query.lower():
        new_query = f"{original_question} background details facts"
    return new_query
