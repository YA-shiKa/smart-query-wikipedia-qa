"""
The Self-Healing RAG pipeline, modeled as a stateful, cyclical LangGraph
workflow instead of a linear retrieve -> generate chain.

    input_guardrail -> retrieve -> generate -> critique -> decide
                            ^                                 |
                            |___________ retry ________________|
                                                                |
                                                          output_guardrail -> END

* input_guardrail   blocks prompt injection / PII / disallowed topics before
                    anything is sent to an LLM.
* retrieve          fetches the Wikipedia page (once) and searches the
                    (cached) FAISS index for the current query.
* generate          produces an answer from the retrieved chunks.
* critique          the critic agent: is the answer actually grounded in the
                    retrieved chunks, or did the model hallucinate?
* decide            routes to another retrieve/generate/critique cycle with a
                    reformulated query, or gives up gracefully.
* output_guardrail  checks the final answer against policy before it's
                    returned to the user.
"""

from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from app.wiki_loader import get_wikipedia_content, split_text
from app.retrieval import get_or_build_index, search_index
from app.qa import answer_question, critique_answer, reformulate_query
from app.security import check_input, check_output, load_policy
from app.config import TOP_K


class RAGState(TypedDict, total=False):
    topic: str
    original_question: str
    question: str
    resolved_title: Optional[str]
    document: Optional[str]
    chunks: List[str]
    index: object
    retrieved_chunks: List[str]
    answer: str
    critique: dict
    attempts: int
    max_attempts: int
    status: str          # "ok" | "insufficient_info" | "blocked" | "error"
    reason: Optional[str]
    trace: List[dict]     # one entry per attempt, for UI transparency


def _input_guardrail_node(state: RAGState) -> RAGState:
    result = check_input(state.get("topic", ""), state.get("original_question", ""))
    if not result.allowed:
        state["status"] = "blocked"
        state["reason"] = result.reason
        state["trace"] = state.get("trace", []) + [{"stage": "input_guardrail", "blocked": True, "category": result.category}]
    return state


def _route_after_input_guardrail(state: RAGState) -> str:
    return "end" if state.get("status") == "blocked" else "retrieve"


def _retrieve_node(state: RAGState) -> RAGState:
    # Fetch the Wikipedia page once and cache it on state; every retry only
    # needs to re-search, not re-fetch/re-embed the article.
    if state.get("document") is None:
        result = get_wikipedia_content(state["topic"])
        if result["status"] != "ok":
            state["status"] = "error"
            if result["status"] == "not_found":
                state["reason"] = f"Could not find any Wikipedia page for '{state['topic']}'"
            elif result["status"] == "ambiguous":
                state["reason"] = f"Topic is ambiguous. Options: {result['options']}"
            else:
                state["reason"] = result.get("message", "Unknown error fetching Wikipedia content.")
            return state
        state["document"] = result["content"]
        state["resolved_title"] = result.get("title", state["topic"])

    chunks, index = get_or_build_index(state["resolved_title"], state["document"], split_text)
    state["chunks"] = chunks
    state["index"] = index
    state["retrieved_chunks"] = search_index(state["question"], index, chunks, k=TOP_K)
    return state


def _route_after_retrieve(state: RAGState) -> str:
    return "end" if state.get("status") == "error" else "generate"


def _generate_node(state: RAGState) -> RAGState:
    context = " ".join(state["retrieved_chunks"])
    state["answer"] = answer_question(state["question"], context)
    return state


def _critique_node(state: RAGState) -> RAGState:
    policy = load_policy()
    critique = critique_answer(
        state["answer"],
        state["retrieved_chunks"],
        threshold=policy.hallucination_similarity_threshold,
    )
    state["critique"] = critique
    state["trace"] = state.get("trace", []) + [{
        "stage": "attempt",
        "attempt": state["attempts"] + 1,
        "query": state["question"],
        "answer": state["answer"],
        "grounded": critique["grounded"],
        "reason": critique["reason"],
    }]
    state["attempts"] += 1
    return state


def _decide_node(state: RAGState) -> RAGState:
    if state["critique"]["grounded"]:
        state["status"] = "ok"
    elif state["attempts"] >= state["max_attempts"]:
        state["status"] = "insufficient_info"
        state["reason"] = (
            "I don't have enough reliably-sourced information in this article to answer that "
            "confidently, so I won't guess."
        )
    else:
        # Not grounded yet, and we still have retries left: reformulate and loop.
        state["question"] = reformulate_query(
            state["original_question"], state["question"], state["retrieved_chunks"]
        )
    return state


def _route_after_decide(state: RAGState) -> str:
    if state.get("status") in ("ok", "insufficient_info"):
        return "output_guardrail"
    return "retrieve"


def _output_guardrail_node(state: RAGState) -> RAGState:
    if state.get("status") != "ok":
        return state
    result = check_output(state["answer"])
    if not result.allowed:
        state["status"] = "blocked"
        state["reason"] = result.reason
        state["trace"] = state.get("trace", []) + [{"stage": "output_guardrail", "blocked": True, "category": result.category}]
    return state


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("input_guardrail", _input_guardrail_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("critique_step", _critique_node)
    graph.add_node("decide", _decide_node)
    graph.add_node("output_guardrail", _output_guardrail_node)

    graph.set_entry_point("input_guardrail")
    graph.add_conditional_edges("input_guardrail", _route_after_input_guardrail, {"end": END, "retrieve": "retrieve"})
    graph.add_conditional_edges("retrieve", _route_after_retrieve, {"end": END, "generate": "generate"})
    graph.add_edge("generate", "critique_step")
    graph.add_edge("critique_step", "decide")
    graph.add_conditional_edges(
        "decide", _route_after_decide, {"retrieve": "retrieve", "output_guardrail": "output_guardrail"}
    )
    graph.add_edge("output_guardrail", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(topic: str, question: str, max_attempts: Optional[int] = None) -> RAGState:
    policy = load_policy()
    initial_state: RAGState = {
        "topic": topic,
        "original_question": question,
        "question": question,
        "attempts": 0,
        "max_attempts": max_attempts if max_attempts is not None else policy.max_retry_attempts,
        "trace": [],
        "status": "running",
    }
    graph = get_graph()
    final_state = graph.invoke(initial_state, {"recursion_limit": 50})
    return final_state
