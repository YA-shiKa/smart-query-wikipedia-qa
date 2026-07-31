"""
LLM Guardrails Gateway
-----------------------
A small middleware layer that sits between the user and the RAG pipeline.

* Input guardrails  -> block prompt-injection/jailbreak attempts, PII leakage
  (e.g. a pasted credit card number), and hard-blocked topics.
* Output guardrails  -> make sure the final answer doesn't contain banned
  phrases (e.g. medical advice) and that citations are attached when the
  policy requires them.

The rules themselves live in policy.yaml so non-engineers can change what's
allowed without touching Python.
"""

import re
import functools
from dataclasses import dataclass, field
from typing import Optional

import yaml

from app.config import POLICY_PATH


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    # Which category tripped, e.g. "blocked_topic", "prompt_injection", "pii", "banned_phrase"
    category: Optional[str] = None


@dataclass
class Policy:
    blocked_topics: list = field(default_factory=list)
    banned_output_phrases: list = field(default_factory=list)
    require_citation: bool = True
    max_retry_attempts: int = 2
    hallucination_similarity_threshold: float = 0.45
    injection_patterns: list = field(default_factory=list)
    pii_patterns: dict = field(default_factory=dict)


def _default_policy() -> Policy:
    # Sane fallback if policy.yaml is missing or fails to parse, so the app
    # degrades to "reasonably safe defaults" instead of crashing or wide open.
    return Policy(
        blocked_topics=[],
        banned_output_phrases=["recommended dosage", "your diagnosis is"],
        require_citation=True,
        max_retry_attempts=2,
        hallucination_similarity_threshold=0.45,
        injection_patterns=[r"ignore (all|the)? ?previous instructions"],
        pii_patterns={
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        },
    )


@functools.lru_cache(maxsize=1)
def _load_policy_cached(mtime: float) -> Policy:
    try:
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Policy(
            blocked_topics=[t.lower() for t in raw.get("blocked_topics", [])],
            banned_output_phrases=[p.lower() for p in raw.get("banned_output_phrases", [])],
            require_citation=raw.get("require_citation", True),
            max_retry_attempts=raw.get("max_retry_attempts", 2),
            hallucination_similarity_threshold=raw.get("hallucination_similarity_threshold", 0.45),
            injection_patterns=raw.get("injection_patterns", []),
            pii_patterns=raw.get("pii_patterns", {}),
        )
    except (FileNotFoundError, yaml.YAMLError):
        return _default_policy()


def load_policy() -> Policy:
    """Reload policy.yaml if it changed on disk (cheap mtime check) so edits
    take effect without restarting the server."""
    try:
        mtime = POLICY_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = -1.0
    return _load_policy_cached(mtime)


def check_input(topic: str, question: str, policy: Optional[Policy] = None) -> GuardrailResult:
    """Run all input guardrails against the raw topic + question."""
    policy = policy or load_policy()
    combined = f"{topic or ''} {question or ''}"
    combined_lower = combined.lower()

    for blocked in policy.blocked_topics:
        if blocked in combined_lower:
            return GuardrailResult(
                allowed=False,
                category="blocked_topic",
                reason="This topic is not permitted by the site's usage policy.",
            )

    for pattern in policy.injection_patterns:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return GuardrailResult(
                allowed=False,
                category="prompt_injection",
                reason="Your input looks like it's trying to override the system's instructions, "
                       "so it was blocked.",
            )

    for pii_type, pattern in policy.pii_patterns.items():
        if re.search(pattern, combined):
            return GuardrailResult(
                allowed=False,
                category="pii",
                reason=f"Your input appears to contain sensitive personal information ({pii_type}). "
                       "Please remove it and try again.",
            )

    return GuardrailResult(allowed=True)


def check_output(answer: str, policy: Optional[Policy] = None) -> GuardrailResult:
    """Run output guardrails against the model's generated answer."""
    policy = policy or load_policy()
    answer_lower = (answer or "").lower()

    for phrase in policy.banned_output_phrases:
        if phrase in answer_lower:
            return GuardrailResult(
                allowed=False,
                category="banned_phrase",
                reason="The generated answer was withheld because it resembled disallowed content "
                       "(e.g. medical advice).",
            )

    for pii_type, pattern in policy.pii_patterns.items():
        if re.search(pattern, answer or ""):
            return GuardrailResult(
                allowed=False,
                category="pii",
                reason=f"The generated answer was withheld because it appeared to contain {pii_type} data.",
            )

    return GuardrailResult(allowed=True)
