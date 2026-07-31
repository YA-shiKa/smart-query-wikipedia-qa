# Self-Healing Wikipedia RAG

A Retrieval-Augmented Generation app over live Wikipedia content that
doesn't just retrieve-and-generate: it **critiques its own answer for
groundedness, retries with a reformulated query if it looks like a
hallucination, and admits when it doesn't know** rather than guessing.

Built with FastAPI, sentence-transformers + FAISS, `flan-t5-large`, and
[LangGraph](https://github.com/langchain-ai/langgraph) for the cyclical
workflow.

## How it works

```
input_guardrail → retrieve → generate → critique → decide
                      ▲                                │
                      └──────────── retry ──────────────┤
                                                          ▼
                                          output_guardrail → done
```

1. **input_guardrail** — blocks prompt injection, PII, and disallowed topics
   before anything reaches an LLM (`app/security.py`, rules in
   `policy.yaml`).
2. **retrieve** — fetches the Wikipedia page (once, then cached) and does a
   FAISS similarity search for the current query.
3. **generate** — `flan-t5-large` answers from the retrieved chunks only.
4. **critique** — the "critic agent": scores cosine similarity between the
   answer and its best-matching retrieved chunk. Below a threshold, the
   answer is treated as ungrounded / likely hallucinated (`app/qa.py:critique_answer`).
5. **decide** — if grounded, done. If not, and retries remain, the query is
   reformulated (`app/qa.py:reformulate_query`) and the loop goes back to
   `retrieve`. If retries are exhausted, the system returns "I don't have
   enough information" instead of making something up.
6. **output_guardrail** — final answer is checked against policy (e.g. no
   medical-advice phrasing, no leaked PII) before it's returned.

This is modeled as a real stateful graph in `app/graph.py` (`langgraph.graph.StateGraph`),
not a linear chain — `decide` can route back to `retrieve` an arbitrary
number of times up to `max_retry_attempts`.

## Guardrails gateway

`app/security.py` + `policy.yaml` implement a small guardrails layer:

- **Input guardrails**: prompt-injection/jailbreak heuristics, PII regexes
  (credit card, SSN, email), and a topic blocklist.
- **Output guardrails**: banned-phrase checks (e.g. the policy explicitly
  blocks the model from phrasing anything as medical advice — dosages,
  diagnoses — since Wikipedia health content is informational, not a
  prescription) and PII-leak checks on the generated answer.
- **Policy engine**: all of the above is configured in `policy.yaml`, which
  a non-engineer can edit without touching Python. The app reloads it
  automatically when the file changes (mtime-based cache).

This is intentionally scoped to what this single-purpose app needs, not a
general-purpose LLM gateway product — but the pattern (rules file in,
allow/block decision + reason out) generalizes directly if you wanted to
sit it in front of other models.

## Eval harness

`eval/` contains a starter "LLM eval CI" setup:

- `eval/golden_dataset.json` — ~18 curated (topic, question, expected
  keywords) pairs, including a few that should be correctly *refused*
  rather than answered. (A production system should grow this to 100+ and
  cover more edge cases — this is a seed, not the finished set.)
- `eval/run_eval.py` — runs the golden set straight through the graph (no
  server needed) and reports accuracy proxy, correct-refusal rate,
  hallucination rate (critic says "ungrounded" but the pipeline returned it
  anyway — shouldn't happen given the graph logic, but it's a useful
  regression signal if someone loosens the threshold), retry-attempt
  distribution, and p50/p95 latency. Exits non-zero if the hallucination
  rate or p95 latency breaches the thresholds at the top of the file.
- `.github/workflows/eval.yml` — runs `eval/run_eval.py` on any push/PR that
  touches `app/`, `policy.yaml`, or `eval/`, and uploads `eval/results.json`
  as a build artifact so you can track metrics over time across runs.

Run it locally:

```bash
python -m eval.run_eval
```

## Project layout

```
app/
  main.py       FastAPI app: /, /health, /query
  graph.py      LangGraph self-healing RAG workflow
  qa.py         generation, critique (groundedness scoring), query reformulation
  retrieval.py  embeddings, FAISS index, per-topic index cache
  wiki_loader.py Wikipedia fetch + chunking
  security.py   guardrails gateway (reads policy.yaml)
  config.py     all tunables, env-var overridable
policy.yaml     guardrail rules (edit without touching code)
eval/           golden dataset + eval runner
tests/          unit tests for guardrails + graph control flow (mocked models, no downloads)
static/         minimal HTML/JS frontend
.github/workflows/eval.yml   CI gate on eval metrics
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://localhost:8000`. Enter a topic (e.g. "Albert Einstein")
and a question. The UI shows whether the answer was grounded on the first
try, how many self-healing retries it took, and (if it gave up) why.

Models are loaded lazily on first request, not at import time, so `import
app.main` / running the test suite doesn't require network access or a
model download.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All tests mock the LLM/embedding calls, so they run in a couple seconds
without downloading `flan-t5-large` or the sentence-transformer.

## Configuration

Everything in `app/config.py` is overridable via environment variables,
e.g.:

```bash
export MAX_RETRY_ATTEMPTS=3
export HALLUCINATION_SIMILARITY_THRESHOLD=0.5
export QA_MODEL=google/flan-t5-base   # smaller/faster model
```

Guardrail behavior (blocked topics, banned phrases, PII patterns, retry
budget, hallucination threshold) is configured separately in `policy.yaml`.

## What changed from the original version

This started as a straightforward retrieve → generate demo. The main
additions:

- **Self-healing loop**: a real LangGraph state machine with a critic agent
  and retry-with-reformulation, instead of returning whatever the model
  said on the first pass.
- **Guardrails gateway**: input/output checks against a configurable
  `policy.yaml`, including an explicit rule blocking medical-advice framing.
- **Eval harness + CI gate**: a golden dataset, a runnable eval script with
  pass/fail thresholds, and a GitHub Actions workflow wired to it.
- **Performance/correctness fixes**: per-topic FAISS index caching (the
  original rebuilt embeddings for the whole article on every request), lazy
  model loading (faster cold start, testable without network access), the
  generation call moved off the event loop via `run_in_threadpool`, and a
  `k = min(k, len(chunks))` guard so short articles don't crash the search.
- **Tests**: unit tests for the guardrails and for the graph's control flow
  (retry, give-up, guardrail short-circuit), all running against mocked
  models so they're fast and don't need network access.

## Notes / limitations

- The critic is a **similarity-based heuristic**, not a second LLM call
  asking "was this grounded?" in free text. That's deliberate: it's cheap,
  deterministic, and doesn't have its own hallucination risk the way a
  second free-text LLM judgment would — but it's still a proxy, not a
  perfect faithfulness metric.
- `flan-t5-large` is a relatively weak instruction-follower for both
  generation and query reformulation; swapping `QA_MODEL` for a stronger
  model (local or via API) will materially improve both answer quality and
  the eval numbers.
- The golden dataset is a seed (~18 examples), not the 100+ production-scale
  set a real eval-gated CI pipeline should have.
