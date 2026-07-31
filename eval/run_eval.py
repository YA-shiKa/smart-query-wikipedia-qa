"""
LLM Eval pipeline (scoped for this project's size).

Runs the golden dataset straight through the LangGraph pipeline (no HTTP
server needed) and reports:
  * answer accuracy proxy  - did the answer contain an expected keyword?
  * faithfulness           - critic's grounded/ungrounded verdict
  * correct refusal rate   - for questions that should NOT be answerable,
                             did the system correctly say "I don't know"
                             instead of hallucinating?
  * latency p50 / p95
  * attempts distribution  - how often the self-healing retry loop kicked in

Exits non-zero (so it can gate a CI merge) if:
  * hallucination rate  > policy threshold, or
  * latency p95         > SLA_LATENCY_SECONDS

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --dataset eval/golden_dataset.json --out eval/results.json

Note: this is a starter harness with ~18 curated examples, not the 100+
production-scale golden set described in the project brief -- growing the
dataset (and wiring this script into a real CI workflow, see
.github/workflows/eval.yml) is the natural next step before relying on it to
gate merges for real.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Allow running as `python -m eval.run_eval` or `python eval/run_eval.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import run_pipeline  # noqa: E402

HALLUCINATION_RATE_THRESHOLD = 0.05  # merge-blocking threshold, mirrors policy.yaml intent
SLA_LATENCY_SECONDS = 30.0


def keyword_hit(answer: str, expected_keywords) -> bool:
    if not expected_keywords:
        return True
    answer_lower = (answer or "").lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def run(dataset_path: str, out_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    latencies = []

    for item in dataset:
        start = time.perf_counter()
        state = run_pipeline(item["topic"], item["question"])
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        status = state.get("status")
        answer = state.get("answer", "")
        should_answer = item.get("should_answer", True)

        if should_answer:
            correct = status == "ok" and keyword_hit(answer, item.get("expected_keywords", []))
            correct_refusal = None
        else:
            correct = status == "insufficient_info"
            correct_refusal = correct

        results.append({
            "topic": item["topic"],
            "question": item["question"],
            "status": status,
            "answer": answer,
            "attempts": state.get("attempts"),
            "grounded": (state.get("critique") or {}).get("grounded"),
            "correct": correct,
            "correct_refusal": correct_refusal,
            "latency_s": round(elapsed, 3),
        })

    answerable = [r for r in results if r["correct_refusal"] is None]
    unanswerable = [r for r in results if r["correct_refusal"] is not None]

    accuracy = (sum(r["correct"] for r in answerable) / len(answerable)) if answerable else 1.0
    refusal_accuracy = (sum(r["correct_refusal"] for r in unanswerable) / len(unanswerable)) if unanswerable else 1.0
    hallucination_rate = (
        sum(1 for r in answerable if r["status"] == "ok" and not r["grounded"]) / len(answerable)
    ) if answerable else 0.0
    avg_attempts = statistics.mean(r["attempts"] for r in results if r["attempts"] is not None)

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    summary = {
        "n_examples": len(results),
        "accuracy_proxy": round(accuracy, 3),
        "correct_refusal_rate": round(refusal_accuracy, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "avg_retry_attempts": round(avg_attempts, 2),
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
    }

    Path(out_path).write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print(json.dumps(summary, indent=2))

    failed = []
    if summary["hallucination_rate"] > HALLUCINATION_RATE_THRESHOLD:
        failed.append(f"hallucination_rate {summary['hallucination_rate']} > {HALLUCINATION_RATE_THRESHOLD}")
    if summary["latency_p95_s"] > SLA_LATENCY_SECONDS:
        failed.append(f"latency_p95_s {summary['latency_p95_s']} > {SLA_LATENCY_SECONDS}")

    if failed:
        print("\nEVAL FAILED:")
        for f in failed:
            print(f" - {f}")
        return 1

    print("\nEVAL PASSED.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "golden_dataset.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = parser.parse_args()
    sys.exit(run(args.dataset, args.out))
