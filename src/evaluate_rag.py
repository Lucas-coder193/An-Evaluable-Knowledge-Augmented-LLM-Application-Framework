"""Evaluate the offline RAG pipeline on answerable and no-answer questions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__:
    from .rag_pipeline import RAGPipeline, build_default_pipeline
    from .retrieve_bm25 import read_jsonl
else:
    from rag_pipeline import RAGPipeline, build_default_pipeline
    from retrieve_bm25 import read_jsonl


def evaluate(pipeline: RAGPipeline, questions: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    refusal_correct = 0
    answerable = 0
    no_answer = 0
    citation_supported = 0
    answerable_citation_supported = 0
    no_answer_refusals = 0
    answerable_top1_hit = 0
    latencies_ms: list[float] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_requests = 0
    for question in questions:
        started = time.perf_counter()
        result = pipeline.run(question["question"])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        latencies_ms.append(latency_ms)
        usage = result.get("generation_usage")
        if isinstance(usage, dict):
            usage_requests += 1
            for field in usage_totals:
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage_totals[field] += value
        expected = set(question.get("expected_evidence", []))
        actual_ids = {row.get("evidence_id") for row in result["supporting_evidence"]}
        # New evaluation rows label refusal explicitly. Keep the legacy
        # evidence-empty fallback for the existing development set.
        expected_refusal = question.get("should_refuse", not expected)
        if not isinstance(expected_refusal, bool):
            raise ValueError(f"question {question.get('id', '<unknown>')} should_refuse must be a boolean")
        if result["should_refuse"] == expected_refusal:
            refusal_correct += 1
        if expected_refusal:
            no_answer += 1
            if result["should_refuse"]:
                no_answer_refusals += 1
        else:
            answerable += 1
            if result["evidence"] and result["evidence"][0].get("evidence_id") in expected:
                answerable_top1_hit += 1
            if result["citation_support"]:
                answerable_citation_supported += 1
        if result["citation_support"]:
            citation_supported += 1
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_evidence": sorted(expected),
                "predicted_refusal": result["should_refuse"],
                "predicted_citations": result["citations"],
                "citation_support": result["citation_support"],
                "top_evidence_id": result["evidence"][0].get("evidence_id") if result["evidence"] else None,
                "evidence_gate": result["evidence_gate"],
                "answer": result["answer"],
                "supporting_evidence_ids": sorted(actual_ids),
                "latency_ms": latency_ms,
                "generation_usage": usage,
            }
        )
    total = len(questions)
    ordered_latencies = sorted(latencies_ms)

    def percentile(percent: float) -> float | None:
        if not ordered_latencies:
            return None
        if len(ordered_latencies) == 1:
            return ordered_latencies[0]
        position = (len(ordered_latencies) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(ordered_latencies) - 1)
        fraction = position - lower
        return round(ordered_latencies[lower] + (ordered_latencies[upper] - ordered_latencies[lower]) * fraction, 2)

    return {
        "questions_total": total,
        "answerable_questions": answerable,
        "no_answer_questions": no_answer,
        "refusal_accuracy": refusal_correct / total if total else 0.0,
        "citation_support_rate": citation_supported / total if total else 0.0,
        "answerable_citation_support_rate": answerable_citation_supported / answerable if answerable else 0.0,
        "no_answer_refusal_rate": no_answer_refusals / no_answer if no_answer else 0.0,
        "answerable_top1_retrieval_hit_rate": answerable_top1_hit / answerable if answerable else 0.0,
        "evaluation_scope": "Small project-authored development set over a 19-chunk pilot corpus; pipeline validation only, not an external benchmark or generalization claim.",
        "retriever": pipeline.retriever.ranking_config,
        "generator": pipeline.generator.name,
        "latency_ms": {
            "count": len(ordered_latencies),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": max(ordered_latencies) if ordered_latencies else None,
        },
        "token_usage": {
            "requests_with_usage": usage_requests,
            **usage_totals,
        },
        "rows": rows,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generator", choices=("extractive", "openai-responses"), default="extractive")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="lightonai/mLateOn")
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--question-id",
        nargs="+",
        dest="question_ids",
        help="只评测指定问题 ID；适合先做少量远程 API 冒烟测试",
    )
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < 1 or not 0.0 <= args.dense_weight <= 1.0:
        parser.error("top-k and candidate-k must be at least 1; dense-weight must be between 0 and 1")
    pipeline = build_default_pipeline(
        args.index,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        embedding_cache=args.embedding_cache,
        model_cache=args.model_cache,
        candidate_k=args.candidate_k,
        dense_weight=args.dense_weight,
        top_k=args.top_k,
        generator_name=args.generator,
    )
    questions = read_jsonl(args.questions)
    if args.question_ids:
        requested = set(args.question_ids)
        known = {str(question.get("id")) for question in questions}
        missing = sorted(requested - known)
        if missing:
            parser.error(f"unknown question id(s): {', '.join(missing)}")
        questions = [question for question in questions if question.get("id") in requested]
    report = evaluate(pipeline, questions)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
