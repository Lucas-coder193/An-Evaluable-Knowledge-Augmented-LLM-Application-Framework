"""Evaluate the retrieval-tool agent on the small development set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Protocol

if __package__:
    from .llm_provider import LLMProviderError, OpenAIResponsesGenerator
    from .rag_tool_demo import run_tool_rag
    from .rerank import load_reranked_index
    from .retrieve_bm25 import read_jsonl
else:
    from llm_provider import LLMProviderError, OpenAIResponsesGenerator
    from rag_tool_demo import run_tool_rag
    from rerank import load_reranked_index
    from retrieve_bm25 import read_jsonl


class ToolAgent(Protocol):
    name: str

    def generate_with_tools(
        self, question: str, registry: Any, *, max_tool_rounds: int = 2
    ) -> dict[str, Any]: ...


def _percentile(values: list[float], percent: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def _retrieval_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    calls = result.get("tool_calls", [])
    if not isinstance(calls, list):
        return []
    return [
        call
        for call in calls
        if isinstance(call, dict) and call.get("name") == "search_fastapi_docs"
    ]


def _retrieved_evidence_ids(calls: list[dict[str, Any]]) -> set[str]:
    evidence_ids: set[str] = set()
    for call in calls:
        tool_result = call.get("result")
        rows = tool_result.get("results", []) if isinstance(tool_result, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("evidence_id"), str):
                evidence_ids.add(row["evidence_id"])
    return evidence_ids


def evaluate(
    agent: ToolAgent,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    max_tool_rounds: int = 2,
) -> dict[str, Any]:
    """Run the agent and summarize tool choice, evidence, refusal and cost."""
    if top_k < 1 or max_tool_rounds < 1:
        raise ValueError("top_k and max_tool_rounds must be positive")
    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_questions = 0
    tool_call_count = 0
    retrieval_called_count = 0
    failed_requests = 0
    workflow_incomplete = 0
    tool_eval_count = 0
    tool_sequence_match_count = 0
    citation_eval_count = 0

    for question in questions:
        expected = {str(value) for value in question.get("expected_evidence", [])}
        # New evaluation rows carry this label explicitly because tool-only
        # questions can be answerable without any document evidence. Keep the
        # old development-set convention as a backward-compatible fallback.
        expected_refusal = question.get("should_refuse", not expected)
        if not isinstance(expected_refusal, bool):
            raise ValueError(f"question {question.get('id', '<unknown>')} should_refuse must be a boolean")
        raw_task_type = question.get("task_type")
        task_type = str(raw_task_type) if raw_task_type is not None else "rag"
        # Rows from the original pilot tests have no task_type metadata.  Keep
        # their historical run behavior instead of passing new workflow-only
        # kwargs to simple fake agents used by those tests.
        run_task_type = task_type if raw_task_type is not None else None
        started = time.perf_counter()
        error: str | None = None
        try:
            result = run_tool_rag(
                str(question["question"]), agent, retriever,
                top_k=top_k,
                max_tool_rounds=max_tool_rounds,
                task_type=run_task_type,
            )
        except LLMProviderError as caught:
            error = str(caught)[:300]
            result = {
                "answer": "评测请求失败。",
                "should_refuse": None,
                "tool_calls": [],
                "citation_check": {"supported": False},
                "workflow_status": "request_failed",
                "error": error,
            }
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        latencies_ms.append(latency_ms)
        status = result.get("workflow_status", "completed")
        if status == "request_failed":
            failed_requests += 1
            error = str(result.get("error", "model request failed"))[:300]
        elif status == "workflow_incomplete":
            workflow_incomplete += 1
            error = str(result.get("error", "workflow did not complete"))[:300]
        evaluated = status in {"completed", "gated_refusal"}
        calls = _retrieval_calls(result)
        retrieved_ids = _retrieved_evidence_ids(calls)
        predicted_refusal = bool(result.get("should_refuse")) if evaluated else None
        citation_applies = task_type in {"rag", "combined", "insufficient_evidence"}
        citation_supported = bool(result.get("citation_check", {}).get("supported")) if citation_applies else None
        if evaluated and citation_applies:
            citation_eval_count += 1
        if evaluated and calls:
            retrieval_called_count += 1
        tool_call_count += len(calls) if evaluated else 0
        expected_tools = question.get("expected_tools", [])
        actual_calls = result.get("tool_calls", [])
        tool_scored = task_type in {"tool", "combined", "multi_step", "tool_failure"}
        tool_sequence_match: bool | None = None
        if tool_scored and evaluated:
            expected_sequence = list(expected_tools) if isinstance(expected_tools, list) else []
            if task_type in {"combined"} and not expected_refusal:
                expected_sequence = [{"name": "search_fastapi_docs"}] + expected_sequence
            actual_sequence = [
                {"name": call.get("name"), "arguments": call.get("arguments", {})}
                for call in actual_calls
                if isinstance(call, dict)
            ] if isinstance(actual_calls, list) else []
            cursor = 0
            matched = True
            for expected_call in expected_sequence:
                found = False
                while cursor < len(actual_sequence):
                    candidate = actual_sequence[cursor]
                    cursor += 1
                    if candidate.get("name") != expected_call.get("name"):
                        continue
                    expected_args = expected_call.get("arguments")
                    if expected_args is not None and candidate.get("arguments") != expected_args:
                        continue
                    found = True
                    break
                if not found:
                    matched = False
                    break
            tool_sequence_match = matched
            tool_eval_count += 1
            tool_sequence_match_count += int(matched)
        usage = result.get("usage")
        if isinstance(usage, dict):
            # The provider aggregates token counts over all model rounds for
            # one question, so this counts questions with usage, not requests.
            usage_questions += 1
            for field in usage_totals:
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage_totals[field] += value
        rows.append(
            {
                "id": question["id"],
                "question": question["question"],
                "task_type": task_type,
                "status": status,
                "expected_evidence": sorted(expected),
                "expected_tools": expected_tools,
                "expected_refusal": expected_refusal,
                "predicted_refusal": predicted_refusal,
                "retrieval_called": bool(calls),
                "retrieval_call_count": len(calls),
                "retrieved_evidence_ids": sorted(retrieved_ids),
                "retrieval_hit": bool(expected & retrieved_ids),
                "citation_check": result.get("citation_check"),
                "citation_applicable": citation_applies,
                "tool_sequence_match": tool_sequence_match,
                "code_block_check": result.get("code_block_check"),
                "answer": result.get("answer"),
                "raw_answer": result.get("raw_answer"),
                "tool_calls": result.get("tool_calls", []),
                "generation_usage": usage,
                "latency_ms": latency_ms,
                "error": error or result.get("error"),
                "evidence_gate": result.get("evidence_gate"),
                "preflight_evidence_ids": result.get("preflight_evidence_ids", []),
            }
        )

    total = len(questions)
    evaluated_rows = [row for row in rows if row["status"] in {"completed", "gated_refusal"}]
    answerable_rows = [row for row in evaluated_rows if not row["expected_refusal"]]
    no_answer_rows = [row for row in evaluated_rows if row["expected_refusal"]]
    retrieval_quality_rows = [
        row for row in evaluated_rows
        if row.get("expected_evidence") and row.get("question")
    ]
    citation_rows = [row for row in evaluated_rows if row.get("citation_applicable")]
    answerable_citation_rows = [row for row in answerable_rows if row.get("citation_applicable")]
    refusal_correct = sum(row["predicted_refusal"] == row["expected_refusal"] for row in evaluated_rows)
    no_answer_refusals = sum(row["predicted_refusal"] is True for row in no_answer_rows)
    answerable_retrieval_rows = [row for row in answerable_rows if row.get("expected_evidence")]
    answerable_retrieval_hits = sum(row["retrieval_hit"] for row in answerable_retrieval_rows)
    answerable_citation_supported = sum(bool(row.get("citation_check", {}).get("supported")) for row in answerable_citation_rows)
    citation_supported_count = sum(bool(row.get("citation_check", {}).get("supported")) for row in citation_rows)
    for row in rows:
        row["retrieval_hit"] = bool(set(row["expected_evidence"]) & set(row["retrieved_evidence_ids"]))
    return {
        "questions_total": total,
        "evaluated_questions": len(evaluated_rows),
        "answerable_questions": len(answerable_rows),
        "no_answer_questions": len(no_answer_rows),
        "failed_requests": failed_requests,
        "workflow_incomplete": workflow_incomplete,
        "refusal_accuracy": refusal_correct / len(evaluated_rows) if evaluated_rows else 0.0,
        # Keep this headline metric backward-compatible: it measures how
        # often the agent called retrieval across all evaluated questions.
        # ``retrieval_required_questions`` and the answerable hit rate below
        # provide the stricter knowledge-question denominator.
        "retrieval_tool_call_rate": retrieval_called_count / len(evaluated_rows) if evaluated_rows else 0.0,
        "retrieval_required_questions": len(retrieval_quality_rows),
        "answerable_retrieval_hit_rate": answerable_retrieval_hits / len(answerable_retrieval_rows) if answerable_retrieval_rows else 0.0,
        "citation_support_rate": citation_supported_count / len(citation_rows) if citation_rows else 0.0,
        "citation_evaluated_questions": len(citation_rows),
        "answerable_citation_support_rate": answerable_citation_supported / len(answerable_citation_rows) if answerable_citation_rows else 0.0,
        "tool_sequence_match_rate": tool_sequence_match_count / tool_eval_count if tool_eval_count else 0.0,
        "tool_sequence_evaluated_questions": tool_eval_count,
        "no_answer_refusal_rate": no_answer_refusals / len(no_answer_rows) if no_answer_rows else 0.0,
        "tool_calls_total": tool_call_count,
        "evaluation_scope": "Project-authored frozen evaluation set over the configured knowledge index; agent validation only, not an external benchmark.",
        "metric_denominator": "Quality metrics exclude request_failed and workflow_incomplete rows; evaluated_questions includes completed model answers and application evidence-gate refusals.",
        "agent": agent.name,
        "retriever": getattr(retriever, "ranking_config", {}),
        "latency_ms": {
            "count": len(latencies_ms),
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "token_usage": {"questions_with_usage": usage_questions, **usage_totals},
        "rows": rows,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/processed/fastapi_v1.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="lightonai/mLateOn")
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    parser.add_argument("--max-tool-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--question-id", nargs="+", dest="question_ids")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 5 or args.candidate_k < 1 or args.max_tool_rounds < 1:
        parser.error("top-k must be 1-5; candidate-k and max-tool-rounds must be positive")
    if not 0.0 <= args.dense_weight <= 1.0:
        parser.error("dense-weight must be between 0 and 1")
    retriever = load_reranked_index(
        args.index,
        args.embedding_model,
        args.embedding_cache,
        args.model_cache,
        args.reranker_model,
        args.candidate_k,
        args.dense_weight,
    )
    questions = read_jsonl(args.questions)
    if args.question_ids:
        requested = set(args.question_ids)
        known = {str(question.get("id")) for question in questions}
        missing = sorted(requested - known)
        if missing:
            parser.error(f"unknown question id(s): {', '.join(missing)}")
        questions = [question for question in questions if question.get("id") in requested]
    agent = OpenAIResponsesGenerator.from_env()
    report = evaluate(
        agent,
        retriever,
        questions,
        top_k=args.top_k,
        max_tool_rounds=args.max_tool_rounds,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
