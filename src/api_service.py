"""Application service for exposing the retrieval agent through HTTP."""

from __future__ import annotations

import time
from typing import Any

if __package__:
    from .rag_tool_demo import run_tool_rag
else:
    from rag_tool_demo import run_tool_rag


class APIRequestError(ValueError):
    """Raised when an API request is outside the service contract."""


def _validate_question(question: Any) -> str:
    if not isinstance(question, str):
        raise APIRequestError("question must be a string")
    question = question.strip()
    if not question:
        raise APIRequestError("question must not be empty")
    if len(question) > 2_000:
        raise APIRequestError("question must be at most 2000 characters")
    return question


def _validate_positive_int(name: str, value: Any, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIRequestError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise APIRequestError(f"{name} must be between 1 and {maximum}")
    return value


def _retrieved_ids(result: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for call in result.get("tool_calls", []):
        if not isinstance(call, dict) or call.get("name") != "search_fastapi_docs":
            continue
        tool_result = call.get("result")
        rows = tool_result.get("results", []) if isinstance(tool_result, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("evidence_id"), str):
                if row["evidence_id"] not in values:
                    values.append(row["evidence_id"])
    return values


class RAGService:
    """Thin application layer that hides workflow internals from HTTP clients."""

    def __init__(self, agent: Any, retriever: Any, *, default_top_k: int = 3, default_max_tool_rounds: int = 2):
        self.agent = agent
        self.retriever = retriever
        self.default_top_k = _validate_positive_int("default_top_k", default_top_k, maximum=5)
        self.default_max_tool_rounds = _validate_positive_int(
            "default_max_tool_rounds", default_max_tool_rounds, maximum=5
        )

    @property
    def agent_name(self) -> str:
        return str(getattr(self.agent, "name", type(self.agent).__name__))

    def health(self) -> dict[str, Any]:
        config = getattr(self.retriever, "ranking_config", {})
        retriever_name = config.get("retriever") if isinstance(config, dict) else None
        return {
            "status": "ok",
            "agent": self.agent_name,
            "retriever": retriever_name or type(self.retriever).__name__,
        }

    def ask(
        self,
        question: Any,
        *,
        top_k: Any = None,
        max_tool_rounds: Any = None,
    ) -> dict[str, Any]:
        question = _validate_question(question)
        top_k = self.default_top_k if top_k is None else _validate_positive_int("top_k", top_k, maximum=5)
        max_tool_rounds = (
            self.default_max_tool_rounds
            if max_tool_rounds is None
            else _validate_positive_int("max_tool_rounds", max_tool_rounds, maximum=5)
        )
        started = time.perf_counter()
        result = run_tool_rag(
            question,
            self.agent,
            self.retriever,
            top_k=top_k,
            max_tool_rounds=max_tool_rounds,
        )
        tool_calls = result.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []
        response = {
            "answer": result.get("answer", "证据不足，无法回答。"),
            "should_refuse": bool(result.get("should_refuse")),
            "workflow_status": result.get("workflow_status", "completed"),
            "workflow_engine": result.get("workflow_engine", "langgraph"),
            "workflow_trace": result.get("workflow_trace", []),
            "validation_retries": result.get("validation_retries", 0),
            "provider_retry_count": result.get("provider_retry_count", 0),
            "provider_retry_events": result.get("provider_retry_events", []),
            "attempt_reports": result.get("attempt_reports", []),
            "tool_error_count": result.get("tool_error_count", sum(
                1
                for call in tool_calls
                if isinstance(call, dict) and call.get("error")
            )),
            "citation_check": result.get("citation_check"),
            "code_block_check": result.get("code_block_check"),
            "evidence_gate": result.get("evidence_gate"),
            "retrieved_evidence_ids": _retrieved_ids(result),
            "usage": result.get("usage"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if response["workflow_status"] in {"request_failed", "workflow_incomplete"}:
            response["error"] = str(result.get("error", "workflow did not complete"))[:300]
        return response


def build_default_service(
    *,
    index_path: str = "data/processed/fastapi_frozen_v2.jsonl",
    embedding_cache: str = "data/models/fastapi_frozen_v2_bge_small_zh_v1_5.npz",
    model_cache: str = "data/models/fastembed",
    candidate_k: int = 20,
    dense_weight: float = 0.5,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    reranker_model: str = "lightonai/mLateOn",
) -> RAGService:
    """Load the production-shaped local retriever and configured LLM lazily."""
    from pathlib import Path

    if __package__:
        from .llm_provider import OpenAIResponsesGenerator
        from .rerank import load_reranked_index
    else:
        from llm_provider import OpenAIResponsesGenerator
        from rerank import load_reranked_index

    retriever = load_reranked_index(
        Path(index_path),
        embedding_model,
        Path(embedding_cache),
        Path(model_cache),
        reranker_model,
        candidate_k,
        dense_weight,
    )
    return RAGService(OpenAIResponsesGenerator.from_env(), retriever)
