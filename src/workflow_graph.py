"""LangGraph orchestration for the existing tool-enabled RAG workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if __package__:
    from .llm_provider import LLMProviderError
    from .rag_tool_demo import check_tool_citations
    from .tools import rag_tool_registry
    from .rag_pipeline import EvidenceGate
else:
    from llm_provider import LLMProviderError
    from rag_tool_demo import check_tool_citations
    from tools import rag_tool_registry
    from rag_pipeline import EvidenceGate


class ToolRAGState(TypedDict, total=False):
    question: str
    top_k: int
    max_tool_rounds: int
    task_type: str | None
    evidence_gate: dict[str, Any]
    preflight_evidence: list[dict[str, Any]]
    preflight_evidence_ids: list[str]
    result: dict[str, Any]
    attempt: int
    attempt_reports: list[dict[str, Any]]
    workflow_trace: list[str]
    next_step: str


def build_tool_rag_graph(
    agent: Any,
    retriever: Any,
    *,
    max_validation_retries: int = 1,
):
    """Build a bounded route → tool → validate → retry/finish graph.

    LangGraph owns the control flow; the existing generator, tool registry,
    evidence gate, and citation checker remain responsible for their existing
    domain behavior.
    """
    if (
        isinstance(max_validation_retries, bool)
        or not isinstance(max_validation_retries, int)
        or not 0 <= max_validation_retries <= 2
    ):
        raise ValueError("max_validation_retries must be an integer between 0 and 2")

    def route(state: ToolRAGState) -> dict[str, Any]:
        task_type = state.get("task_type")
        gate_applies = task_type is None or task_type in {"rag", "insufficient_evidence"}
        try:
            rows = retriever.search(state["question"], top_k=state["top_k"]) if gate_applies else []
        except Exception as error:
            # Retrieval is a dependency boundary. Do not misreport an outage
            # as insufficient evidence or send exception details to clients.
            return {
                "next_step": "finish",
                "workflow_trace": ["route"],
                "result": {
                    "answer": "检索服务失败，暂时无法回答。",
                    "should_refuse": True,
                    "tool_calls": [],
                    "workflow_status": "request_failed",
                    "error": f"preflight retrieval failed: {type(error).__name__}",
                },
            }
        evidence = [
            {
                "label": f"E{rank}",
                "evidence_id": row.get("evidence_id", row.get("chunk_id")),
                "text": row.get("text", ""),
            }
            for rank, row in enumerate(rows, start=1)
        ]
        gate = (
            EvidenceGate().assess(state["question"], evidence)
            if gate_applies
            else {
                "should_refuse": False,
                "reason": "not_applicable_to_task_type",
                "lexical_coverage": None,
            }
        )
        trace = ["route"]
        common = {
            "evidence_gate": gate,
            "preflight_evidence": evidence,
            "preflight_evidence_ids": [row["evidence_id"] for row in evidence],
            "workflow_trace": trace,
        }
        if gate["should_refuse"]:
            return {
                **common,
                "next_step": "finish",
                "result": {
                    "answer": "证据不足，无法回答。",
                    "should_refuse": True,
                    "tool_calls": [],
                    "usage": None,
                    "workflow_status": "gated_refusal",
                    "evidence_gate": gate,
                    "preflight_evidence_ids": common["preflight_evidence_ids"],
                    "citation_check": {"supported": False, "reason": "gated_before_generation"},
                    "workflow_engine": "langgraph",
                    "workflow_trace": trace + ["finish"],
                    "validation_retries": 0,
                },
            }
        return {**common, "next_step": "execute"}

    def execute(state: ToolRAGState) -> dict[str, Any]:
        task_type = state.get("task_type")
        attempt = state.get("attempt", 0) + 1
        question = state["question"]
        if attempt > 1:
            question += (
                "\n\n校验反馈：上一轮回答未通过引用校验。请重新调用知识检索工具，"
                "只引用本轮工具结果中实际提供的证据标签；若证据不足，请明确拒答。"
            )
        generation_kwargs: dict[str, Any] = {"max_tool_rounds": state["max_tool_rounds"]}
        if task_type in {"rag", "combined"}:
            generation_kwargs["required_first_tool"] = "search_fastapi_docs"
        try:
            result = agent.generate_with_tools(
                question,
                rag_tool_registry(retriever, max_top_k=state["top_k"]),
                **generation_kwargs,
            )
        except LLMProviderError as error:
            result = {
                "answer": "模型响应无效，无法可靠回答。",
                "should_refuse": True,
                "tool_calls": [],
                "workflow_status": "request_failed",
                "error": f"model response failed: {type(error).__name__}",
                "provider_retry_events": getattr(error, "retry_events", []),
            }
        result = dict(result)
        result["evidence_gate"] = state["evidence_gate"]
        result["preflight_evidence_ids"] = state["preflight_evidence_ids"]
        trace = list(state.get("workflow_trace", [])) + ["retrieve_tool"]
        result["workflow_engine"] = "langgraph"
        # Remember the generator's refusal BEFORE citation checking changes it.
        result["generation_refused"] = bool(result.get("should_refuse"))
        return {"result": result, "attempt": attempt, "workflow_trace": trace}

    def validate(state: ToolRAGState) -> dict[str, Any]:
        result = state["result"]
        task_type = state.get("task_type")
        tool_calls = result.get("tool_calls", [])
        if isinstance(tool_calls, list) and any(
            isinstance(call, dict) and call.get("error") for call in tool_calls
        ):
            result["should_refuse"] = True
        citation_required = task_type is None or task_type in {"rag", "combined", "insufficient_evidence"}
        if citation_required and result.get("workflow_status", "completed") == "completed":
            check_tool_citations(result)
        elif result.get("workflow_status", "completed") == "completed":
            result["citation_check"] = {"supported": None, "reason": "not_applicable_to_task_type"}
        trace = list(state.get("workflow_trace", [])) + ["validate"]
        result["workflow_trace"] = trace
        result["validation_retries"] = max(0, state.get("attempt", 1) - 1)
        report = {
            "attempt": state["attempt"],
            "workflow_status": result.get("workflow_status", "completed"),
            "should_refuse": bool(result.get("should_refuse")),
            "citation_supported": result.get("citation_check", {}).get("supported"),
            "usage": result.get("usage"),
            "provider_retry_events": result.get("provider_retry_events", []),
            "tool_error_count": sum(
                bool(call.get("error")) for call in tool_calls if isinstance(call, dict)
            ) if isinstance(tool_calls, list) else 0,
        }
        return {
            "result": result,
            "workflow_trace": trace,
            "attempt_reports": [*state.get("attempt_reports", []), report],
        }

    def after_validation(state: ToolRAGState) -> Literal["retry", "finish"]:
        result = state["result"]
        if result.get("workflow_status", "completed") != "completed":
            return "finish"
        if result.get("generation_refused") or any(
            isinstance(call, dict) and call.get("error") for call in result.get("tool_calls", [])
        ):
            return "finish"
        retrieval_calls = [
            call for call in result.get("tool_calls", [])
            if call.get("name") == "search_fastapi_docs"
        ]
        if retrieval_calls and not any(call.get("result", {}).get("results") for call in retrieval_calls):
            return "finish"
        citation_check = result.get("citation_check", {})
        if citation_check.get("supported") is not False:
            return "finish"
        if state.get("attempt", 1) <= max_validation_retries:
            return "retry"
        return "finish"

    def finish(state: ToolRAGState) -> dict[str, Any]:
        result = dict(state["result"])
        trace = list(state.get("workflow_trace", []))
        if not trace or trace[-1] != "finish":
            trace.append("finish")
        result["workflow_trace"] = trace
        result["validation_retries"] = max(0, state.get("attempt", 0) - 1)
        result["workflow_engine"] = "langgraph"
        result.pop("generation_refused", None)
        reports = state.get("attempt_reports", [])
        result["attempt_reports"] = reports
        usage: dict[str, int] = {}
        events: list[dict[str, Any]] = []
        for report in reports:
            for field, value in (report.get("usage") or {}).items():
                usage[field] = usage.get(field, 0) + value
            events.extend(
                {**event, "workflow_attempt": report["attempt"]}
                for event in report["provider_retry_events"]
            )
        if usage:
            result["usage"] = usage
        result["provider_retry_count"] = len(events)
        result["provider_retry_events"] = events
        result["tool_error_count"] = sum(report["tool_error_count"] for report in reports)
        return {"result": result, "workflow_trace": trace}

    graph = StateGraph(ToolRAGState)
    graph.add_node("route", route)
    graph.add_node("retrieve_tool", execute)
    graph.add_node("validate", validate)
    graph.add_node("finish", finish)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda state: state["next_step"], {"execute": "retrieve_tool", "finish": "finish"})
    graph.add_edge("retrieve_tool", "validate")
    graph.add_conditional_edges("validate", after_validation, {"retry": "retrieve_tool", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


def run_tool_rag_graph(
    question: str,
    agent: Any,
    retriever: Any,
    *,
    top_k: int = 3,
    max_tool_rounds: int = 2,
    task_type: str | None = None,
    max_validation_retries: int = 1,
) -> dict[str, Any]:
    """Invoke the compiled graph and return the established result shape."""
    graph = build_tool_rag_graph(agent, retriever, max_validation_retries=max_validation_retries)
    state = graph.invoke(
        {
            "question": question,
            "top_k": top_k,
            "max_tool_rounds": max_tool_rounds,
            "task_type": task_type,
            "attempt": 0,
            "workflow_trace": [],
        }
    )
    return state["result"]
