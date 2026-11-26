"""Ask the model to retrieve FastAPI documentation through a local tool."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .llm_provider import OpenAIResponsesGenerator
    from .rerank import load_reranked_index
else:
    from llm_provider import OpenAIResponsesGenerator
    from rerank import load_reranked_index


_CITATION_PATTERN = re.compile(r"【([^】]+)】")
_CODE_BLOCK_PATTERN = re.compile(
    r"(?P<prefix>(?:(?:(?:最小|完整|一个)?\s*(?:代码)?示例)|例如)\s*[:：]?\s*)?"
    r"```(?:[a-zA-Z0-9_+.-]+)?[ \t]*\r?\n(?P<code>[\s\S]*?)```"
)


def _normalise_code(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("`", "")).strip()


def _remove_unsupported_code_blocks(answer: str, evidence_texts: list[str]) -> tuple[str, int, int]:
    """Remove multiline code that is not present verbatim in retrieved evidence."""
    checked = 0
    unsupported = 0
    normalised_evidence = [_normalise_code(text) for text in evidence_texts]

    def replace(match: re.Match[str]) -> str:
        nonlocal checked, unsupported
        checked += 1
        code = _normalise_code(match.group("code"))
        if code and any(code in evidence for evidence in normalised_evidence):
            return match.group(0)
        unsupported += 1
        return ""

    cleaned = _CODE_BLOCK_PATTERN.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, checked, unsupported


def check_tool_citations(result: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when the final answer omits or invents retrieval citations."""
    allowed_by_label: dict[str, str] = {}
    allowed_ids: set[str] = set()
    evidence_texts: list[str] = []
    for call in result.get("tool_calls", []):
        if not isinstance(call, dict) or call.get("name") != "search_fastapi_docs":
            continue
        tool_result = call.get("result")
        rows = tool_result.get("results", []) if isinstance(tool_result, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                evidence_id = row.get("evidence_id")
                label = row.get("label")
                if not isinstance(evidence_id, str) or not evidence_id.strip():
                    continue
                allowed_ids.add(evidence_id)
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    evidence_texts.append(text)
                if isinstance(label, str) and label.strip():
                    allowed_by_label[label] = evidence_id

    answer = result.get("answer", "")
    if isinstance(answer, str):
        cleaned_answer, code_blocks_checked, unsupported_code_blocks = _remove_unsupported_code_blocks(
            answer, evidence_texts
        )
        grounding_supported = unsupported_code_blocks == 0
        result["code_block_check"] = {
            "code_blocks_checked": code_blocks_checked,
            "unsupported_code_blocks": unsupported_code_blocks,
            "answer_modified": not grounding_supported,
        }
        if not grounding_supported:
            result["raw_answer"] = answer
            result["answer"] = cleaned_answer or "证据不足，无法提供完整代码示例。"
            if not cleaned_answer:
                result["should_refuse"] = True
        answer = result["answer"]
    cited_tokens = _CITATION_PATTERN.findall(answer) if isinstance(answer, str) else []
    mapped_ids = [allowed_by_label.get(token, token if token in allowed_ids else None) for token in cited_tokens]
    unsupported = [token for token, evidence_id in zip(cited_tokens, mapped_ids) if evidence_id is None]
    cited_ids = [evidence_id for evidence_id in mapped_ids if evidence_id is not None]
    supported = bool(allowed_ids) and bool(cited_tokens) and not unsupported
    report = {
        "supported": supported,
        "cited_labels": cited_tokens,
        "cited_evidence_ids": cited_ids,
        "allowed_evidence_ids": sorted(allowed_ids),
        "unsupported_citations": unsupported,
    }
    result["citation_check"] = report
    if not supported:
        result["raw_answer"] = answer
        result["answer"] = "证据不足，无法回答。"
        result["should_refuse"] = True
    return result


def run_tool_rag(
    question: str,
    agent: Any,
    retriever: Any,
    *,
    top_k: int = 3,
    max_tool_rounds: int = 2,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Run the LangGraph-orchestrated tool agent workflow.

    The public function remains stable so the CLI, evaluator, and HTTP service
    all share the same workflow implementation.
    """
    if __package__:
        from .workflow_graph import run_tool_rag_graph
    else:
        from workflow_graph import run_tool_rag_graph
    return run_tool_rag_graph(
        question,
        agent,
        retriever,
        top_k=top_k,
        max_tool_rounds=max_tool_rounds,
        task_type=task_type,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/processed/fastapi_v1.jsonl"))
    parser.add_argument("--query", required=True, help="Question about the indexed FastAPI docs")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum chunks returned by the search tool")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="lightonai/mLateOn")
    parser.add_argument(
        "--embedding-cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz")
    )
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    parser.add_argument("--max-tool-rounds", type=int, default=2)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 5 or args.candidate_k < 1:
        parser.error("top-k must be between 1 and 5; candidate-k must be at least 1")
    if not 0.0 <= args.dense_weight <= 1.0:
        parser.error("dense-weight must be between 0 and 1")
    if args.max_tool_rounds < 1:
        parser.error("max-tool-rounds must be at least 1")

    retriever = load_reranked_index(
        args.index,
        args.embedding_model,
        args.embedding_cache,
        args.model_cache,
        args.reranker_model,
        args.candidate_k,
        args.dense_weight,
    )
    agent = OpenAIResponsesGenerator.from_env()
    result = run_tool_rag(
        args.query,
        agent,
        retriever,
        top_k=args.top_k,
        max_tool_rounds=args.max_tool_rounds,
    )
    result["retriever"] = retriever.ranking_config
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
