"""A small, API-agnostic RAG pipeline with an extractive offline baseline.

The pipeline deliberately keeps retrieval, evidence gating, generation, and
citation validation separate.  This makes it possible to replace the
extractive generator with an LLM provider later without changing retrieval or
evaluation code.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__:
    from .rag_prompts import build_prompt
    from .rerank import load_reranked_index
    from .retrieve_bm25 import read_jsonl, tokenize
else:
    from rag_prompts import build_prompt
    from rerank import load_reranked_index
    from retrieve_bm25 import read_jsonl, tokenize


# These words mostly describe the form of a question or the shared corpus
# domain.  Removing them prevents a document from looking supported merely
# because it contains the word "FastAPI" or a generic question word.
STOP_TOKENS = {
    "a", "an", "and", "are", "be", "by", "can", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "which",
    "with", "you", "fastapi", "官方", "如何", "什么", "哪些", "哪个", "哪家", "是否",
    "怎么", "怎样", "哪里", "在哪", "为什么", "可以", "能够", "能否", "会", "吗", "呢",
    "的", "了", "是", "在", "和", "与", "对", "把", "从", "中", "后", "还", "都",
    "一个", "通常", "应该", "需要", "如何", "多少", "情况", "做法",
}


class Retriever(Protocol):
    ranking_config: dict[str, Any]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class Generator(Protocol):
    name: str

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]: ...


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOP_TOKENS}


def lexical_coverage(question: str, text: str) -> float:
    """Return the fraction of question content tokens found in one chunk."""
    query_tokens = _content_tokens(question)
    if not query_tokens:
        return 0.0
    document_tokens = _content_tokens(text)
    return len(query_tokens & document_tokens) / len(query_tokens)


@dataclass(frozen=True)
class EvidenceGate:
    """A conservative evidence sufficiency check for the offline baseline."""

    min_lexical_coverage: float = 0.20

    def assess(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not evidence:
            return {
                "should_refuse": True,
                "reason": "no_retrieval_results",
                "lexical_coverage": 0.0,
                "top_rerank_score": None,
            }
        # A relevant chunk can appear below the first reranked result when a
        # question combines multiple concepts.  Gate against the best lexical
        # coverage in the returned evidence rather than trusting one rank.
        scored = [
            (lexical_coverage(question, str(row.get("text", ""))), index, row)
            for index, row in enumerate(evidence)
        ]
        coverage, _, top = max(scored, key=lambda item: (item[0], -item[1]))
        if coverage < self.min_lexical_coverage:
            return {
                "should_refuse": True,
                "reason": "insufficient_question_term_support",
                "lexical_coverage": round(coverage, 4),
                "top_rerank_score": top.get("rerank_score", top.get("score")),
            }
        return {
            "should_refuse": False,
            "reason": "evidence_passed_lexical_gate",
            "lexical_coverage": round(coverage, 4),
            "top_rerank_score": top.get("rerank_score", top.get("score")),
        }


class ExtractiveGenerator:
    """Offline answer generator that returns source text verbatim.

    It is intentionally not presented as an LLM.  Its purpose is to validate
    the end-to-end evidence, citation, and refusal plumbing before an API
    provider is introduced.
    """

    name = "extractive-evidence-v1"

    def generate(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not evidence:
            return {"answer": "证据不足，无法回答。", "citations": [], "should_refuse": True}
        top = evidence[0]
        text = str(top.get("text", "")).strip()
        # Chunks are deliberately short (one or two complete sentences).  We
        # return the full top chunk so a semicolon-separated condition and its
        # conclusion cannot be accidentally cut apart by sentence selection.
        answer = text
        return {"answer": answer, "citations": [top["label"]], "should_refuse": False}


def _normalise_citations(raw: Any, evidence: list[dict[str, Any]]) -> list[str]:
    allowed = {str(row["label"]) for row in evidence}
    values = raw if isinstance(raw, list) else [raw]
    citations: list[str] = []
    for value in values:
        label = str(value).strip()
        if label in allowed and label not in citations:
            citations.append(label)
    return citations


class RAGPipeline:
    """Retrieve evidence, gate it, generate an answer, and validate citations."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator | None = None,
        top_k: int = 5,
        gate: EvidenceGate | None = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k
        self.gate = gate or EvidenceGate()

    def run(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        raw_results = self.retriever.search(question, top_k=self.top_k)
        evidence: list[dict[str, Any]] = []
        for index, row in enumerate(raw_results, start=1):
            evidence.append(
                {
                    "label": f"E{index}",
                    "evidence_id": row.get("evidence_id", row.get("chunk_id")),
                    "source_title": row.get("source_title"),
                    "source_url": row.get("source_url"),
                    "text": row.get("text", ""),
                    "score": row.get("score"),
                    "rerank_score": row.get("rerank_score"),
                    "candidate_rank": row.get("candidate_rank"),
                }
            )
        gate = self.gate.assess(question, evidence)
        prompt = build_prompt(question, evidence)
        generation_usage: dict[str, Any] | None = None
        if gate["should_refuse"]:
            answer = "证据不足，无法回答。"
            citations: list[str] = []
            confidence = 0.0
        else:
            generated = self.generator.generate(question, evidence)
            raw_usage = generated.get("usage")
            if isinstance(raw_usage, dict):
                generation_usage = raw_usage
            model_refusal = generated.get("should_refuse", False)
            if model_refusal is True:
                answer = "证据不足，无法回答。"
                citations = []
                gate = {**gate, "should_refuse": True, "reason": "generator_requested_refusal"}
                confidence = 0.0
            elif model_refusal is not False:
                answer = "证据不足，无法回答。"
                citations = []
                gate = {**gate, "should_refuse": True, "reason": "generator_returned_invalid_refusal_flag"}
                confidence = 0.0
            else:
                raw_answer = generated.get("answer")
                raw_citations = generated.get("citations", [])
                answer = raw_answer.strip() if isinstance(raw_answer, str) else ""
                citations = _normalise_citations(raw_citations, evidence) if isinstance(raw_citations, list) else []
                allowed_labels = {str(row["label"]) for row in evidence}
                invalid_citation = (
                    not isinstance(raw_citations, list)
                    or any(not isinstance(label, str) or label not in allowed_labels for label in raw_citations)
                )
                if not answer or not citations or invalid_citation:
                    # A non-empty answer without a valid citation is not safe
                    # to present as a grounded answer in this baseline.
                    answer = "证据不足，无法回答。"
                    citations = []
                    gate = {
                        **gate,
                        "should_refuse": True,
                        "reason": "generator_returned_invalid_answer_or_citation",
                    }
                    confidence = 0.0
                else:
                    confidence = round(min(1.0, 0.2 + 0.8 * float(gate["lexical_coverage"])), 4)
        citation_rows = [row for row in evidence if row["label"] in citations]
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "citation_support": bool(citations) and all(label in {row["label"] for row in evidence} for label in citations),
            "should_refuse": bool(gate["should_refuse"]),
            "confidence": confidence,
            "evidence": evidence,
            "supporting_evidence": citation_rows,
            "evidence_gate": gate,
            "prompt": prompt,
            "retriever": self.retriever.ranking_config,
            "generator": self.generator.name,
            "generation_usage": generation_usage,
        }


def build_default_pipeline(
    index_path: Path,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    reranker_model: str = "lightonai/mLateOn",
    embedding_cache: Path | None = Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"),
    model_cache: Path | None = Path("data/models/fastembed"),
    candidate_k: int = 20,
    dense_weight: float = 0.5,
    top_k: int = 5,
    generator_name: str = "extractive",
) -> RAGPipeline:
    if generator_name == "extractive":
        generator: Generator = ExtractiveGenerator()
    elif generator_name == "openai-responses":
        if __package__:
            from .llm_provider import OpenAIResponsesGenerator
        else:
            from llm_provider import OpenAIResponsesGenerator
        generator = OpenAIResponsesGenerator.from_env()
    else:
        raise ValueError("generator_name must be extractive or openai-responses")
    retriever = load_reranked_index(
        index_path,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
        model_cache=model_cache,
        reranker_model=reranker_model,
        candidate_k=candidate_k,
        dense_weight=dense_weight,
    )
    return RAGPipeline(retriever, generator=generator, top_k=top_k)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generator", choices=("extractive", "openai-responses"), default="extractive")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="lightonai/mLateOn")
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    parser.add_argument("--output", type=Path)
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
    rendered = json.dumps(pipeline.run(args.query), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
