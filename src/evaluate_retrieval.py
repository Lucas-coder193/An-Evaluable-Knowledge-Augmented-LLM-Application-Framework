"""Evaluate retrieval against a small, manually labeled JSONL development set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol

if __package__:
    from .retrieve_bm25 import BM25, read_jsonl
else:
    from retrieve_bm25 import BM25, read_jsonl


class Retriever(Protocol):
    """Minimal interface shared by sparse and dense retrieval systems."""

    ranking_config: dict[str, Any]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


def evaluate(index: Retriever, questions: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    answerable = [row for row in questions if row.get("expected_evidence")]
    if not answerable:
        raise ValueError("evaluation set has no answerable questions")

    cutoffs = sorted({1, min(3, top_k), top_k})
    recalls = {cutoff: [] for cutoff in cutoffs}
    ndcgs = {cutoff: [] for cutoff in cutoffs}
    reciprocal_ranks: list[float] = []
    top1_misses: list[dict[str, Any]] = []
    for question in answerable:
        relevant = set(question["expected_evidence"])
        # Count each evidence section once so a long section cannot inflate its rank.
        ranked_evidence: list[str] = []
        for result in index.search(question["question"], top_k=max(top_k * 3, top_k)):
            evidence_id = result["evidence_id"]
            if evidence_id not in ranked_evidence:
                ranked_evidence.append(evidence_id)
            if len(ranked_evidence) >= max(cutoffs):
                break

        first_rank = next((rank for rank, evidence_id in enumerate(ranked_evidence, start=1) if evidence_id in relevant), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        if not ranked_evidence or ranked_evidence[0] not in relevant:
            top1_misses.append({
                "id": question["id"],
                "question": question["question"],
                "expected_evidence": sorted(relevant),
                "top_results": ranked_evidence[:top_k],
            })
        for cutoff in cutoffs:
            prefix = ranked_evidence[:cutoff]
            hits = {evidence_id for evidence_id in prefix if evidence_id in relevant}
            recalls[cutoff].append(len(hits) / len(relevant))
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, evidence_id in enumerate(prefix, start=1)
                if evidence_id in relevant
            )
            ideal_hits = min(len(relevant), cutoff)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcgs[cutoff].append(dcg / idcg if idcg else 0.0)

    no_answer = [row for row in questions if not row.get("expected_evidence")]
    report: dict[str, Any] = {
        "questions_total": len(questions),
        "answerable_questions": len(answerable),
        "no_answer_questions": len(no_answer),
        f"mrr@{top_k}": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ranking_config": index.ranking_config,
        "evaluation_scope": "Small project-authored development set over a 19-chunk pilot corpus; pipeline validation only, not an external benchmark or generalization claim.",
        "note": "Retrieval metrics include only questions with labeled evidence; no-answer cases are reserved for later refusal evaluation.",
        "top1_misses": top1_misses,
    }
    report.update({f"recall@{cutoff}": sum(values) / len(values) for cutoff, values in recalls.items()})
    report.update({f"ndcg@{cutoff}": sum(values) / len(values) for cutoff, values in ndcgs.items()})
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--section-boost", type=float, default=0.0)
    parser.add_argument("--source-title-boost", type=float, default=0.0)
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid", "hybrid_weighted", "rerank"), default="bm25")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="lightonai/mLateOn")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"),
        help="Cached dense index; ignored by BM25",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("data/models/fastembed"),
        help="Local directory for downloaded embedding model files",
    )
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate depth for hybrid retrieval")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant for hybrid retrieval")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Dense score weight for weighted hybrid retrieval")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.section_boost < 0 or args.source_title_boost < 0:
        parser.error("field boosts must be non-negative")

    if args.retriever == "bm25":
        index = BM25(
            read_jsonl(args.index),
            section_boost=args.section_boost,
            source_title_boost=args.source_title_boost,
        )
    elif args.retriever == "dense":
        if __package__:
            from .retrieve_dense import DenseIndex
        else:
            from retrieve_dense import DenseIndex
        index = DenseIndex.from_jsonl(
            args.index,
            model_name=args.embedding_model,
            cache_path=args.embedding_cache,
            cache_dir=args.model_cache,
        )
    elif args.retriever in ("hybrid", "hybrid_weighted"):
        if __package__:
            from .retrieve_hybrid import load_hybrid_index
        else:
            from retrieve_hybrid import load_hybrid_index
        if args.candidate_k < 1 or args.rrf_k < 1 or not 0.0 <= args.dense_weight <= 1.0:
            parser.error("--candidate-k and --rrf-k must be at least 1; --dense-weight must be between 0 and 1")
        index = load_hybrid_index(
            args.index,
            model_name=args.embedding_model,
            cache_path=args.embedding_cache,
            cache_dir=args.model_cache,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            fusion="weighted" if args.retriever == "hybrid_weighted" else "rrf",
            dense_weight=args.dense_weight,
        )
    else:
        if __package__:
            from .rerank import load_reranked_index
        else:
            from rerank import load_reranked_index
        if args.candidate_k < 1 or not 0.0 <= args.dense_weight <= 1.0:
            parser.error("--candidate-k must be at least 1; --dense-weight must be between 0 and 1")
        index = load_reranked_index(
            args.index,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            model_cache=args.model_cache,
            reranker_model=args.reranker_model,
            candidate_k=args.candidate_k,
            dense_weight=args.dense_weight,
        )
    report = evaluate(index, read_jsonl(args.questions), args.top_k)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
