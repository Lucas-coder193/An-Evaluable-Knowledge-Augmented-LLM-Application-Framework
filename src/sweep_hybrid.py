"""Run an exploratory RRF and weighted-fusion parameter grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .evaluate_retrieval import evaluate
    from .retrieve_bm25 import BM25, read_jsonl
    from .retrieve_dense import DenseIndex
    from .retrieve_hybrid import HybridRRF, HybridWeighted
else:
    from evaluate_retrieval import evaluate
    from retrieve_bm25 import BM25, read_jsonl
    from retrieve_dense import DenseIndex
    from retrieve_hybrid import HybridRRF, HybridWeighted


def run_grid(
    sparse: BM25,
    dense: DenseIndex,
    questions: list[dict[str, Any]],
    top_k: int = 5,
    candidate_k_values: tuple[int, ...] = (3, 5, 10, 20),
    rrf_k_values: tuple[int, ...] = (1, 5, 10, 20, 60),
    dense_weights: tuple[float, ...] = tuple(index / 10 for index in range(11)),
) -> dict[str, Any]:
    if top_k < 1 or any(value < 1 for value in candidate_k_values + rrf_k_values):
        raise ValueError("top_k, candidate depths, and rrf_k values must be at least 1")
    if any(not 0.0 <= value <= 1.0 for value in dense_weights):
        raise ValueError("dense weights must be between 0 and 1")

    rrf_results = []
    for candidate_k in candidate_k_values:
        for rrf_k in rrf_k_values:
            result = evaluate(HybridRRF(sparse, dense, candidate_k=candidate_k, rrf_k=rrf_k), questions, top_k)
            rrf_results.append({
                "candidate_k": candidate_k,
                "rrf_k": rrf_k,
                "metrics": {key: value for key, value in result.items() if key.startswith(("recall@", "ndcg@", "mrr@"))},
                "top1_misses": [row["id"] for row in result["top1_misses"]],
            })

    weighted_results = []
    candidate_k = max(candidate_k_values)
    for dense_weight in dense_weights:
        result = evaluate(
            HybridWeighted(sparse, dense, candidate_k=candidate_k, dense_weight=dense_weight),
            questions,
            top_k,
        )
        weighted_results.append({
            "candidate_k": candidate_k,
            "dense_weight": dense_weight,
            "metrics": {key: value for key, value in result.items() if key.startswith(("recall@", "ndcg@", "mrr@"))},
            "top1_misses": [row["id"] for row in result["top1_misses"]],
        })

    return {
        "top_k": top_k,
        "question_count": len(questions),
        "rrf_grid": rrf_results,
        "weighted_grid": weighted_results,
        "warning": "Exploratory only: these parameter values were examined on this development set. Do not report the best row as an independent test result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    documents = read_jsonl(args.index)
    sparse = BM25(documents)
    dense = DenseIndex.from_jsonl(
        args.index,
        model_name=args.model,
        cache_path=args.embedding_cache,
        cache_dir=args.model_cache,
    )
    report = run_grid(sparse, dense, read_jsonl(args.questions), top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
