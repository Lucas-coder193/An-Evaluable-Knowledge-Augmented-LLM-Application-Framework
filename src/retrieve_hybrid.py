"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

if __package__:
    from .retrieve_bm25 import BM25, read_jsonl
    from .retrieve_dense import DenseIndex
else:
    from retrieve_bm25 import BM25, read_jsonl
    from retrieve_dense import DenseIndex


class Retriever(Protocol):
    ranking_config: dict[str, Any]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


def _result_key(result: dict[str, Any]) -> str:
    key = result.get("chunk_id") or result.get("evidence_id")
    if not key:
        raise ValueError("every retrieval result needs chunk_id or evidence_id")
    return str(key)


class HybridRRF:
    """Fuse sparse and dense rankings without comparing incompatible scores."""

    def __init__(self, sparse: Retriever, dense: Retriever, candidate_k: int = 20, rrf_k: int = 60):
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        self.sparse = sparse
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    @property
    def ranking_config(self) -> dict[str, Any]:
        return {
            "retriever": "hybrid_rrf",
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "fusion_formula": "sum(1 / (rrf_k + rank))",
            "components": {
                "sparse": self.sparse.ranking_config,
                "dense": self.dense.ranking_config,
            },
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        sparse_results = self.sparse.search(query, top_k=self.candidate_k)
        dense_results = self.dense.search(query, top_k=self.candidate_k)
        records: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(sparse_results, start=1):
            key = _result_key(result)
            row = records.setdefault(key, {**result, "_sparse_rank": None, "_dense_rank": None})
            row["_sparse_rank"] = rank
            row["_sparse_score"] = float(result.get("score", 0.0))
            row["_rrf_score"] = row.get("_rrf_score", 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, result in enumerate(dense_results, start=1):
            key = _result_key(result)
            row = records.setdefault(key, {**result, "_sparse_rank": None, "_dense_rank": None})
            row["_dense_rank"] = rank
            row["_dense_score"] = float(result.get("score", 0.0))
            row["_rrf_score"] = row.get("_rrf_score", 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(
            records.values(),
            key=lambda row: (
                -row.get("_rrf_score", 0.0),
                row.get("_sparse_rank") or self.candidate_k + 1,
                row.get("_dense_rank") or self.candidate_k + 1,
                _result_key(row),
            ),
        )
        output: list[dict[str, Any]] = []
        for row in ranked[:top_k]:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            clean.update(
                {
                    "score": row.get("_rrf_score", 0.0),
                    "rrf_score": row.get("_rrf_score", 0.0),
                    "sparse_rank": row.get("_sparse_rank"),
                    "dense_rank": row.get("_dense_rank"),
                    "sparse_score": row.get("_sparse_score"),
                    "dense_score": row.get("_dense_score"),
                }
            )
            output.append(clean)
        return output


def _minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high == low:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


class HybridWeighted:
    """Fuse min-max-normalized sparse and dense scores."""

    def __init__(self, sparse: Retriever, dense: Retriever, candidate_k: int = 20, dense_weight: float = 0.5):
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        self.sparse = sparse
        self.dense = dense
        self.candidate_k = candidate_k
        self.dense_weight = dense_weight

    @property
    def ranking_config(self) -> dict[str, Any]:
        return {
            "retriever": "hybrid_weighted",
            "candidate_k": self.candidate_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": 1.0 - self.dense_weight,
            "normalization": "per-query min-max over each returned candidate list; missing score=0",
            "fusion_formula": "(1-dense_weight)*sparse_norm + dense_weight*dense_norm",
            "components": {
                "sparse": self.sparse.ranking_config,
                "dense": self.dense.ranking_config,
            },
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        sparse_results = self.sparse.search(query, top_k=self.candidate_k)
        dense_results = self.dense.search(query, top_k=self.candidate_k)
        sparse_map = {_result_key(result): result for result in sparse_results}
        dense_map = {_result_key(result): result for result in dense_results}
        sparse_raw = {key: float(result.get("score", 0.0)) for key, result in sparse_map.items()}
        dense_raw = {key: float(result.get("score", 0.0)) for key, result in dense_map.items()}
        sparse_norm = _minmax_scores(sparse_raw)
        dense_norm = _minmax_scores(dense_raw)
        sparse_ranks = {key: rank for rank, key in enumerate(sparse_map, start=1)}
        dense_ranks = {key: rank for rank, key in enumerate(dense_map, start=1)}
        rows: list[tuple[float, str, dict[str, Any]]] = []
        for key in set(sparse_map) | set(dense_map):
            base = sparse_map.get(key) or dense_map[key]
            score = (1.0 - self.dense_weight) * sparse_norm.get(key, 0.0) + self.dense_weight * dense_norm.get(key, 0.0)
            rows.append((score, key, base))
        rows.sort(key=lambda row: (-row[0], row[1]))
        output: list[dict[str, Any]] = []
        for score, key, base in rows[:top_k]:
            output.append(
                {
                    **base,
                    "score": score,
                    "weighted_score": score,
                    "sparse_score": sparse_raw.get(key),
                    "dense_score": dense_raw.get(key),
                    "sparse_normalized": sparse_norm.get(key, 0.0),
                    "dense_normalized": dense_norm.get(key, 0.0),
                    "sparse_rank": sparse_ranks.get(key),
                    "dense_rank": dense_ranks.get(key),
                }
            )
        return output


def load_hybrid_index(
    documents_path: Path,
    model_name: str,
    cache_path: Path | None,
    cache_dir: Path | None,
    candidate_k: int = 20,
    rrf_k: int = 60,
    fusion: str = "rrf",
    dense_weight: float = 0.5,
) -> HybridRRF | HybridWeighted:
    documents = read_jsonl(documents_path)
    sparse = BM25(documents)
    dense = DenseIndex.from_jsonl(
        documents_path,
        model_name=model_name,
        cache_path=cache_path,
        cache_dir=cache_dir,
    )
    if fusion == "rrf":
        return HybridRRF(sparse, dense, candidate_k=candidate_k, rrf_k=rrf_k)
    if fusion == "weighted":
        return HybridWeighted(sparse, dense, candidate_k=candidate_k, dense_weight=dense_weight)
    raise ValueError("fusion must be rrf or weighted")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--fusion", choices=("rrf", "weighted"), default="rrf")
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < 1 or args.rrf_k < 1 or not 0.0 <= args.dense_weight <= 1.0:
        parser.error("top-k, candidate-k and rrf-k must be at least 1; dense-weight must be between 0 and 1")
    index = load_hybrid_index(
        args.index, args.model, args.cache, args.model_cache, args.candidate_k, args.rrf_k, args.fusion, args.dense_weight
    )
    for rank, result in enumerate(index.search(args.query, args.top_k), start=1):
        print(
            f"[{rank}] score={result['score']:.6f} | sparse_rank={result['sparse_rank']} | "
            f"dense_rank={result['dense_rank']} | {result['evidence_id']} | {result['source_title']}"
        )
        print(f"    Source: {result['source_url']}")
        print(f"    {result['text'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
