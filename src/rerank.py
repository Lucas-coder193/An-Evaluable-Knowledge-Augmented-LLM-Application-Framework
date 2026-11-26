"""Second-stage candidate reranking with late interaction or test doubles."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

if __package__:
    from .retrieve_hybrid import HybridWeighted, load_hybrid_index
else:
    from retrieve_hybrid import HybridWeighted, load_hybrid_index


class PairReranker(Protocol):
    ranking_config: dict[str, Any]

    def score(self, query: str, documents: list[str]) -> np.ndarray: ...


def _as_token_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        return matrix[0]
    raise ValueError("late-interaction embeddings must have shape [tokens, dimensions]")


def maxsim_score(query_tokens: np.ndarray, document_tokens: np.ndarray) -> float:
    """ColBERT-style MaxSim: sum over query tokens of their best document match."""
    query_tokens = _as_token_matrix(query_tokens)
    document_tokens = _as_token_matrix(document_tokens)
    if query_tokens.shape[1] != document_tokens.shape[1]:
        raise ValueError("query and document token embeddings have different dimensions")
    if not np.isfinite(query_tokens).all() or not np.isfinite(document_tokens).all():
        raise ValueError("token embeddings contain NaN or infinite values")
    query_norm = np.linalg.norm(query_tokens, axis=1, keepdims=True)
    document_norm = np.linalg.norm(document_tokens, axis=1, keepdims=True)
    query_unit = np.divide(query_tokens, query_norm, out=np.zeros_like(query_tokens), where=query_norm > 0)
    document_unit = np.divide(document_tokens, document_norm, out=np.zeros_like(document_tokens), where=document_norm > 0)
    similarities = query_unit @ document_unit.T
    return float(np.max(similarities, axis=1).sum())


class LateInteractionReranker:
    """FastEmbed adapter for a multilingual late-interaction model."""

    def __init__(self, model_name: str = "lightonai/mLateOn", cache_dir: Path | None = None):
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError as error:
            raise RuntimeError("Late-interaction reranking requires FastEmbed") from error
        self.model_name = model_name
        self.model = LateInteractionTextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.library_version = importlib.metadata.version("fastembed")

    @property
    def ranking_config(self) -> dict[str, Any]:
        return {
            "reranker": "late_interaction_maxsim",
            "model": self.model_name,
            "library": "fastembed",
            "library_version": self.library_version,
            "similarity": "sum_of_query_token_max_cosine",
        }

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        query_tokens = _as_token_matrix(next(iter(self.model.query_embed(query))))
        document_tokens = [_as_token_matrix(value) for value in self.model.embed(documents)]
        return np.asarray([maxsim_score(query_tokens, document) for document in document_tokens], dtype=np.float32)


class RerankedRetriever:
    """Retrieve a candidate pool, rerank it, then return the final top-k."""

    def __init__(self, base_retriever: Any, reranker: PairReranker, candidate_k: int = 20):
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    @property
    def ranking_config(self) -> dict[str, Any]:
        return {
            "retriever": "two_stage_rerank",
            "candidate_k": self.candidate_k,
            "candidate_retriever": self.base_retriever.ranking_config,
            "reranker": self.reranker.ranking_config,
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = self.base_retriever.search(query, top_k=self.candidate_k)
        if not candidates:
            return []
        texts = [self._candidate_text(row) for row in candidates]
        rerank_scores = self.reranker.score(query, texts)
        if len(rerank_scores) != len(candidates):
            raise ValueError("reranker must return one score per candidate")
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda pair: (-float(rerank_scores[pair[0]]), pair[0]))
        output: list[dict[str, Any]] = []
        for rerank_rank, (candidate_index, candidate) in enumerate(indexed[:top_k], start=1):
            output.append(
                {
                    **candidate,
                    "score": float(rerank_scores[candidate_index]),
                    "rerank_score": float(rerank_scores[candidate_index]),
                    "rerank_rank": rerank_rank,
                    "candidate_rank": candidate_index + 1,
                    "candidate_score": candidate.get("score"),
                }
            )
        return output

    @staticmethod
    def _candidate_text(row: dict[str, Any]) -> str:
        section = str(row.get("section_title", "")).strip()
        body = str(row.get("text", "")).strip()
        return "\n\n".join(part for part in (section, body) if part)


def load_reranked_index(
    documents_path: Path,
    embedding_model: str,
    embedding_cache: Path | None,
    model_cache: Path | None,
    reranker_model: str = "lightonai/mLateOn",
    candidate_k: int = 20,
    dense_weight: float = 0.5,
) -> RerankedRetriever:
    base = load_hybrid_index(
        documents_path,
        model_name=embedding_model,
        cache_path=embedding_cache,
        cache_dir=model_cache,
        candidate_k=candidate_k,
        fusion="weighted",
        dense_weight=dense_weight,
    )
    reranker = LateInteractionReranker(reranker_model, cache_dir=model_cache)
    return RerankedRetriever(base, reranker, candidate_k=candidate_k)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="lightonai/mLateOn")
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < 1 or not 0.0 <= args.dense_weight <= 1.0:
        parser.error("top-k and candidate-k must be at least 1; dense-weight must be between 0 and 1")
    index = load_reranked_index(
        args.index,
        args.embedding_model,
        args.embedding_cache,
        args.model_cache,
        args.reranker_model,
        args.candidate_k,
        args.dense_weight,
    )
    for rank, result in enumerate(index.search(args.query, args.top_k), start=1):
        print(f"[{rank}] rerank={result['rerank_score']:.4f} | candidate_rank={result['candidate_rank']} | {result['evidence_id']}")
        print(f"    Source: {result.get('source_url', '')}")
        print(f"    {result.get('text', '').replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
