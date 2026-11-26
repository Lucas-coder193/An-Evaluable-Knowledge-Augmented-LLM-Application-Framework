from typing import Any

import pytest

from src.retrieve_hybrid import HybridRRF, HybridWeighted


class FakeRetriever:
    ranking_config = {"retriever": "fake"}

    def __init__(self, results: list[dict[str, Any]]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def result(chunk_id: str, score: float) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "evidence_id": f"guide#{chunk_id}",
        "source_title": "Guide",
        "source_url": "https://example.org/guide",
        "text": chunk_id,
        "score": score,
    }


def test_rrf_rewards_documents_present_in_both_rankings() -> None:
    sparse = FakeRetriever([result("sparse-only", 4), result("shared", 3)])
    dense = FakeRetriever([result("dense-only", 0.9), result("shared", 0.8)])
    index = HybridRRF(sparse, dense, candidate_k=2, rrf_k=1)

    results = index.search("query", top_k=3)

    assert [row["chunk_id"] for row in results] == ["shared", "sparse-only", "dense-only"]
    assert results[0]["sparse_rank"] == 2
    assert results[0]["dense_rank"] == 2
    assert results[0]["source_url"] == "https://example.org/guide"
    assert sparse.calls == [("query", 2)]
    assert dense.calls == [("query", 2)]


def test_rrf_is_deterministic_for_ties_and_reports_config() -> None:
    sparse = FakeRetriever([result("a", 1)])
    dense = FakeRetriever([result("b", 1)])
    index = HybridRRF(sparse, dense, candidate_k=1, rrf_k=60)

    results = index.search("query", top_k=2)

    assert [row["chunk_id"] for row in results] == ["a", "b"]
    assert index.ranking_config["retriever"] == "hybrid_rrf"
    assert index.ranking_config["rrf_k"] == 60


def test_weighted_fusion_normalizes_incompatible_score_ranges() -> None:
    sparse = FakeRetriever([result("lexical", 100.0), result("shared", 90.0), result("low", 1.0)])
    dense = FakeRetriever([result("shared", 0.8), result("semantic", 0.7)])
    index = HybridWeighted(sparse, dense, candidate_k=3, dense_weight=0.5)

    results = index.search("query", top_k=3)

    assert results[0]["chunk_id"] == "shared"
    assert results[0]["sparse_normalized"] == pytest.approx(89 / 99)
    assert results[0]["dense_normalized"] == 1.0
    assert index.ranking_config["normalization"].startswith("per-query min-max")
