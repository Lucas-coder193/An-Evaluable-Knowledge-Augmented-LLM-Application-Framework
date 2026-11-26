import numpy as np
import pytest

from src.rerank import RerankedRetriever, maxsim_score


class FakeRetriever:
    ranking_config = {"retriever": "fake_candidates"}

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.rows[:top_k]


class FakeReranker:
    ranking_config = {"reranker": "fake"}

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, documents))
        return self.scores[: len(documents)]


def row(chunk_id):
    return {
        "chunk_id": chunk_id,
        "evidence_id": f"guide#{chunk_id}",
        "section_title": "Section",
        "text": chunk_id,
        "score": 0.1,
        "source_url": "https://example.org/guide",
    }


def test_maxsim_rewards_a_query_token_with_a_matching_document_token():
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    document = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    unrelated = np.asarray([[-1.0, 0.0], [0.0, -1.0]])

    assert maxsim_score(query, document) > maxsim_score(query, unrelated)
    assert maxsim_score(query, document) == pytest.approx(2.0, abs=0.02)


def test_reranked_retriever_uses_candidate_pool_and_preserves_provenance():
    base = FakeRetriever([row("first"), row("second"), row("third")])
    reranker = FakeReranker([0.2, 0.9, 0.4])
    index = RerankedRetriever(base, reranker, candidate_k=3)

    results = index.search("query", top_k=2)

    assert [item["chunk_id"] for item in results] == ["second", "third"]
    assert results[0]["candidate_rank"] == 2
    assert results[0]["rerank_rank"] == 1
    assert results[0]["source_url"] == "https://example.org/guide"
    assert base.calls == [("query", 3)]
    assert reranker.calls[0][0] == "query"


def test_reranker_rejects_wrong_number_of_scores():
    base = FakeRetriever([row("one")])
    reranker = FakeReranker([])

    with pytest.raises(ValueError, match="one score"):
        RerankedRetriever(base, reranker, candidate_k=1).search("query", top_k=1)
