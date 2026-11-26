import json
from pathlib import Path

import numpy as np

from src.evaluate_retrieval import evaluate
from src.retrieve_dense import DenseIndex


class TinyEncoder:
    version = "test"

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.document_calls += 1
        return np.asarray([self.vectors.get(text, [1.0, 0.0]) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.vectors.get(text, [1.0, 0.0]), dtype=np.float32)


def write_documents(path: Path, documents: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents), encoding="utf-8")


def test_dense_search_ranks_by_cosine_and_preserves_provenance() -> None:
    docs = [
        {"chunk_id": "other", "evidence_id": "guide#other", "section_title": "Other", "text": "Other content", "source_url": "https://example.org/other"},
        {"chunk_id": "target", "evidence_id": "guide#target", "section_title": "Target", "text": "Target content", "source_url": "https://example.org/target"},
    ]
    encoder = TinyEncoder({"Target\n\nTarget content": [1.0, 0.0], "Other\n\nOther content": [0.0, 1.0], "target query": [1.0, 0.0]})
    index = DenseIndex(docs, encoder.embed_documents(["Other\n\nOther content", "Target\n\nTarget content"]), "test-model", "fingerprint").attach_encoder(encoder)

    results = index.search("target query", top_k=1)

    assert results[0]["chunk_id"] == "target"
    assert results[0]["source_url"] == "https://example.org/target"
    assert results[0]["score"] == 1.0


def test_dense_cache_round_trip_and_reuses_embeddings(tmp_path: Path) -> None:
    documents_path = tmp_path / "chunks.jsonl"
    cache_path = tmp_path / "dense.npz"
    docs = [{"chunk_id": "one", "evidence_id": "guide#one", "section_title": "One", "text": "content"}]
    write_documents(documents_path, docs)
    first_encoder = TinyEncoder({"One\n\ncontent": [0.0, 1.0], "query": [0.0, 1.0]})
    DenseIndex.from_jsonl(documents_path, "test-model", cache_path, encoder=first_encoder)

    second_encoder = TinyEncoder({"query": [0.0, 1.0]})
    loaded = DenseIndex.from_jsonl(documents_path, "test-model", cache_path, encoder=second_encoder)

    assert first_encoder.document_calls == 1
    assert second_encoder.document_calls == 0
    assert loaded.search("query", top_k=1)[0]["evidence_id"] == "guide#one"
    assert loaded.ranking_config["embedding_library_version"] == "test"


def test_dense_cache_rebuilds_when_source_corpus_changes(tmp_path: Path) -> None:
    documents_path = tmp_path / "chunks.jsonl"
    cache_path = tmp_path / "dense.npz"
    write_documents(documents_path, [{"chunk_id": "one", "evidence_id": "guide#one", "text": "old"}])
    DenseIndex.from_jsonl(documents_path, "test-model", cache_path, encoder=TinyEncoder())
    write_documents(documents_path, [{"chunk_id": "one", "evidence_id": "guide#one", "text": "new"}])
    encoder = TinyEncoder()

    loaded = DenseIndex.from_jsonl(documents_path, "test-model", cache_path, encoder=encoder)

    assert encoder.document_calls == 1
    assert loaded.documents[0]["text"] == "new"


def test_shared_evaluator_accepts_dense_retriever() -> None:
    docs = [{"chunk_id": "one", "evidence_id": "guide#one", "text": "answer"}]
    encoder = TinyEncoder({"question": [1.0, 0.0]})
    index = DenseIndex(docs, np.asarray([[1.0, 0.0]]), "test-model", "fingerprint").attach_encoder(encoder)

    report = evaluate(index, [{"id": "q1", "question": "question", "expected_evidence": ["guide#one"]}], top_k=1)

    assert report["recall@1"] == 1.0
    assert report["ranking_config"]["retriever"] == "dense_cosine"
