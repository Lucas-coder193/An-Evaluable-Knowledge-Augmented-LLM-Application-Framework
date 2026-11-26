"""Small-corpus dense retrieval using FastEmbed and cosine similarity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

if __package__:
    from .retrieve_bm25 import read_jsonl
else:
    from retrieve_bm25 import read_jsonl


class Encoder(Protocol):
    """Encoder interface kept small so index behavior can be tested offline."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEncoder:
    """Adapter around FastEmbed's CPU-friendly query/document embedding API."""

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires FastEmbed. Install the project vector requirements first."
            ) from error
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.version = importlib.metadata.version("fastembed")
        import onnxruntime

        self.execution_device = onnxruntime.get_device()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(list(self.model.query_embed(text)), dtype=np.float32)


def _unit_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("embeddings must be a two-dimensional matrix with at least one column")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def _embedding_text(document: dict[str, Any]) -> str:
    """Give the model a section label along with the body that BM25 searches."""
    parts = [document.get("section_title", ""), document.get("text", "")]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _corpus_fingerprint(documents: list[dict[str, Any]]) -> str:
    canonical = json.dumps(documents, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DenseIndex:
    """In-memory cosine index with a safe, reusable NumPy cache on disk."""

    CACHE_SCHEMA_VERSION = 1

    def __init__(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
        model_name: str,
        corpus_fingerprint: str,
    ):
        if not documents:
            raise ValueError("dense index needs at least one document")
        matrix = _unit_normalize(embeddings)
        if matrix.shape[0] != len(documents):
            raise ValueError("one document embedding is required for each document")
        self.documents = documents
        self.embeddings = matrix
        self.model_name = model_name
        self.corpus_fingerprint = corpus_fingerprint

    @property
    def ranking_config(self) -> dict[str, Any]:
        return {
            "retriever": "dense_cosine",
            "embedding_model": self.model_name,
            "embedding_library": "fastembed",
            "embedding_library_version": getattr(getattr(self, "_query_encoder", None), "version", "unknown"),
            "execution_device": getattr(getattr(self, "_query_encoder", None), "execution_device", "unknown"),
            "embedding_dimension": int(self.embeddings.shape[1]),
            "document_text_fields": ["section_title", "text"],
            "similarity": "cosine",
            "cache_schema_version": self.CACHE_SCHEMA_VERSION,
        }

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        cache_path: Path | None = None,
        cache_dir: Path | None = None,
        encoder: Encoder | None = None,
    ) -> "DenseIndex":
        documents = read_jsonl(path)
        if not documents:
            raise ValueError("dense index needs at least one document")
        fingerprint = _corpus_fingerprint(documents)
        active_encoder = encoder or FastEmbedEncoder(model_name, cache_dir=cache_dir)
        if cache_path and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    metadata = json.loads(str(cache["metadata"].item()))
                    valid = (
                        metadata.get("schema_version") == cls.CACHE_SCHEMA_VERSION
                        and metadata.get("model_name") == model_name
                        and metadata.get("corpus_fingerprint") == fingerprint
                    )
                    if valid:
                        return cls(documents, cache["embeddings"], model_name, fingerprint).attach_encoder(active_encoder)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                # A missing, corrupt, or outdated cache is rebuilt from its source documents.
                pass

        texts = [_embedding_text(document) for document in documents]
        embeddings = _unit_normalize(active_encoder.embed_documents(texts))
        index = cls(documents, embeddings, model_name, fingerprint).attach_encoder(active_encoder)
        if cache_path:
            index.save(cache_path)
        return index

    def save(self, path: Path) -> None:
        metadata = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "model_name": self.model_name,
            "corpus_fingerprint": self.corpus_fingerprint,
            "documents": self.documents,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        with temp_path.open("wb") as output:
            np.savez_compressed(
                output,
                embeddings=self.embeddings,
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
        temp_path.replace(path)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = _unit_normalize(self._encode_query(query))
        if query_vector.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query and document embeddings have different dimensions")
        scores = self.embeddings @ query_vector[0]
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [{**self.documents[index], "score": float(scores[index])} for index in ranked]

    def _encode_query(self, query: str) -> np.ndarray:
        if not hasattr(self, "_query_encoder"):
            raise RuntimeError("This index was loaded from cache without an encoder; attach an encoder to search.")
        return self._query_encoder.embed_query(query)

    def attach_encoder(self, encoder: Encoder) -> "DenseIndex":
        """Attach the runtime encoder after loading a cache without recomputing documents."""
        self._query_encoder = encoder
        return self


def load_dense_index(
    documents_path: Path,
    model_name: str,
    cache_path: Path | None,
    cache_dir: Path | None,
) -> DenseIndex:
    encoder = FastEmbedEncoder(model_name, cache_dir=cache_dir)
    return DenseIndex.from_jsonl(
        documents_path,
        model_name=model_name,
        cache_path=cache_path,
        cache_dir=cache_dir,
        encoder=encoder,
    ).attach_encoder(encoder)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Chunk JSONL produced by chunk_documents.py")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--cache", type=Path, default=Path("data/processed/fastapi_v1_bge_small_zh_v1_5.npz"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/models/fastembed"))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    index = load_dense_index(args.index, args.model, args.cache, args.model_cache)
    for rank, result in enumerate(index.search(args.query, args.top_k), start=1):
        print(f"[{rank}] cosine={result['score']:.4f} | {result['evidence_id']} | {result['source_title']}")
        print(f"    Source: {result['source_url']}")
        print(f"    {result['text'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
