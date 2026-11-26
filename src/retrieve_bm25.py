"""A small, dependency-free BM25 baseline for the capstone corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character bigrams using only stdlib."""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.findall(text.lower()):
        if "\u4e00" <= match[0] <= "\u9fff":
            if len(match) == 1:
                tokens.append(match)
            else:
                tokens.extend(match[index : index + 2] for index in range(len(match) - 1))
        else:
            tokens.append(match)
    return tokens


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class _BM25Field:
    """BM25 statistics for one document field, such as body or section title."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        field_value: Callable[[dict[str, Any]], str],
        k1: float,
        b: float,
    ):
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(field_value(doc))) for doc in documents]
        self.lengths = [sum(freq.values()) for freq in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query_terms: set[str], document_index: int) -> float:
        frequencies = self.term_frequencies[document_index]
        length_norm = self.k1 * (1.0 - self.b + self.b * self.lengths[document_index] / self.average_length)
        total = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency:
                total += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / (frequency + length_norm)
        return total


class BM25:
    """Okapi BM25 with optional boosts for section and document titles."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        section_boost: float = 0.0,
        source_title_boost: float = 0.0,
    ):
        if not documents:
            raise ValueError("BM25 needs at least one document")
        if section_boost < 0 or source_title_boost < 0:
            raise ValueError("field boosts must be non-negative")
        self.documents = documents
        self.section_boost = section_boost
        self.source_title_boost = source_title_boost
        self.content_index = _BM25Field(documents, lambda row: row.get("text", ""), k1, b)
        self.section_index = _BM25Field(documents, lambda row: row.get("section_title", ""), k1, b)
        self.title_index = _BM25Field(documents, lambda row: row.get("source_title", ""), k1, b)

    @property
    def ranking_config(self) -> dict[str, Any]:
        return {
            "retriever": "bm25",
            "bm25_k1": self.content_index.k1,
            "bm25_b": self.content_index.b,
            "section_boost": self.section_boost,
            "source_title_boost": self.source_title_boost,
        }

    def score(self, query: str, document_index: int) -> float:
        query_terms = set(tokenize(query))
        return (
            self.content_index.score(query_terms, document_index)
            + self.section_boost * self.section_index.score(query_terms, document_index)
            + self.source_title_boost * self.title_index.score(query_terms, document_index)
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        scored = [
            (self.score(query, index), index)
            for index in range(len(self.documents))
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**self.documents[index], "score": score}
            for score, index in scored[:top_k]
            if score > 0
        ]


def load_index(path: Path) -> BM25:
    return BM25(read_jsonl(path))


def main() -> None:
    # Keep Chinese evidence readable when Python inherits a legacy Windows console encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Chunk JSONL produced by chunk_documents.py")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--section-boost", type=float, default=0.0, help="Weight for BM25 matches in section headings")
    parser.add_argument("--source-title-boost", type=float, default=0.0, help="Weight for BM25 matches in document titles")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.section_boost < 0 or args.source_title_boost < 0:
        parser.error("field boosts must be non-negative")

    index = BM25(
        read_jsonl(args.index),
        section_boost=args.section_boost,
        source_title_boost=args.source_title_boost,
    )
    results = index.search(args.query, args.top_k)
    if not results:
        print("No lexical matches.")
        return
    for rank, result in enumerate(results, start=1):
        print(f"[{rank}] score={result['score']:.4f} | {result['evidence_id']} | {result['source_title']}")
        print(f"    Source: {result['source_url']}")
        print(f"    {result['text'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
