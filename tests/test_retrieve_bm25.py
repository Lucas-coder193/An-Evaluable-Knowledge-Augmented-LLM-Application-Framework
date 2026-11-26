from src.retrieve_bm25 import BM25


def make_chunk(chunk_id: str, text: str, section_title: str) -> dict[str, str]:
    return {
        "chunk_id": chunk_id,
        "evidence_id": f"doc#{chunk_id}",
        "source_url": "https://example.org/docs",
        "source_title": "API Guide",
        "section_title": section_title,
        "text": text,
    }


def test_body_terms_find_the_matching_chunk_and_keep_provenance() -> None:
    expected = make_chunk("timeouts", "Configure request timeout in seconds.", "Client settings")
    unrelated = make_chunk("headers", "Set a custom response header.", "Response headers")
    result = BM25([unrelated, expected]).search("request timeout", top_k=1)

    assert result[0]["chunk_id"] == "timeouts"
    assert result[0]["source_url"] == "https://example.org/docs"


def test_section_boost_can_retrieve_a_heading_match_missing_from_body() -> None:
    query_parameters = make_chunk("query", "Values are parsed and checked.", "Query Parameters")
    request_body = make_chunk("body", "JSON is parsed and validated.", "Request Body")

    baseline = BM25([request_body, query_parameters]).search("query parameters", top_k=1)
    fielded = BM25([request_body, query_parameters], section_boost=2.0).search("query parameters", top_k=1)

    assert baseline == []
    assert fielded[0]["chunk_id"] == "query"


def test_query_with_no_lexical_match_returns_no_results() -> None:
    chunks = [make_chunk("one", "Configure request timeout in seconds.", "Client settings")]

    assert BM25(chunks).search("quantum entanglement", top_k=3) == []
