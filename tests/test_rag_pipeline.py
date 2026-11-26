from typing import Any

from src.rag_pipeline import EvidenceGate, ExtractiveGenerator, RAGPipeline, build_prompt, lexical_coverage


class FakeRetriever:
    ranking_config = {"retriever": "fake"}

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.calls.append((query, top_k))
        return self.rows[:top_k]


def row(evidence_id: str, text: str, score: float = 1.0) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_title": "Guide",
        "source_url": "https://example.org/guide",
        "text": text,
        "score": score,
        "rerank_score": score,
    }


def test_extractive_pipeline_builds_labeled_context_and_valid_citation():
    retriever = FakeRetriever([row("guide#answer", "使用 POST 创建资源，响应状态码可以设置为 201。")])
    pipeline = RAGPipeline(retriever, top_k=3)

    result = pipeline.run("创建资源时 POST 可以使用什么状态码？")

    assert result["should_refuse"] is False
    assert result["answer"] == "使用 POST 创建资源，响应状态码可以设置为 201。"
    assert result["citations"] == ["E1"]
    assert result["citation_support"] is True
    assert "E1" in result["prompt"]
    assert retriever.calls == [("创建资源时 POST 可以使用什么状态码？", 3)]


def test_gate_refuses_when_top_chunk_only_matches_generic_domain_terms():
    retriever = FakeRetriever([row("guide#generic", "FastAPI 会自动生成 API 文档。", score=20.0)])
    pipeline = RAGPipeline(retriever, top_k=1)

    result = pipeline.run("FastAPI 官方推荐哪家云厂商托管 PostgreSQL？")

    assert lexical_coverage("FastAPI 官方推荐哪家云厂商托管 PostgreSQL？", retriever.rows[0]["text"]) == 0.0
    assert result["should_refuse"] is True
    assert result["answer"] == "证据不足，无法回答。"
    assert result["citations"] == []
    assert result["evidence_gate"]["reason"] == "insufficient_question_term_support"


def test_gate_accepts_relevant_lower_ranked_chunk_when_top_chunk_is_generic():
    retriever = FakeRetriever(
        [
            row("guide#generic", "FastAPI 会自动生成 API 文档。", score=20.0),
            row("guide#answer", "重量级计算或跨多机调度应交给外部任务队列。", score=19.0),
        ]
    )
    result = RAGPipeline(retriever, top_k=2).run("哪些工作应考虑交给外部任务队列？")

    assert result["should_refuse"] is False
    assert result["evidence_gate"]["reason"] == "evidence_passed_lexical_gate"


def test_invalid_generator_citations_are_not_allowed_to_pass():
    class BadGenerator(ExtractiveGenerator):
        name = "bad"

        def generate(self, question, evidence):
            return {"answer": "unsupported", "citations": ["E99"]}

    retriever = FakeRetriever([row("guide#answer", "POST 可以设置为 201。")])
    result = RAGPipeline(retriever, generator=BadGenerator()).run("POST 可以设置为多少？")

    assert result["should_refuse"] is True
    assert result["citation_support"] is False
    assert result["evidence_gate"]["reason"] == "generator_returned_invalid_answer_or_citation"


def test_prompt_requires_grounded_json_ready_answering_behavior():
    prompt = build_prompt("问题", [{"label": "E1", "text": "证据", "source_url": "url"}])

    assert "只能依据给出的证据回答" in prompt
    assert "E1" in prompt
    assert "不要在 answer 正文中输出" in prompt
    assert "回答末尾用" not in prompt
    assert "证据不足" in prompt


def test_gate_handles_no_results():
    decision = EvidenceGate().assess("问题", [])

    assert decision["should_refuse"] is True
    assert decision["reason"] == "no_retrieval_results"


def test_generator_can_request_a_grounded_refusal():
    class RefusingGenerator(ExtractiveGenerator):
        name = "refusing"

        def generate(self, question, evidence):
            return {"answer": "I do not know", "citations": [], "should_refuse": True}

    retriever = FakeRetriever([row("guide#answer", "POST 可以设置为 201。")])
    result = RAGPipeline(retriever, generator=RefusingGenerator()).run("POST 可以设置为什么？")

    assert result["should_refuse"] is True
    assert result["answer"] == "证据不足，无法回答。"
    assert result["citations"] == []
    assert result["evidence_gate"]["reason"] == "generator_requested_refusal"


def test_mixed_valid_and_invalid_citations_fail_closed():
    class MixedCitationGenerator(ExtractiveGenerator):
        name = "mixed-citations"

        def generate(self, question, evidence):
            return {"answer": "answer", "citations": ["E1", "E99"], "should_refuse": False}

    retriever = FakeRetriever([row("guide#answer", "POST 可以设置为 201。")])
    result = RAGPipeline(retriever, generator=MixedCitationGenerator()).run("POST 可以设置为什么？")

    assert result["should_refuse"] is True
    assert result["citations"] == []
    assert result["evidence_gate"]["reason"] == "generator_returned_invalid_answer_or_citation"


def test_pipeline_exposes_generation_usage_for_evaluation():
    class UsageGenerator(ExtractiveGenerator):
        name = "usage-generator"

        def generate(self, question, evidence):
            return {
                "answer": "POST 可以设置为 201。",
                "citations": ["E1"],
                "should_refuse": False,
                "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
            }

    retriever = FakeRetriever([row("guide#answer", "创建资源的 POST 可以将状态码设为 201。")])
    result = RAGPipeline(retriever, generator=UsageGenerator()).run("创建资源 POST 状态码可以设为多少？")

    assert result["generation_usage"] == {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18}
