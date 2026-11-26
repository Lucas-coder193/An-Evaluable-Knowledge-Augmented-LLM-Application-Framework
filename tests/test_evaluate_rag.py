from types import SimpleNamespace

from src.evaluate_rag import evaluate


class FakePipeline:
    generator = SimpleNamespace(name="fake-generator")
    retriever = SimpleNamespace(ranking_config={"retriever": "fake"})

    def run(self, question):
        if question == "answerable":
            return {
                "should_refuse": False,
                "supporting_evidence": [{"evidence_id": "doc#answer"}],
                "evidence": [{"evidence_id": "doc#answer"}],
                "citation_support": True,
                "citations": ["E1"],
                "evidence_gate": {"reason": "evidence_passed_lexical_gate"},
                "answer": "grounded answer",
                "generation_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        return {
            "should_refuse": True,
            "supporting_evidence": [],
            "evidence": [],
            "citation_support": False,
            "citations": [],
            "evidence_gate": {"reason": "insufficient_question_term_support"},
            "answer": "refusal",
            "generation_usage": None,
        }


def test_evaluate_reports_latency_and_token_usage_for_small_run():
    report = evaluate(
        FakePipeline(),
        [
            {"id": "q1", "question": "answerable", "expected_evidence": ["doc#answer"]},
            {"id": "q2", "question": "out of scope", "expected_evidence": []},
        ],
    )

    assert report["questions_total"] == 2
    assert report["no_answer_refusal_rate"] == 1.0
    assert report["token_usage"] == {
        "requests_with_usage": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert report["latency_ms"]["count"] == 2
    assert report["latency_ms"]["p50"] >= 0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"]
    assert all(row["latency_ms"] >= 0 for row in report["rows"])


def test_explicit_refusal_label_keeps_answerable_non_rag_question_out_of_no_answer_bucket():
    class ToolLikePipeline(FakePipeline):
        def run(self, question):
            return {
                "should_refuse": False,
                "supporting_evidence": [],
                "evidence": [],
                "citation_support": False,
                "citations": [],
                "evidence_gate": {"reason": "not_applicable"},
                "answer": "4",
                "generation_usage": None,
            }

    report = evaluate(
        ToolLikePipeline(),
        [{
            "id": "calc-1",
            "question": "Calculate 2 + 2",
            "expected_evidence": [],
            "should_refuse": False,
        }],
    )

    assert report["answerable_questions"] == 1
    assert report["no_answer_questions"] == 0
