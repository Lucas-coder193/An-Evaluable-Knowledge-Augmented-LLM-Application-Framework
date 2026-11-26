from src.evaluate_tool_agent import evaluate


class FakeRetriever:
    ranking_config = {"retriever": "fake"}

    def search(self, query, top_k=5):
        if "outside" in query:
            return []
        return [
            {
                "evidence_id": "guide#answer",
                "source_title": "Guide",
                "source_url": "https://example.test",
                "section_title": "Answer",
                "text": "POST 可以返回 201。",
                "score": 1.0,
            }
        ][:top_k]


class FakeAgent:
    name = "fake-tool-agent"

    def generate_with_tools(self, question, registry, *, max_tool_rounds=2):
        if "outside" in question:
            return {
                "answer": "证据不足，无法回答。",
                "should_refuse": True,
                "tool_calls": [
                    {
                        "name": "search_fastapi_docs",
                        "result": {"query": question, "results": []},
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            }
        tool_result = registry.execute(
            "search_fastapi_docs", {"query": question, "top_k": 1}
        )
        return {
            "answer": "可以返回 201。【guide#answer】",
            "should_refuse": False,
            "tool_calls": [{"name": "search_fastapi_docs", "result": tool_result}],
            "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
        }


def test_tool_agent_evaluation_reports_calls_hits_refusals_and_usage():
    report = evaluate(
        FakeAgent(),
        FakeRetriever(),
        [
            {"id": "q1", "question": "POST status 201", "expected_evidence": ["guide#answer"]},
            {"id": "q2", "question": "outside", "expected_evidence": []},
        ],
    )

    assert report["questions_total"] == 2
    assert report["refusal_accuracy"] == 1.0
    assert report["retrieval_tool_call_rate"] == 0.5
    assert report["answerable_retrieval_hit_rate"] == 1.0
    assert report["citation_support_rate"] == 0.5
    assert report["answerable_citation_support_rate"] == 1.0
    assert report["no_answer_refusal_rate"] == 1.0
    assert report["tool_calls_total"] == 1
    assert report["token_usage"] == {
        "questions_with_usage": 1,
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
    }
    assert report["rows"][0]["retrieval_hit"] is True


def test_explicit_refusal_label_keeps_tool_only_question_answerable():
    report = evaluate(
        FakeAgent(),
        FakeRetriever(),
        [{
            "id": "calc-1",
            "question": "Calculate 2 + 2",
            "expected_evidence": [],
            "should_refuse": False,
        }],
    )

    assert report["no_answer_questions"] == 0
    assert report["answerable_questions"] == 1
    assert report["rows"][0]["expected_refusal"] is False


def test_tool_agent_evaluation_fails_closed_when_agent_request_errors():
    class ErrorAgent(FakeAgent):
        def generate_with_tools(self, question, registry, *, max_tool_rounds=2):
            raise RuntimeError("unexpected")

    # Only provider/configuration errors are expected in the real adapter; a
    # programming error should remain visible instead of being hidden.
    try:
        evaluate(
            ErrorAgent(),
            FakeRetriever(),
            [{"id": "q1", "question": "POST status 201", "expected_evidence": ["guide#answer"]}],
        )
    except RuntimeError as error:
        assert str(error) == "unexpected"
    else:
        raise AssertionError("unexpected programming errors must not be swallowed")


def test_tool_agent_evaluation_excludes_failed_requests_from_quality_metrics():
    class FailedAgent(FakeAgent):
        def generate_with_tools(self, question, registry, *, max_tool_rounds=2):
            return {
                "answer": "模型请求失败，无法可靠回答。",
                "should_refuse": True,
                "tool_calls": [],
                "workflow_status": "request_failed",
                "error": "HTTP 503",
                "usage": {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10},
            }

    report = evaluate(
        FailedAgent(),
        FakeRetriever(),
        [{"id": "q1", "question": "POST status 201", "expected_evidence": ["guide#answer"]}],
    )

    assert report["failed_requests"] == 1
    assert report["evaluated_questions"] == 0
    assert report["refusal_accuracy"] == 0.0
    assert report["answerable_questions"] == 0
    assert report["rows"][0]["predicted_refusal"] is None
    assert report["token_usage"]["total_tokens"] == 10


def test_tool_agent_evaluation_excludes_incomplete_workflows_from_quality_metrics():
    class IncompleteAgent(FakeAgent):
        def generate_with_tools(self, question, registry, *, max_tool_rounds=2):
            return {
                "answer": "工具调用达到轮数上限，无法可靠回答。",
                "should_refuse": True,
                "tool_calls": [],
                "workflow_status": "workflow_incomplete",
                "error": "tool-call loop exceeded max_tool_rounds=2",
                "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
            }

    report = evaluate(
        IncompleteAgent(),
        FakeRetriever(),
        [{"id": "q1", "question": "POST status 201", "expected_evidence": ["guide#answer"]}],
    )

    assert report["workflow_incomplete"] == 1
    assert report["evaluated_questions"] == 0
    assert report["refusal_accuracy"] == 0.0
    assert report["rows"][0]["predicted_refusal"] is None
    assert report["token_usage"]["total_tokens"] == 15
