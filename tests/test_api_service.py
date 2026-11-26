from src.api_service import APIRequestError, RAGService
import pytest


class FakeRetriever:
    ranking_config = {"retriever": "fake"}

    def search(self, query, top_k=5):
        if "outside" in query:
            return []
        return [
            {
                "evidence_id": "guide#answer",
                "text": "POST 可以返回 201。",
                "source_title": "Guide",
                "section_title": "Answer",
            }
        ][:top_k]


class FakeAgent:
    name = "fake-agent"

    def generate_with_tools(self, question, registry, *, max_tool_rounds=2):
        result = registry.execute("search_fastapi_docs", {"query": question, "top_k": 1})
        return {
            "answer": "可以返回 201。【E1】",
            "should_refuse": False,
            "tool_calls": [{"name": "search_fastapi_docs", "result": result}],
            "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        }


def test_service_returns_stable_public_response_without_raw_tool_trace():
    result = RAGService(FakeAgent(), FakeRetriever()).ask("POST status 201")

    assert result["answer"] == "可以返回 201。【E1】"
    assert result["workflow_status"] == "completed"
    assert result["workflow_engine"] == "langgraph"
    assert result["workflow_trace"] == ["route", "retrieve_tool", "validate", "finish"]
    assert result["validation_retries"] == 0
    assert result["provider_retry_count"] == 0
    assert result["provider_retry_events"] == []
    assert result["attempt_reports"][0]["attempt"] == 1
    assert result["tool_error_count"] == 0
    assert result["retrieved_evidence_ids"] == ["guide#answer"]
    assert result["code_block_check"]["unsupported_code_blocks"] == 0
    assert result["usage"]["total_tokens"] == 9
    assert "tool_calls" not in result
    assert result["latency_ms"] >= 0


def test_service_gates_outside_question_without_model_call():
    class NoCallAgent(FakeAgent):
        def generate_with_tools(self, *args, **kwargs):
            raise AssertionError("model should not be called")

    result = RAGService(NoCallAgent(), FakeRetriever()).ask("outside")

    assert result["workflow_status"] == "gated_refusal"
    assert result["should_refuse"] is True
    assert result["retrieved_evidence_ids"] == []


def test_service_validates_request_limits():
    service = RAGService(FakeAgent(), FakeRetriever())
    for kwargs, message in [
        ({"question": ""}, "question must not be empty"),
        ({"question": "x", "top_k": 6}, "top_k must be between 1 and 5"),
        ({"question": "x", "max_tool_rounds": 0}, "max_tool_rounds must be between 1 and 5"),
    ]:
        try:
            service.ask(**kwargs)
        except APIRequestError as error:
            assert message in str(error)
        else:
            raise AssertionError("invalid request should fail")


def test_http_app_generates_openapi_and_serves_health_and_ask():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api_server import create_app

    app = create_app(RAGService(FakeAgent(), FakeRetriever()))
    client = TestClient(app)

    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/v1/ask" in schema.json()["paths"]
    ask_schema = schema.json()["paths"]["/v1/ask"]["post"]
    assert ask_schema["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/AskResponse"
    )
    assert {"422", "502", "504"}.issubset(ask_schema["responses"])

    homepage = client.get("/")
    assert homepage.status_code == 200
    assert "Grounded Retrieval Agent" in homepage.text

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "agent": "fake-agent",
        "retriever": "fake",
    }

    answer = client.post("/v1/ask", json={"question": "POST status 201"})
    assert answer.status_code == 200
    assert answer.json()["citation_check"]["supported"] is True
    assert answer.json()["workflow_engine"] == "langgraph"
    assert answer.json()["retrieved_evidence_ids"] == ["guide#answer"]

    invalid = client.post("/v1/ask", json={"question": "", "top_k": 3})
    assert invalid.status_code == 422

    invalid_top_k = client.post("/v1/ask", json={"question": "valid", "top_k": 6})
    assert invalid_top_k.status_code == 422


def test_http_maps_provider_failure_and_incomplete_workflow_to_gateway_errors():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api_server import create_app

    class FailedAgent(FakeAgent):
        def __init__(self, workflow_status):
            self.workflow_status = workflow_status

        def generate_with_tools(self, question, registry, *, max_tool_rounds=2):
            return {
                "answer": "无法回答。",
                "should_refuse": True,
                "tool_calls": [],
                "workflow_status": self.workflow_status,
                "error": "upstream unavailable",
            }

    request = {"question": "POST status 201"}
    failed_client = TestClient(create_app(RAGService(FailedAgent("request_failed"), FakeRetriever())))
    incomplete_client = TestClient(create_app(RAGService(FailedAgent("workflow_incomplete"), FakeRetriever())))

    failed = failed_client.post("/v1/ask", json=request)
    assert failed.status_code == 502
    assert failed.json()["detail"] == "upstream unavailable"

    incomplete = incomplete_client.post("/v1/ask", json=request)
    assert incomplete.status_code == 504
    assert incomplete.json()["detail"] == "upstream unavailable"


def test_http_retry_exhaustion_preserves_failure_counters():
    from fastapi.testclient import TestClient
    from src.api_server import create_app
    from src.llm_provider import OpenAIResponsesGenerator, TransientLLMProviderError

    calls = []

    def unavailable(*args):
        calls.append(1)
        raise TransientLLMProviderError('upstream unavailable', status_code=503)

    agent = OpenAIResponsesGenerator(
        api_key='test-secret', model='test-model', transport=unavailable, sleeper=lambda _: None
    )
    with TestClient(create_app(RAGService(agent, FakeRetriever()))) as client:
        response = client.post('/v1/ask', json={'question': 'POST status 201'})
    assert response.status_code == 502
    assert len(calls) == 3
    body = response.json()
    assert body['provider_retry_count'] == 2
    assert body['validation_retries'] == 0
    assert len(body['attempt_reports']) == 1
    assert all(event['workflow_attempt'] == 1 for event in body['provider_retry_events'])
