from src.rag_tool_demo import run_tool_rag
from src.llm_provider import LLMProviderError
from src.workflow_graph import run_tool_rag_graph
import pytest


class FakeRetriever:
    def search(self, query, top_k=5):
        return [{"evidence_id": "guide#status", "text": "POST can use status code 201."}]


class RetryAgent:
    def __init__(self):
        self.calls = 0

    def generate_with_tools(self, question, registry, *, max_tool_rounds=2, required_first_tool=None):
        self.calls += 1
        if self.calls == 1:
            answer = "201 is supported."
        else:
            answer = "201 is supported.【E1】"
        search = registry.execute("search_fastapi_docs", {"query": "status code 201", "top_k": 1})
        return {
            "answer": answer,
            "should_refuse": False,
            "tool_calls": [{"name": "search_fastapi_docs", "result": search}],
            "workflow_status": "completed",
        }


def test_langgraph_retries_once_after_citation_validation_failure():
    agent = RetryAgent()
    result = run_tool_rag("How can POST return 201?", agent, FakeRetriever())

    assert agent.calls == 2
    assert result["workflow_engine"] == "langgraph"
    assert result["workflow_trace"] == [
        "route",
        "retrieve_tool",
        "validate",
        "retrieve_tool",
        "validate",
        "finish",
    ]
    assert result["validation_retries"] == 1
    assert result["citation_check"]["supported"] is True
    assert result["should_refuse"] is False


def test_langgraph_finishes_after_retry_limit_without_claiming_success():
    class AlwaysMissingCitation(RetryAgent):
        def generate_with_tools(self, question, registry, *, max_tool_rounds=2, required_first_tool=None):
            self.calls += 1
            search = registry.execute("search_fastapi_docs", {"query": "status code 201", "top_k": 1})
            return {
                "answer": "201 is supported.",
                "should_refuse": False,
                "tool_calls": [{"name": "search_fastapi_docs", "result": search}],
                "workflow_status": "completed",
            }

    agent = AlwaysMissingCitation()
    result = run_tool_rag("How can POST return 201?", agent, FakeRetriever())

    assert agent.calls == 2
    assert result["validation_retries"] == 1
    assert result["citation_check"]["supported"] is False
    assert result["should_refuse"] is True


def test_langgraph_short_circuits_on_evidence_gate():
    class EmptyRetriever:
        def search(self, query, top_k=5):
            return []

    class NoCallAgent:
        def generate_with_tools(self, *_args, **_kwargs):
            raise AssertionError("gated question must not invoke the model")

    result = run_tool_rag("out of scope question", NoCallAgent(), EmptyRetriever())

    assert result["workflow_status"] == "gated_refusal"
    assert result["workflow_engine"] == "langgraph"
    assert result["workflow_trace"] == ["route", "finish"]


def test_validation_retry_keeps_usage_and_provider_events_from_both_attempts():
    class MeteredAgent(RetryAgent):
        def generate_with_tools(self, *args, **kwargs):
            result = super().generate_with_tools(*args, **kwargs)
            result["usage"] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            result["provider_retry_events"] = [{"reason": "http_429", "model_round": 1}]
            return result

    result = run_tool_rag("How can POST return 201?", MeteredAgent(), FakeRetriever())
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    assert result["provider_retry_count"] == 2
    assert [event["workflow_attempt"] for event in result["provider_retry_events"]] == [1, 2]
    assert len(result["attempt_reports"]) == 2


@pytest.mark.parametrize("failure", ["refusal", "tool_error", "empty_evidence"])
def test_validation_does_not_restart_refusal_tool_failure_or_empty_evidence(failure):
    class FailingAgent(RetryAgent):
        def generate_with_tools(self, *args, **kwargs):
            result = super().generate_with_tools(*args, **kwargs)
            if failure == "refusal":
                result["should_refuse"] = True
            elif failure == "tool_error":
                result["tool_calls"][0] = {"name": "calculator", "error": "invalid expression"}
            else:
                result["tool_calls"][0]["result"]["results"] = []
            return result

    agent = FailingAgent()
    result = run_tool_rag("How can POST return 201?", agent, FakeRetriever())
    assert agent.calls == 1
    assert result["validation_retries"] == 0
    assert result["should_refuse"] is True


def test_preflight_retrieval_outage_is_not_mislabelled_as_no_evidence():
    class BrokenRetriever:
        def search(self, *args, **kwargs):
            raise RuntimeError("internal path must not be exposed")

    agent = RetryAgent()
    result = run_tool_rag("How can POST return 201?", agent, BrokenRetriever())
    assert agent.calls == 0
    assert result["workflow_status"] == "request_failed"
    assert result["error"] == "preflight retrieval failed: RuntimeError"


def test_invalid_provider_response_ends_graph_without_workflow_retry():
    class InvalidAgent(RetryAgent):
        def generate_with_tools(self, *args, **kwargs):
            self.calls += 1
            raise LLMProviderError("invalid JSON")

    agent = InvalidAgent()
    result = run_tool_rag("How can POST return 201?", agent, FakeRetriever())
    assert agent.calls == 1
    assert result["workflow_status"] == "request_failed"
    assert result["validation_retries"] == 0


def test_citation_retry_can_be_disabled():
    agent = RetryAgent()
    result = run_tool_rag_graph("How can POST return 201?", agent, FakeRetriever(), max_validation_retries=0)
    assert agent.calls == 1
    assert result["should_refuse"] is True


def test_provider_failure_on_second_attempt_keeps_prior_usage_and_retry_metadata():
    class FailSecondAttempt(RetryAgent):
        def generate_with_tools(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                search = args[1].execute("search_fastapi_docs", {"query": "status", "top_k": 1})
                return {
                    "answer": "Missing citation.",
                    "should_refuse": False,
                    "tool_calls": [{"name": "search_fastapi_docs", "result": search}],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    "provider_retry_events": [{"reason": "http_429"}],
                }
            raise LLMProviderError("gateway unavailable")

    agent = FailSecondAttempt()
    result = run_tool_rag("How can POST return 201?", agent, FakeRetriever())
    assert result["workflow_status"] == "request_failed"
    assert result["validation_retries"] == 1
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert result["provider_retry_count"] == 1
    assert result["provider_retry_events"][0]["workflow_attempt"] == 1
    assert [report["workflow_status"] for report in result["attempt_reports"]] == ["completed", "request_failed"]


def test_script_style_import_runs_graph_without_relative_import_error():
    import subprocess
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from rag_tool_demo import run_tool_rag; "
        "retriever = type('Empty', (), {'search': lambda *a, **k: []})(); "
        "assert run_tool_rag('question', object(), retriever)['workflow_status'] == 'gated_refusal'"
    )
    completed = subprocess.run([sys.executable, "-c", code, str(source)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
