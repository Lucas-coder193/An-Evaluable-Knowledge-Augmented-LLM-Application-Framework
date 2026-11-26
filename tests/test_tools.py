import pytest
from src.tools import ToolDefinition


def test_unexpected_tool_exception_is_sanitized_and_executed_only_once():
    calls = []

    def broken_handler(arguments):
        calls.append(arguments)
        raise RuntimeError('private backend details')

    tool = ToolDefinition('broken', 'test', {}, broken_handler)
    with pytest.raises(ToolExecutionError) as caught:
        tool.execute({})
    assert str(caught.value) == 'tool execution failed: RuntimeError'
    assert calls == [{}]

from src.tools import (
    ToolExecutionError,
    ToolRegistry,
    calculator_tool,
    date_conversion_tool,
    default_tool_registry,
    local_tool_registry,
    retrieval_tool,
    structured_lookup_tool,
)


def test_calculator_tool_returns_a_json_ready_result():
    result = calculator_tool().execute({"expression": "(20 + 10) / 3"})

    assert result == {"expression": "(20 + 10) / 3", "value": 10.0}


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "2 < 3", "x + 1"])
def test_calculator_rejects_code_names_and_comparisons(expression):
    with pytest.raises(ToolExecutionError):
        calculator_tool().execute({"expression": expression})


def test_calculator_rejects_large_exponents_and_bad_arguments():
    with pytest.raises(ToolExecutionError, match="exponent"):
        calculator_tool().execute({"expression": "2 ** 101"})
    with pytest.raises(ToolExecutionError, match="JSON object"):
        calculator_tool().execute(["1 + 1"])


def test_registry_exposes_explicit_responses_schema_and_rejects_unknown_tools():
    registry = default_tool_registry()
    schema = registry.schemas()[0]

    assert schema["type"] == "function"
    assert schema["name"] == "calculator"
    assert schema["parameters"]["additionalProperties"] is False
    with pytest.raises(ToolExecutionError, match="unknown tool"):
        registry.execute("not_registered", {})


def test_local_registry_contains_bounded_non_retrieval_tools():
    registry = local_tool_registry()
    assert {schema["name"] for schema in registry.schemas()} == {
        "calculator", "date_conversion", "structured_lookup"
    }
    assert date_conversion_tool().execute({"date": "2026-10-26"})["weekday_number"] == 1
    assert date_conversion_tool().execute({"date": "1970-01-01", "operation": "unix_timestamp"})["unix_timestamp"] == 0
    assert structured_lookup_tool().execute({"status_code": "201"})["name"] == "Created"


def test_bounded_tools_reject_invalid_arguments():
    with pytest.raises(ToolExecutionError, match="YYYY-MM-DD"):
        date_conversion_tool().execute({"date": "26/10/2026"})
    with pytest.raises(ToolExecutionError, match="supports status codes"):
        structured_lookup_tool().execute({"status_code": "500"})


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([calculator_tool()])
    with pytest.raises(ValueError, match="duplicate tool"):
        registry.register(calculator_tool())


class FakeRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.rows[:top_k]


def test_retrieval_tool_returns_bounded_provenance_and_calls_retriever():
    retriever = FakeRetriever(
        [
            {
                "chunk_id": "chunk-1",
                "evidence_id": "guide#one",
                "source_title": "Guide",
                "source_url": "https://example.test/guide",
                "section_title": "One",
                "text": "Answer text",
                "score": 0.75,
                "rerank_score": 0.8,
                "internal": object(),
            }
        ]
    )
    tool = retrieval_tool(retriever, max_top_k=3)

    result = tool.execute({"query": "how", "top_k": 1})

    assert retriever.calls == [("how", 1)]
    assert result == {
        "query": "how",
        "results": [
            {
                "label": "E1",
                "evidence_id": "guide#one",
                "source_title": "Guide",
                "source_url": "https://example.test/guide",
                "section_title": "One",
                "text": "Answer text",
                "score": 0.75,
                "rerank_score": 0.8,
            }
        ],
    }


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"query": ""}, "non-empty query"),
        ({"query": "query", "top_k": True}, "top_k must be an integer"),
        ({"query": "query", "top_k": 4}, "between 1 and 3"),
        ({"query": "query", "unexpected": 1}, "unknown arguments"),
    ],
)
def test_retrieval_tool_rejects_invalid_arguments(arguments, message):
    tool = retrieval_tool(FakeRetriever([]), max_top_k=3)

    with pytest.raises(ToolExecutionError, match=message):
        tool.execute(arguments)
