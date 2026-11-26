import json

import pytest

from src.llm_provider import (
    LLMProviderError,
    OpenAIResponsesGenerator,
    TransientLLMProviderError,
    _urllib_post_json,
)
from src.tools import ToolRegistry, default_tool_registry, retrieval_tool


def response_with_text(text: str) -> dict:
    return {
        "id": "resp_test",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def test_responses_generator_builds_strict_schema_request_and_parses_output():
    calls = []

    def fake_transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return response_with_text(
            json.dumps(
                {"answer": "路径参数放在 URL 中。", "citations": ["E1"], "should_refuse": False},
                ensure_ascii=False,
            )
        )

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        base_url="https://api.example.test/v1/",
        timeout_seconds=9,
        transport=fake_transport,
    )

    result = generator.generate(
        "路径参数是什么？",
        [{"label": "E1", "text": "路径参数出现在 URL 路径中。", "source_url": "https://example.test"}],
    )

    assert result == {"answer": "路径参数放在 URL 中。", "citations": ["E1"], "should_refuse": False}
    url, headers, payload, timeout = calls[0]
    assert url == "https://api.example.test/v1/responses"
    assert headers["Authorization"] == "Bearer test-secret"
    assert payload["model"] == "test-model"
    assert payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    assert payload["input"][0]["role"] == "system"
    assert "未受信任的数据" in payload["input"][0]["content"]
    assert "[E1]" in payload["input"][1]["content"]
    assert timeout == 9


def test_responses_generator_preserves_standard_usage_counters_only():
    response = response_with_text(
        json.dumps({"answer": "依据证据作答。", "citations": ["E1"], "should_refuse": False})
    )
    response["usage"] = {
        "input_tokens": 123,
        "output_tokens": 45,
        "total_tokens": 168,
        "provider_debug": "ignored",
    }
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: response,
    )

    result = generator.generate("question", [{"label": "E1", "text": "evidence"}])

    assert result["usage"] == {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}


def test_responses_generator_surfaces_provider_refusal_as_structured_refusal():
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
                }
            ],
        },
    )

    assert generator.generate("question", []) == {
        "answer": "I cannot help with that.",
        "citations": [],
        "should_refuse": True,
    }


@pytest.mark.parametrize(
    "response, message",
    [
        ({"status": "incomplete", "output": []}, "did not complete"),
        (response_with_text("not json"), "not valid JSON"),
        (response_with_text('{"answer":"x","citations":[]}'), "required fields"),
        (response_with_text('{"answer":"x","citations":[],"should_refuse":"false"}'), "boolean"),
    ],
)
def test_responses_generator_rejects_incomplete_or_invalid_output(response, message):
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: response,
    )

    with pytest.raises(LLMProviderError, match=message):
        generator.generate("question", [])


def test_responses_generator_requires_https_style_base_url_without_embedded_credentials():
    with pytest.raises(ValueError, match="absolute HTTP"):
        OpenAIResponsesGenerator(api_key="key", model="model", base_url="not-a-url")
    with pytest.raises(ValueError, match="must not contain credentials"):
        OpenAIResponsesGenerator(api_key="key", model="model", base_url="https://user:pass@example.test/v1")


def test_generator_reads_configuration_from_environment_without_printing_secrets(monkeypatch):
    monkeypatch.delenv("RAG_LLM_API_MODE", raising=False)
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_LLM_MODEL", "configured-model")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://gateway.example.test/v1")

    generator = OpenAIResponsesGenerator.from_env()

    assert generator.model == "configured-model"
    assert generator.endpoint == "https://gateway.example.test/v1/responses"
    assert "test-secret" not in repr(generator)


def test_generator_reads_chat_completions_mode_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("RAG_LLM_MODEL", "configured-model")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("RAG_LLM_API_MODE", "chat_completions")

    generator = OpenAIResponsesGenerator.from_env()

    assert generator.api_mode == "chat_completions"
    assert generator.endpoint == "https://openrouter.ai/api/v1/chat/completions"
    assert generator.name == "openai-chat_completions:configured-model"


def test_chat_completions_adapter_builds_json_mode_request_and_normalizes_usage():
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, payload))
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"answer": "可以返回 201。", "citations": ["E1"], "should_refuse": False}),
                },
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        transport=transport,
    )
    result = generator.generate("question", [{"label": "E1", "text": "evidence"}])

    assert result["answer"] == "可以返回 201。"
    assert result["usage"] == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    assert calls[0][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0][1]["response_format"] == {"type": "json_object"}
    assert calls[0][1]["max_tokens"] == 512
    assert calls[0][1]["messages"][1]["role"] == "user"


def test_chat_completions_tool_calls_round_trip_and_preserve_call_id():
    payloads = []
    reasoning_details = [{"type": "reasoning.text", "text": "Use the calculator."}]
    responses = iter([
        {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "先计算。",
                    "reasoning_details": reasoning_details,
                    "tool_calls": [{
                        "id": "call_calc_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"expression\":\"2 + 2\"}"},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
        {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "结果是 4。"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
    ])

    def transport(url, headers, payload, timeout):
        payloads.append(payload)
        return next(responses)

    generator = OpenAIResponsesGenerator(
        api_key="test-secret", model="test-model", api_mode="chat_completions", transport=transport
    )
    result = generator.generate_with_tools("计算 2 + 2", default_tool_registry())

    assert result["answer"] == "结果是 4。"
    assert result["usage"] == {"input_tokens": 30, "output_tokens": 9, "total_tokens": 39}
    assert result["tool_calls"][0]["call_id"] == "call_calc_1"
    assert result["tool_calls"][0]["result"]["value"] == 4
    assert payloads[0]["tools"][0]["type"] == "function"
    assert payloads[0]["tools"][0]["function"]["name"] == "calculator"
    assert payloads[0]["tool_choice"] == "auto"
    assert payloads[1]["messages"][-2]["tool_calls"][0]["id"] == "call_calc_1"
    assert payloads[1]["messages"][-2]["content"] == "先计算。"
    assert payloads[1]["messages"][-2]["reasoning_details"] == reasoning_details
    assert payloads[1]["messages"][-1]["tool_call_id"] == "call_calc_1"
    assert json.loads(payloads[1]["messages"][-1]["content"])["result"]["value"] == 4


def test_chat_completions_honors_required_first_tool():
    payloads = []
    responses = iter([
        {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [{
            "id": "search-1", "type": "function",
            "function": {"name": "search_fastapi_docs", "arguments": "{\"query\":\"201\"}"},
        }]}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "答案。【E1】"}}]},
    ])

    class Retriever:
        def search(self, query, top_k=5):
            return [{"evidence_id": "guide#201", "text": "201 means created"}]

    from src.tools import rag_tool_registry

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        api_mode="chat_completions",
        transport=lambda _url, _headers, payload, _timeout: payloads.append(payload) or next(responses),
    )
    generator.generate_with_tools("lookup 201", rag_tool_registry(Retriever()), required_first_tool="search_fastapi_docs")

    assert payloads[0]["tool_choice"] == {"type": "function", "function": {"name": "search_fastapi_docs"}}


def test_transient_provider_error_retries_with_bounded_backoff_and_reports_it():
    calls = []
    sleeps = []

    def flaky_transport(*_args):
        calls.append(1)
        if len(calls) == 1:
            raise TransientLLMProviderError("temporary", status_code=429, retry_after=90)
        return response_with_text(
            json.dumps({"answer": "可以返回 201。", "citations": ["E1"], "should_refuse": False})
        )

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=flaky_transport,
        max_request_retries=2,
        retry_backoff_seconds=0.1,
        max_retry_after_seconds=2,
        sleeper=sleeps.append,
    )

    result = generator.generate("question", [{"label": "E1", "text": "evidence"}])

    assert len(calls) == 2
    assert sleeps == [2]
    assert result["provider_retry_count"] == 1
    assert result["provider_retry_events"] == [
        {"attempt": 1, "reason": "http_429", "delay_seconds": 2, "status_code": 429}
    ]


def test_non_retryable_provider_error_is_not_repeated():
    calls = []

    def unauthorized(*_args):
        calls.append(1)
        raise LLMProviderError("LLM API returned HTTP 401")

    generator = OpenAIResponsesGenerator(
        api_key="test-secret", model="test-model", transport=unauthorized, sleeper=lambda _delay: None
    )
    with pytest.raises(LLMProviderError, match="401"):
        generator.generate("question", [])
    assert len(calls) == 1


def test_transient_then_permanent_failure_preserves_retry_history():
    failures = iter([TransientLLMProviderError('temporary', status_code=503), LLMProviderError('HTTP 401')])
    sleeps = []

    def transport(*args):
        raise next(failures)

    generator = OpenAIResponsesGenerator(
        api_key='test-secret', model='test-model', transport=transport, sleeper=sleeps.append
    )
    result = generator.generate_with_tools('calculate', default_tool_registry())
    assert result['workflow_status'] == 'request_failed'
    assert result['provider_retry_count'] == 1
    assert result['provider_retry_events'][0]['reason'] == 'http_503'
    assert sleeps == [0.5]


def test_http_transport_marks_only_retryable_status_codes_as_transient(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    def rate_limited(*_args, **_kwargs):
        raise HTTPError(
            "https://example.test/responses",
            429,
            "rate limited",
            {"Retry-After": "3"},
            BytesIO(b'{"error":{"message":"slow down"}}'),
        )

    monkeypatch.setattr("src.llm_provider.urlopen", rate_limited)
    with pytest.raises(TransientLLMProviderError) as raised:
        _urllib_post_json("https://example.test/responses", {}, {}, 1)
    assert raised.value.status_code == 429
    assert raised.value.retry_after == 3


def test_retry_exhaustion_is_reported_as_controlled_tool_workflow_failure():
    calls = []

    def always_unavailable(*_args):
        calls.append(1)
        raise TransientLLMProviderError("upstream unavailable", status_code=503)

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=always_unavailable,
        max_request_retries=2,
        retry_backoff_seconds=0,
        sleeper=lambda _delay: None,
    )
    result = generator.generate_with_tools("compute", default_tool_registry())

    assert len(calls) == 3
    assert result["workflow_status"] == "request_failed"
    assert result["provider_retry_count"] == 2
    assert [event["reason"] for event in result["provider_retry_events"]] == ["http_503", "http_503"]


def test_generator_fails_fast_when_environment_configuration_is_missing(monkeypatch):
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)

    with pytest.raises(LLMProviderError, match="Set RAG_LLM_API_KEY"):
        OpenAIResponsesGenerator.from_env()


def test_tool_call_loop_executes_registered_tool_and_returns_final_text():
    calls = []

    def fake_transport(url, headers, payload, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"expression":"(20 + 10) / 3"}',
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            }
        assert payload["input"][-1]["type"] == "function_call_output"
        assert '"value": 10.0' in payload["input"][-1]["output"]
        return response_with_text("结果是 10。") | {
            "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}
        }

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=fake_transport,
    )
    result = generator.generate_with_tools("计算 (20 + 10) / 3", default_tool_registry())

    assert result["answer"] == "结果是 10。"
    assert result["should_refuse"] is False
    assert result["tool_calls"][0]["name"] == "calculator"
    assert result["tool_calls"][0]["result"] == {"expression": "(20 + 10) / 3", "value": 10.0}
    assert result["usage"] == {"input_tokens": 50, "output_tokens": 13, "total_tokens": 63}
    assert calls[0]["tools"][0]["name"] == "calculator"


def test_tool_call_loop_can_require_a_specific_first_tool():
    payloads = []
    responses = iter(
        [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "search_1",
                        "name": "search_fastapi_docs",
                        "arguments": '{"query":"状态码"}',
                    }
                ],
            },
            response_with_text("证据支持。"),
        ]
    )

    class Retriever:
        def search(self, query, top_k=5):
            return [{"evidence_id": "guide#one", "text": "201 表示已创建。"}]

    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda _url, _headers, payload, _timeout: payloads.append(payload) or next(responses),
    )
    from src.tools import rag_tool_registry

    generator.generate_with_tools(
        "检索状态码",
        rag_tool_registry(Retriever()),
        required_first_tool="search_fastapi_docs",
    )

    assert payloads[0]["tool_choice"] == {"type": "function", "name": "search_fastapi_docs"}
    assert payloads[1]["tool_choice"] == "auto"


def test_tool_call_loop_returns_controlled_error_for_invalid_arguments():
    responses = iter(
        [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "calculator",
                        "arguments": '{"expression":"open(\\"secret\\")"}',
                    }
                ],
            },
            response_with_text("无法完成计算。"),
        ]
    )
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: next(responses),
    )

    result = generator.generate_with_tools("计算这个表达式", default_tool_registry())

    assert result["answer"] == "无法完成计算。"
    assert result["tool_calls"][0]["error"]


def test_tool_error_forces_refusal_even_when_model_omits_the_flag():
    responses = iter(
        [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "calculator",
                        "arguments": '{"expression":"2 / 0"}',
                    }
                ],
            },
            response_with_text("结果为 0。"),
        ]
    )
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: next(responses),
    )

    result = generator.generate_with_tools("计算 2 / 0", default_tool_registry())

    assert result["should_refuse"] is True


def test_tool_call_loop_returns_controlled_error_for_unknown_tool():
    responses = iter(
        [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_unknown",
                        "name": "delete_everything",
                        "arguments": "{}",
                    }
                ],
            },
            response_with_text("我不能调用未注册的工具。"),
        ]
    )
    sent_inputs = []
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda _url, _headers, payload, _timeout: sent_inputs.append(payload["input"]) or next(responses),
    )

    result = generator.generate_with_tools("删除所有文件", default_tool_registry())

    assert "unknown tool" in result["tool_calls"][0]["error"]
    tool_output = sent_inputs[1][-1]
    assert tool_output["type"] == "function_call_output"
    assert '"ok": false' in tool_output["output"]


def test_tool_call_loop_includes_retrieval_tool_contract_and_round_trip_result():
    class Retriever:
        def search(self, query, top_k=5):
            return [
                {
                    "evidence_id": "guide#one",
                    "source_title": "Guide",
                    "source_url": "https://example.test",
                    "section_title": "Response status codes",
                    "text": "POST 可以返回 201。",
                    "score": 1.0,
                    "rerank_score": 0.9,
                }
            ][:top_k]

    responses = iter(
        [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "search_1",
                        "name": "search_fastapi_docs",
                        "arguments": '{"query":"POST 状态码","top_k":1}',
                    }
                ],
            },
            response_with_text("可以返回 201。【guide#one】"),
        ]
    )
    payloads = []
    generator = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda _url, _headers, payload, _timeout: payloads.append(payload) or next(responses),
    )

    result = generator.generate_with_tools(
        "创建资源时如何返回 201？",
        ToolRegistry([retrieval_tool(Retriever())]),
    )

    assert result["answer"] == "可以返回 201。【guide#one】"
    assert result["tool_calls"][0]["result"]["results"][0]["evidence_id"] == "guide#one"
    assert result["tool_calls"][0]["result"]["results"][0]["source_url"] == "https://example.test"
    assert payloads[0]["tools"][0]["name"] == "search_fastapi_docs"
    assert "必须先调用 search_fastapi_docs" in payloads[0]["input"][0]["content"]
    assert "【E1】" in payloads[0]["input"][0]["content"]
    assert "不要输出【evidence_id】" in payloads[0]["input"][0]["content"]
    assert "2 到 4 句话" in payloads[0]["input"][0]["content"]
    assert "证据没有给出完整代码或必要导入时，只作概念说明" in payloads[0]["input"][0]["content"]
    returned_to_model = json.loads(payloads[1]["input"][-1]["output"])
    assert returned_to_model["result"]["results"] == [
        {
            "label": "E1",
            "evidence_id": "guide#one",
            "section_title": "Response status codes",
            "text": "POST 可以返回 201。",
        }
    ]


def test_tool_call_loop_rejects_malformed_json_and_enforces_round_limit():
    malformed = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "calculator",
                    "arguments": "not-json",
                }
            ],
        },
    )
    with pytest.raises(LLMProviderError, match="arguments were not valid JSON"):
        malformed.generate_with_tools("计算", default_tool_registry())

    looping = OpenAIResponsesGenerator(
        api_key="test-secret",
        model="test-model",
        transport=lambda *_: {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_loop",
                    "name": "calculator",
                    "arguments": '{"expression":"1 + 1"}',
                }
            ],
        },
    )
    result = looping.generate_with_tools("一直计算", default_tool_registry(), max_tool_rounds=2)
    assert result["workflow_status"] == "workflow_incomplete"
    assert result["error"] == "tool-call loop exceeded max_tool_rounds=2"
    assert result["tool_calls"][-1]["executed"] is False


def test_tool_call_loop_preserves_partial_usage_on_provider_error():
    responses = iter(
        [
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"expression":"2 + 2"}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        ]
    )

    def fake_transport(*_):
        try:
            return next(responses)
        except StopIteration:
            raise LLMProviderError("provider unavailable")

    generator = OpenAIResponsesGenerator(api_key="test-secret", model="test-model", transport=fake_transport)
    result = generator.generate_with_tools("计算 2 + 2", default_tool_registry())

    assert result["workflow_status"] == "request_failed"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert len(result["tool_calls"]) == 1
