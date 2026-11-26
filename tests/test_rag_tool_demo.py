from src.rag_tool_demo import check_tool_citations, run_tool_rag


def make_result(answer):
    return {
        "answer": answer,
        "should_refuse": False,
        "tool_calls": [
            {
                "name": "search_fastapi_docs",
                "result": {
                    "results": [
                        {"label": "E1", "evidence_id": "response-status-code#设置响应状态码"},
                    ]
                },
            }
        ],
    }


def test_citation_check_maps_result_labels_to_canonical_evidence_ids():
    result = check_tool_citations(make_result("设置 status_code=201。【E1】"))

    assert result["citation_check"]["supported"] is True
    assert result["citation_check"]["cited_labels"] == ["E1"]
    assert result["citation_check"]["cited_evidence_ids"] == ["response-status-code#设置响应状态码"]
    assert result["should_refuse"] is False


def test_citation_check_also_accepts_a_real_canonical_evidence_id():
    evidence_id = "response-status-code#设置响应状态码"
    result = check_tool_citations(make_result(f"设置 status_code=201。【{evidence_id}】"))

    assert result["citation_check"]["supported"] is True
    assert result["citation_check"]["cited_evidence_ids"] == [evidence_id]


def test_citation_check_fails_closed_when_citation_is_missing_or_unknown():
    missing = check_tool_citations(make_result("设置 status_code=201。"))
    unknown = check_tool_citations(make_result("设置 status_code=201。【made-up#source】"))

    assert missing["citation_check"]["supported"] is False
    assert missing["answer"] == "证据不足，无法回答。"
    assert unknown["citation_check"]["unsupported_citations"] == ["made-up#source"]
    assert unknown["should_refuse"] is True


def test_citation_check_rejects_literal_schema_placeholder():
    result = check_tool_citations(make_result("答案。【evidence_id】"))

    assert result["citation_check"]["supported"] is False
    assert result["citation_check"]["unsupported_citations"] == ["evidence_id"]
    assert result["should_refuse"] is True


def test_citation_check_removes_code_block_not_present_in_retrieved_evidence():
    result = make_result(
        "装饰器可以设置状态码【E1】。示例：\n```python\n"
        'from fastapi import FastAPI\n\n@app.post("/items/", status_code=201)\n'
        'async def create_item():\n    return {"message": "created"}\n```'
    )
    result["tool_calls"][0]["result"]["results"][0]["text"] = (
        '可以设置 `@app.post("/items/", status_code=201)`。'
    )

    checked = check_tool_citations(result)

    assert checked["citation_check"]["supported"] is True
    assert checked["code_block_check"] == {
        "code_blocks_checked": 1,
        "unsupported_code_blocks": 1,
        "answer_modified": True,
    }
    assert "```" not in checked["answer"]
    assert "async def create_item" not in checked["answer"]
    assert "装饰器可以设置状态码" in checked["answer"]
    assert checked["raw_answer"].startswith("装饰器可以设置状态码")


def test_citation_check_removes_code_block_intro_with_the_block():
    result = make_result(
        "可以设置状态码【E1】。最小示例：\n```python\n"
        '@app.post("/items/", status_code=201)\n```'
    )

    checked = check_tool_citations(result)

    assert checked["answer"] == "可以设置状态码【E1】。"
    assert "最小" not in checked["answer"]


def test_citation_check_keeps_code_block_fully_present_in_evidence():
    code = 'from fastapi import status\n@app.post("/items/", status_code=status.HTTP_201_CREATED)'
    result = make_result(f"可以这样写【E1】：\n```python\n{code}\n```")
    result["tool_calls"][0]["result"]["results"][0]["text"] = f"示例：\n{code}"

    checked = check_tool_citations(result)

    assert checked["code_block_check"]["unsupported_code_blocks"] == 0
    assert checked["code_block_check"]["answer_modified"] is False
    assert "status.HTTP_201_CREATED" in checked["answer"]


def test_tool_rag_gates_out_of_scope_question_without_calling_model():
    class Retriever:
        def search(self, query, top_k=5):
            return []

    class Agent:
        def generate_with_tools(self, *_args, **_kwargs):
            raise AssertionError("the model should not be called after a no-result gate")

    result = run_tool_rag("out of scope", Agent(), Retriever())

    assert result["workflow_status"] == "gated_refusal"
    assert result["should_refuse"] is True
    assert result["evidence_gate"]["reason"] == "no_retrieval_results"


def test_tool_rag_gates_irrelevant_dense_results_without_calling_model():
    class Retriever:
        def search(self, query, top_k=5):
            return [
                {
                    "evidence_id": "first-steps#api-docs",
                    "text": "FastAPI 会自动生成 API 文档。",
                }
            ]

    class Agent:
        def generate_with_tools(self, *_args, **_kwargs):
            raise AssertionError("the model should not be called when retrieved evidence is unrelated")

    question = "FastAPI 官方推荐哪家云厂商托管 PostgreSQL？"
    result = run_tool_rag(question, Agent(), Retriever())

    assert result["workflow_status"] == "gated_refusal"
    assert result["should_refuse"] is True
    assert result["evidence_gate"]["reason"] == "insufficient_question_term_support"


def test_rag_registry_exposes_retrieval_and_bounded_local_tools():
    from src.tools import rag_tool_registry

    class Retriever:
        def search(self, query, top_k=5):
            return []

    registry = rag_tool_registry(Retriever())
    assert {schema["name"] for schema in registry.schemas()} == {
        "search_fastapi_docs", "calculator", "date_conversion", "structured_lookup"
    }


def test_combined_workflow_requires_retrieval_as_the_first_tool():
    class Retriever:
        def search(self, query, top_k=5):
            return [{
                "evidence_id": "guide#status",
                "text": "创建资源通常使用 201。",
            }]

    class Agent:
        def __init__(self):
            self.required_first_tool = None

        def generate_with_tools(self, question, registry, *, max_tool_rounds=2, required_first_tool=None):
            self.required_first_tool = required_first_tool
            result = registry.execute("search_fastapi_docs", {"query": question, "top_k": 1})
            return {
                "answer": "创建资源通常使用 201。【E1】",
                "should_refuse": False,
                "tool_calls": [{"name": "search_fastapi_docs", "result": result}],
            }

    agent = Agent()
    result = run_tool_rag(
        "检索创建资源的状态码",
        agent,
        Retriever(),
        task_type="combined",
    )

    assert agent.required_first_tool == "search_fastapi_docs"
    assert result["citation_check"]["supported"] is True
