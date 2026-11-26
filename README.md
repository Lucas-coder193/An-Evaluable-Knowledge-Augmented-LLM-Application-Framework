# An Evaluable Knowledge-Augmented LLM Application Framework

A FastAPI-documentation QA service. BM25 and dense retrieval feed a late-interaction reranker. A LangGraph workflow gates evidence, runs registered tools, and checks citations. Transient provider requests and citation repairs have separate retry limits.

## Run locally

From the repository root, use Python 3.12 or 3.13 and install the dependencies. These examples use PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt -r requirements-dev.txt
```

Test retrieval and run the offline suite without an API key.

```powershell
.\.venv\Scripts\python.exe -m src.retrieve_bm25 --index data/processed/fastapi_frozen_v2.jsonl --query 'FastAPI POST status_code 201' --top-k 3
.\.venv\Scripts\python.exe -m pytest -q
```

To serve live answers, enter an OpenRouter key in the current terminal and start the service.

```powershell
$env:RAG_LLM_BASE_URL = 'https://openrouter.ai/api/v1'
$env:RAG_LLM_MODEL = 'deepseek/deepseek-v3.2'
$env:RAG_LLM_API_MODE = 'chat_completions'
$key = Read-Host 'OpenRouter API key' -AsSecureString
$env:RAG_LLM_API_KEY = [System.Net.NetworkCredential]::new('', $key).Password.Trim()
.\.venv\Scripts\python.exe -m src.api_server --host 127.0.0.1 --port 8000
```

The first start downloads the embedding and reranking models. Open [the demo](http://127.0.0.1:8000/) or [Swagger UI](http://127.0.0.1:8000/docs). To send a live smoke request, run the following in a second terminal.

```powershell
.\.venv\Scripts\python.exe -m src.api_smoke --timeout 300
```

## Evaluation

The frozen index contains 35 passages from 11 Chinese summaries of official FastAPI tutorials. The [snapshot](data/snapshots/fastapi_frozen_v2.json) records the sources and chunking settings. The [evaluation inputs](data/evaluation/) cover 150 questions split by question family.

| Saved v7 result | Development | Test |
| --- | ---: | ---: |
| Reached an answer or a controlled refusal | 103/105 | 44/45 |
| Valid citation identifiers on eligible questions | 63/65 | 23/25 |
| Tool-round limit reached | 2 | 1 |

The v7 runs predate LangGraph and Chat Completions support. A valid citation identifier confirms that it came from returned evidence. The [run artifacts](reports/) also include retrieval baselines and extractive RAG results. The current HTTP workflow completed a single live document question after one network retry. Docker build and runtime validation are pending.

To evaluate BM25 on the saved test inputs, run:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_retrieval --index data/processed/fastapi_frozen_v2.jsonl --questions data/evaluation/tool_agent_test.jsonl --top-k 5
```

## Code

- [`src/api_server.py`](src/api_server.py) exposes the HTTP endpoints and the demo.
- [`src/workflow_graph.py`](src/workflow_graph.py) handles evidence gating, tool execution, and citation repair.
- [`src/llm_provider.py`](src/llm_provider.py) adapts Responses and Chat Completions and implements provider retries.
- [`src/tools.py`](src/tools.py) registers document search, calculation, date conversion, and status lookup.
