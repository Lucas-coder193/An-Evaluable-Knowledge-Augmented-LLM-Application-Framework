"""Run a low-cost HTTP smoke test against a running local API service."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request_json(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 10.0) -> tuple[int, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from None
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"request failed: {type(error).__name__}") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("service returned invalid JSON") from None
    if not isinstance(body, dict):
        raise RuntimeError("service response must be a JSON object")
    return status, body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="创建资源时如何让 POST 响应状态码为 201？")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tool-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        health_status, health = _request_json(f"{base_url}/healthz", timeout=args.timeout)
        if health_status != 200 or health.get("status") != "ok":
            raise RuntimeError(f"health check failed: HTTP {health_status} {health}")
        ask_status, answer = _request_json(
            f"{base_url}/v1/ask",
            method="POST",
            payload={
                "question": args.question,
                "top_k": args.top_k,
                "max_tool_rounds": args.max_tool_rounds,
            },
            timeout=args.timeout,
        )
        if ask_status != 200:
            raise RuntimeError(f"ask request failed: HTTP {ask_status} {answer}")
        required = {
            "answer",
            "workflow_status",
            "should_refuse",
            "latency_ms",
            "workflow_engine",
            "provider_retry_count",
            "tool_error_count",
        }
        missing = sorted(required - set(answer))
        if missing:
            raise RuntimeError(f"ask response is missing fields: {', '.join(missing)}")
    except RuntimeError as error:
        print(f"SMOKE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps({"health": health, "ask": answer}, ensure_ascii=False, indent=2))
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
