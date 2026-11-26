"""FastAPI HTTP entrypoint for the retrieval Agent."""

import argparse
from pathlib import Path
from typing import Any

if __package__:
    from .api_service import APIRequestError, RAGService, build_default_service
else:
    from api_service import APIRequestError, RAGService, build_default_service


def create_app(service: RAGService | None = None) -> Any:
    """Create the HTTP app without loading models until the app is requested."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError as error:
        raise RuntimeError("Install requirements-api.txt before starting the HTTP service") from error

    class AskRequest(BaseModel):
        question: str = Field(min_length=1, max_length=2_000)
        top_k: int = Field(default=3, ge=1, le=5)
        max_tool_rounds: int = Field(default=2, ge=1, le=5)

    class AskResponse(BaseModel):
        answer: str
        should_refuse: bool
        workflow_status: str
        workflow_engine: str = "langgraph"
        workflow_trace: list[str] = Field(default_factory=list)
        validation_retries: int = 0
        provider_retry_count: int = 0
        provider_retry_events: list[dict[str, Any]] = Field(default_factory=list)
        attempt_reports: list[dict[str, Any]] = Field(default_factory=list)
        tool_error_count: int = 0
        citation_check: dict[str, Any] | None = None
        code_block_check: dict[str, Any] | None = None
        evidence_gate: dict[str, Any] | None = None
        retrieved_evidence_ids: list[str]
        usage: dict[str, int] | None = None
        latency_ms: float

    app = FastAPI(title="Grounded Retrieval Agent", version="0.1.0")
    app.state.service = service

    @app.get("/", include_in_schema=False)
    def homepage() -> FileResponse:
        return FileResponse(Path(__file__).with_name("web") / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        if app.state.service is None:
            return {"status": "starting"}
        return app.state.service.health()

    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        responses={
            422: {"description": "Invalid request parameters"},
            502: {"description": "Model provider request failed"},
            504: {"description": "Agent tool workflow did not complete"},
        },
    )
    def ask(request: AskRequest) -> AskResponse:
        if app.state.service is None:
            raise HTTPException(status_code=503, detail="service is not initialized")
        try:
            result = app.state.service.ask(
                request.question,
                top_k=request.top_k,
                max_tool_rounds=request.max_tool_rounds,
            )
        except APIRequestError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        status = result.get("workflow_status")
        if status in {"request_failed", "workflow_incomplete"}:
            # Preserve the existing detail string while also exposing bounded
            # reliability counters when a request exhausts its retry budget.
            return JSONResponse(
                status_code=502 if status == "request_failed" else 504,
                content={
                    "detail": result.get("error", "workflow did not complete"),
                    **{
                        field: result.get(field)
                        for field in (
                            "workflow_status", "workflow_trace", "validation_retries",
                            "provider_retry_count", "provider_retry_events", "attempt_reports",
                            "tool_error_count", "usage",
                        )
                    },
                },
            )
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install requirements-api.txt before starting the HTTP service") from error
    uvicorn.run(create_app(build_default_service()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
