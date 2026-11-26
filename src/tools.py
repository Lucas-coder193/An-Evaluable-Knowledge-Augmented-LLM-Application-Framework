"""Small, validated local tools for the workflow layer.

Tools are deliberately ordinary Python functions behind a registry.  The
model may request a named tool later, but it never receives permission to
execute arbitrary Python or choose a callable by string interpolation.
"""

from __future__ import annotations

import ast
import datetime as dt
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable, Iterable


class ToolExecutionError(ValueError):
    """Raised when a tool name or its arguments are invalid."""


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    """A tool contract exposed to a model and executable by the application."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_responses_tool(self) -> dict[str, Any]:
        """Render the OpenAI Responses function-tool shape."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def execute(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ToolExecutionError("tool arguments must be a JSON object")
        try:
            result = self.handler(arguments)
        except ToolExecutionError:
            raise
        except Exception as error:
            raise ToolExecutionError(f"tool execution failed: {type(error).__name__}") from None
        if not isinstance(result, dict):
            raise ToolExecutionError("tool result must be a JSON object")
        return result


class ToolRegistry:
    """Name-to-contract registry with explicit duplicate and unknown checks."""

    def __init__(self, definitions: Iterable[ToolDefinition] = ()):
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if not definition.name or not definition.name.strip():
            raise ValueError("tool name must not be empty")
        if definition.name in self._definitions:
            raise ValueError(f"duplicate tool: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError:
            raise ToolExecutionError(f"unknown tool: {name}") from None

    def schemas(self) -> list[dict[str, Any]]:
        return [definition.as_responses_tool() for definition in self._definitions.values()]

    def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        return self.get(name).execute(arguments)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _checked_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolExecutionError("calculator accepts numbers only")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolExecutionError("calculator result must be finite")
    if isinstance(value, int) and len(str(abs(value))) > 100:
        raise ToolExecutionError("calculator integer is too large")
    return value


def _evaluate_expression(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        return _checked_number(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _checked_number(_UNARY_OPERATORS[type(node.op)](_evaluate_expression(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
            raise ToolExecutionError("calculator exponent is limited to 100")
        return _checked_number(_BINARY_OPERATORS[type(node.op)](left, right))
    raise ToolExecutionError("calculator supports numbers, parentheses, and arithmetic operators only")


def _calculate(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolExecutionError("calculator requires a non-empty expression")
    expression = expression.strip()
    if len(expression) > 200:
        raise ToolExecutionError("calculator expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate_expression(tree.body)
    except SyntaxError:
        raise ToolExecutionError("calculator expression is invalid") from None
    return {"expression": expression, "value": value}


def calculator_tool() -> ToolDefinition:
    """Return the safe arithmetic tool used by the first workflow milestone."""
    return ToolDefinition(
        name="calculator",
        description="计算一个只包含数字、括号和基本算术运算符的表达式。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "例如 (20 + 10) / 3",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=_calculate,
    )


def date_conversion_tool() -> ToolDefinition:
    """Return a deterministic UTC date conversion tool."""

    def _convert(arguments: dict[str, Any]) -> dict[str, Any]:
        date_value = arguments.get("date")
        operation = arguments.get("operation", "weekday")
        if not isinstance(date_value, str) or not date_value.strip():
            raise ToolExecutionError("date_conversion requires a non-empty ISO date")
        if operation not in {"weekday", "unix_timestamp"}:
            raise ToolExecutionError("date_conversion operation must be weekday or unix_timestamp")
        try:
            parsed = dt.date.fromisoformat(date_value.strip())
        except ValueError:
            raise ToolExecutionError("date_conversion requires a date in YYYY-MM-DD format") from None
        result: dict[str, Any] = {"date": parsed.isoformat(), "operation": operation}
        if operation == "weekday":
            result["weekday"] = parsed.strftime("%A")
            result["weekday_number"] = parsed.isoweekday()
        else:
            result["unix_timestamp"] = int(
                dt.datetime.combine(parsed, dt.time(), tzinfo=dt.timezone.utc).timestamp()
            )
        return result

    return ToolDefinition(
        name="date_conversion",
        description="把 YYYY-MM-DD 日期转换为星期或 UTC Unix 时间戳。",
        parameters={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD，例如 2026-10-26"},
                "operation": {
                    "type": "string",
                    "enum": ["weekday", "unix_timestamp"],
                    "description": "转换类型，默认 weekday",
                },
            },
            "required": ["date"],
            "additionalProperties": False,
        },
        handler=_convert,
    )


_HTTP_STATUS_LOOKUP: dict[str, dict[str, Any]] = {
    "201": {"code": 201, "name": "Created", "description": "请求成功并创建了资源。"},
    "204": {"code": 204, "name": "No Content", "description": "请求成功，但响应不应包含消息体。"},
    "404": {"code": 404, "name": "Not Found", "description": "服务器找不到请求的资源。"},
    "422": {"code": 422, "name": "Unprocessable Content", "description": "请求格式正确但语义校验未通过。"},
}


def structured_lookup_tool() -> ToolDefinition:
    """Return a bounded lookup over a small, versioned status-code table."""

    def _lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        key = arguments.get("status_code")
        if isinstance(key, int) and not isinstance(key, bool):
            key = str(key)
        if not isinstance(key, str) or key.strip() not in _HTTP_STATUS_LOOKUP:
            raise ToolExecutionError("structured_lookup supports status codes 201, 204, 404, and 422")
        return dict(_HTTP_STATUS_LOOKUP[key.strip()])

    return ToolDefinition(
        name="structured_lookup",
        description="查询受限 HTTP 状态码表，返回结构化名称和说明。",
        parameters={
            "type": "object",
            "properties": {
                "status_code": {
                    "type": "string",
                    "enum": sorted(_HTTP_STATUS_LOOKUP),
                    "description": "要查询的 HTTP 状态码",
                }
            },
            "required": ["status_code"],
            "additionalProperties": False,
        },
        handler=_lookup,
    )


def default_tool_registry() -> ToolRegistry:
    """Keep the standalone calculator demo intentionally single-purpose."""
    return ToolRegistry([calculator_tool()])


def local_tool_registry() -> ToolRegistry:
    """Return the local deterministic tools used by the application workflow."""
    return ToolRegistry([calculator_tool(), date_conversion_tool(), structured_lookup_tool()])


def retrieval_tool(
    retriever: Any,
    *,
    name: str = "search_fastapi_docs",
    max_top_k: int = 5,
    max_query_length: int = 500,
    max_text_length: int = 1_200,
) -> ToolDefinition:
    """Expose a read-only retriever through a bounded JSON tool contract.

    Only provenance fields needed by the model are returned.  Raw retriever
    rows are deliberately not passed through because they may contain
    non-JSON values or implementation details.
    """
    if not name.strip():
        raise ValueError("retrieval tool name must not be empty")
    if max_top_k < 1 or max_top_k > 20:
        raise ValueError("max_top_k must be between 1 and 20")
    if max_query_length < 1 or max_text_length < 1:
        raise ValueError("retrieval output limits must be positive")
    if not hasattr(retriever, "search") or not callable(retriever.search):
        raise TypeError("retriever must provide a callable search method")

    def _search(arguments: dict[str, Any]) -> dict[str, Any]:
        unknown = set(arguments) - {"query", "top_k"}
        if unknown:
            raise ToolExecutionError(f"search tool received unknown arguments: {sorted(unknown)}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolExecutionError("search tool requires a non-empty query")
        query = query.strip()
        if len(query) > max_query_length:
            raise ToolExecutionError(f"search query exceeds {max_query_length} characters")
        top_k = arguments.get("top_k", min(3, max_top_k))
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ToolExecutionError("search top_k must be an integer")
        if not 1 <= top_k <= max_top_k:
            raise ToolExecutionError(f"search top_k must be between 1 and {max_top_k}")
        try:
            rows = retriever.search(query, top_k=top_k)
        except Exception as error:
            raise ToolExecutionError(f"search execution failed: {type(error).__name__}") from None
        if not isinstance(rows, list):
            raise ToolExecutionError("search result must be a list")

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows[:top_k], start=1):
            if not isinstance(row, dict):
                raise ToolExecutionError("search result rows must be JSON objects")
            evidence_id = row.get("evidence_id") or row.get("chunk_id")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise ToolExecutionError("search result is missing evidence_id")
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            result: dict[str, Any] = {
                "label": f"E{rank}",
                "evidence_id": evidence_id,
                "source_title": str(row.get("source_title", "")),
                "source_url": str(row.get("source_url", "")),
                "section_title": str(row.get("section_title", "")),
                "text": text[:max_text_length],
            }
            for field in ("score", "rerank_score"):
                value = row.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    result[field] = float(value)
            results.append(result)
        return {"query": query, "results": results}

    return ToolDefinition(
        name=name,
        description="只读检索 FastAPI 文档，返回带 evidence_id 的相关证据。回答文档问题前应先调用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要检索的文档问题"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_top_k,
                    "description": f"返回的证据条数，范围 1 到 {max_top_k}",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_search,
    )


def rag_tool_registry(retriever: Any, *, max_top_k: int = 5) -> ToolRegistry:
    """Create the shared bounded registry used by the RAG/tool workflow."""
    return ToolRegistry(
        [
            retrieval_tool(retriever, max_top_k=max_top_k),
            calculator_tool(),
            date_conversion_tool(),
            structured_lookup_tool(),
        ]
    )
