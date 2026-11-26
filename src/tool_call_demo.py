"""Run one small model-to-tool-to-model workflow demonstration."""

from __future__ import annotations

import argparse
import json
import sys

if __package__:
    from .llm_provider import OpenAIResponsesGenerator
    from .tools import default_tool_registry
else:
    from llm_provider import OpenAIResponsesGenerator
    from tools import default_tool_registry


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Question for the calculator-tool demo")
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    args = parser.parse_args()
    if args.max_tool_rounds < 1:
        parser.error("max-tool-rounds must be at least 1")

    agent = OpenAIResponsesGenerator.from_env()
    result = agent.generate_with_tools(
        args.question,
        default_tool_registry(),
        max_tool_rounds=args.max_tool_rounds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
