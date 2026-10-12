"""Size of the copilot tool prefix: every tool schema today versus the eager
core plus the registry tools.

Tokens are estimated with tiktoken (cl100k) scaled by the measured Anthropic
ratio for this content; pass ``--exact`` with ``ANTHROPIC_API_KEY`` set to
count with the Messages API instead.  The dollar figure prices the saving as
a cache write, because a cold prefix is written once per fresh session.
"""

import argparse
import json
import os
from collections.abc import Mapping

import anthropic
import tiktoken
from anthropic.types import ToolParam
from pydantic import BaseModel

from backend.copilot.capabilities.sources import EAGER_CORE, RETIRED_TOOLS
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool

# Measured on the live prefix: Anthropic tokens / cl100k tokens.
ANTHROPIC_PER_CL100K = 1.5
# OpenRouter cache-write price for the Claude route, USD per million tokens.
CACHE_WRITE_USD_PER_M = 3.75
COUNT_MODEL = "claude-sonnet-5"


class ToolSchema(BaseModel):
    """One tool as the model sees it (OpenAI ``function`` shape)."""

    name: str
    description: str
    parameters: dict[str, object]

    def as_anthropic(self) -> ToolParam:
        return ToolParam(
            name=self.name, description=self.description, input_schema=self.parameters
        )


# Draft schemas for the registry tools; the real ones land with the tools.
REGISTRY_TOOL_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        name="find_capability",
        description=(
            "Search everything the platform can do: integrations, blocks, MCP "
            "servers and tools. Results are ranked and show connection state. "
            "Call this before saying something is not possible."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Service, action or block name.",
                },
                "context": {
                    "type": "string",
                    "enum": ["direct", "graph"],
                    "default": "direct",
                },
                "kind": {
                    "type": "string",
                    "enum": ["tool", "block", "mcp_server", "agent"],
                },
            },
            "required": ["query"],
        },
    ),
    ToolSchema(
        name="describe_capability",
        description="Full input/output schema for one capability id before first use.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "expand": {"type": "boolean", "default": False},
            },
            "required": ["id"],
        },
    ),
    ToolSchema(
        name="run_capability",
        description=(
            "Run a capability by id with its input. An unconnected capability "
            "returns a sign-in card: show it and stop. May pause for review."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "input": {"type": "object"},
                "validate_only": {"type": "boolean", "default": False},
            },
            "required": ["id", "input"],
        },
    ),
    ToolSchema(
        name="resume_capability",
        description="Resume a paused run_capability call after the user approved it.",
        parameters={
            "type": "object",
            "properties": {
                "review_id": {"type": "string"},
                "input_overrides": {"type": "object"},
            },
            "required": ["review_id"],
        },
    ),
]


class PrefixReport(BaseModel):
    today_tools: int
    today_tokens: int
    registry_tools: int
    registry_tokens: int
    exact: bool

    @property
    def saved_tokens(self) -> int:
        return max(0, self.today_tokens - self.registry_tokens)

    def saved_usd_per_cold_prefix(self) -> float:
        return self.saved_tokens / 1_000_000 * CACHE_WRITE_USD_PER_M


def today_schemas(tools: Mapping[str, BaseTool]) -> list[ToolSchema]:
    return [_schema(tool) for tool in tools.values() if tool.is_available]


def registry_schemas(tools: Mapping[str, BaseTool]) -> list[ToolSchema]:
    eager = [
        _schema(tool)
        for name, tool in tools.items()
        if tool.is_available and name in EAGER_CORE and name not in RETIRED_TOOLS
    ]
    return eager + REGISTRY_TOOL_SCHEMAS


def _schema(tool: BaseTool) -> ToolSchema:
    return ToolSchema.model_validate(dict(tool.as_openai_tool()["function"]))


def estimate_tokens(schemas: list[ToolSchema]) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = json.dumps([schema.model_dump() for schema in schemas], ensure_ascii=False)
    return round(len(encoding.encode(text)) * ANTHROPIC_PER_CL100K)


def exact_tokens(schemas: list[ToolSchema]) -> int:
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model=COUNT_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        tools=[schema.as_anthropic() for schema in schemas],
    )
    return response.input_tokens


def measure(tools: Mapping[str, BaseTool], *, exact: bool = False) -> PrefixReport:
    count = exact_tokens if exact else estimate_tokens
    today, registry = today_schemas(tools), registry_schemas(tools)
    return PrefixReport(
        today_tools=len(today),
        today_tokens=count(today),
        registry_tools=len(registry),
        registry_tokens=count(registry),
        exact=exact,
    )


def format_report(report: PrefixReport) -> str:
    method = "Anthropic count_tokens" if report.exact else "cl100k x 1.5 estimate"
    return "\n".join(
        [
            f"tool prefix ({method})",
            f"  today:    {report.today_tools:3d} tools  {report.today_tokens:6,d} tokens",
            f"  registry: {report.registry_tools:3d} tools  {report.registry_tokens:6,d} tokens",
            f"  saved:    {report.saved_tokens:,d} tokens per cold prefix"
            + f"  (${report.saved_usd_per_cold_prefix():.3f}"
            + f" at ${CACHE_WRITE_USD_PER_M}/M cache write)",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exact", action="store_true", help="count with the Anthropic API"
    )
    args = parser.parse_args()
    exact = args.exact and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if args.exact and not exact:
        print("ANTHROPIC_API_KEY not set; falling back to the estimate")
    print(format_report(measure(TOOL_REGISTRY, exact=exact)))


if __name__ == "__main__":
    main()
