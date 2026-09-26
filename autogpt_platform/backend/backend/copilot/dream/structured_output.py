"""How ``structured_completion`` asks each provider for structured output.

JSON mode (``force_json_output``) constrains OpenAI, OpenRouter, Groq and
Ollama, but the native Anthropic Messages call ignores it. There the output
is constrained the way the dream batch path does it: one forced tool whose
input schema is the response model, read back from the tool call.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.copilot.model_normalize import normalize_model_for_anthropic
from backend.util.llm.providers import ProviderLiteral, ProviderResponse
from backend.util.llm.tool_use import force_tool_choice, pydantic_to_anthropic_tool

# Anthropic tool names take letters, digits, "_" and "-", 64 at most.
_TOOL_NAME_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_NAME_MAX_LENGTH = 64


class StructuredRequest(BaseModel):
    """What ``call_provider`` is asked for beyond the prompt."""

    model_config = ConfigDict(frozen=True)

    model: str
    force_json_output: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None


def structured_request(
    provider: ProviderLiteral, model: str, response_model: type[BaseModel]
) -> StructuredRequest:
    """JSON mode, except on the native Anthropic API: there the model takes
    its native spelling and the call carries one forced tool built from
    *response_model*.

    Raises ``ValueError`` for a non-Anthropic model on the ``anthropic``
    provider.
    """
    if provider != "anthropic":
        return StructuredRequest(model=model, force_json_output=True)
    tool_name = output_tool_name(response_model)
    tool = pydantic_to_anthropic_tool(
        response_model,
        tool_name=tool_name,
        description=(
            f"Return the {response_model.__name__} result: call this once, "
            "with every field the schema requires."
        ),
    )
    return StructuredRequest(
        model=normalize_model_for_anthropic(model),
        tools=[tool],
        tool_choice=force_tool_choice(tool_name),
    )


def structured_payload(response: ProviderResponse) -> str:
    """The text to parse: the forced tool's arguments when the model called
    it, the message text otherwise."""
    if response.tool_calls:
        return response.tool_calls[0].function.arguments
    return (response.content or "").strip()


def output_tool_name(response_model: type[BaseModel]) -> str:
    safe_name = _TOOL_NAME_UNSAFE_RE.sub("_", response_model.__name__)
    return f"emit_{safe_name}"[:_TOOL_NAME_MAX_LENGTH]
