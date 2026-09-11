"""Cross-provider format conversion + response extraction helpers.

Lives next to ``providers.call_provider`` so callers don't need to
pull in the block layer for type conversions. The functions here
mirror what was historically in ``backend/blocks/llm.py``; after
Step 3 of the rollout (see plans/idempotent-launching-moth.md),
``blocks/llm.py`` will import them from this module instead of
keeping its own copies.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable

import anthropic
from anthropic.types import ToolParam
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-call tool-call carrier types
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation a model emitted."""

    name: str
    arguments: str


class ToolContentBlock(BaseModel):
    """A tool-use content block normalized across OpenAI + Anthropic shapes."""

    id: str
    type: str
    function: ToolCall


# ---------------------------------------------------------------------------
# OpenAI → Anthropic conversions
# ---------------------------------------------------------------------------


def convert_openai_tool_fmt_to_anthropic(
    openai_tools: list[dict] | None = None,
) -> Iterable[ToolParam] | anthropic.NotGiven:
    """Convert OpenAI tool definitions into Anthropic ``ToolParam`` shape.

    Returns ``anthropic.NOT_GIVEN`` when the caller passed no tools so
    the SDK omits the field from the serialized request (Anthropic
    rejects an empty tools array with HTTP 400).
    """
    if not openai_tools or len(openai_tools) == 0:
        return anthropic.NOT_GIVEN

    anthropic_tools: list[ToolParam] = []
    for tool in openai_tools:
        # Accept both forms: {"type":"function","function":{...}} (OpenAI
        # canonical) and {"name":..., "parameters":...} (raw function def).
        if "function" in tool:
            function_data = tool["function"]
        else:
            function_data = tool

        anthropic_tool: ToolParam = {
            "name": function_data["name"],
            "description": function_data.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": function_data.get("parameters", {}).get("properties", {}),
                "required": function_data.get("parameters", {}).get("required", []),
            },
        }
        anthropic_tools.append(anthropic_tool)

    return anthropic_tools


# ---------------------------------------------------------------------------
# Response-shape extractors (OpenAI-compatible providers)
# ---------------------------------------------------------------------------


def extract_openrouter_cost(response: OpenAIChatCompletion) -> float | None:
    """Pull OpenRouter's per-request USD cost off a chat-completion response.

    OpenRouter populates a ``cost`` field on ``usage`` when the request
    body sets ``usage: {"include": True}``. The OpenAI SDK's typed
    ``CompletionUsage`` doesn't declare it; pydantic v2 makes it
    available on ``model_extra``.

    Returns ``None`` for missing, non-finite, or negative values so
    callers don't accidentally bill at zero or NaN.
    """
    usage = response.usage
    if usage is None:
        return None
    extras = usage.model_extra or {}
    cost = extras.get("cost")
    if cost is None:
        return None
    try:
        cost_f = float(cost)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cost_f) or cost_f < 0:
        return None
    return cost_f


def extract_openai_reasoning(response: Any) -> str | None:
    """Extract reasoning content from an OpenAI-compatible chat response.

    Searches the conventional locations (response-level, choice-level,
    message-level) so the same helper works across OpenAI's evolving
    SDK surface and OpenRouter's normalized response.
    """
    if not response.choices:
        logger.warning("LLM response has empty choices in extract_openai_reasoning")
        return None
    choice = response.choices[0]
    if hasattr(choice, "reasoning") and getattr(choice, "reasoning", None):
        return str(getattr(choice, "reasoning"))
    if hasattr(response, "reasoning") and getattr(response, "reasoning", None):
        return str(getattr(response, "reasoning"))
    if hasattr(choice.message, "reasoning") and getattr(
        choice.message, "reasoning", None
    ):
        return str(getattr(choice.message, "reasoning"))
    return None


def extract_openai_tool_calls(response: Any) -> list[ToolContentBlock] | None:
    """Extract tool calls from an OpenAI-compatible chat response."""
    if not response.choices:
        logger.warning("LLM response has empty choices in extract_openai_tool_calls")
        return None
    if response.choices[0].message.tool_calls:
        return [
            ToolContentBlock(
                id=tool.id,
                type=tool.type,
                function=ToolCall(
                    name=tool.function.name,
                    arguments=tool.function.arguments,
                ),
            )
            for tool in response.choices[0].message.tool_calls
        ]
    return None


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def sanitize_messages_for_utf8(messages: list[dict]) -> None:
    """Replace unpaired surrogates in message content in-place.

    httpx encodes the JSON request body to UTF-8; unpaired surrogates
    (e.g., copy-pasted emojis stored as raw code points) raise
    ``UnicodeEncodeError`` before the request even leaves the process.
    Mirrors the historical sanitization in
    ``backend/blocks/llm.py::_llm_call``.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            try:
                content.encode("utf-8")
            except UnicodeEncodeError:
                logger.warning("Sanitized unpaired surrogates in LLM prompt content")
                msg["content"] = content.encode("utf-8", errors="surrogatepass").decode(
                    "utf-8", errors="replace"
                )
