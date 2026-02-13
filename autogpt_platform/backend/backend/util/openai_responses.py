"""Helpers for OpenAI Responses API migration.

This module provides utilities for conditionally using OpenAI's Responses API
instead of Chat Completions for reasoning models (o1, o3, etc.) that require it.
"""

from typing import Any

# Exact model identifiers that require the Responses API.
# Use exact matching to avoid false positives on future models.
# NOTE: Update this set when OpenAI releases new reasoning models.
REASONING_MODELS = frozenset(
    {
        # O1 family
        "o1",
        "o1-mini",
        "o1-preview",
        "o1-2024-12-17",
        # O3 family
        "o3",
        "o3-mini",
        "o3-2025-04-16",
        "o3-mini-2025-01-31",
    }
)


def requires_responses_api(model: str) -> bool:
    """Check if model requires the Responses API (exact match).

    Args:
        model: The model identifier string (e.g., "o3-mini", "gpt-4o")

    Returns:
        True if the model requires responses.create, False otherwise
    """
    return model in REASONING_MODELS


def convert_tools_to_responses_format(tools: list[dict] | None) -> list[dict]:
    """Convert Chat Completions tool format to Responses API format.

    The Responses API uses internally-tagged polymorphism (flatter structure)
    and functions are strict by default.

    Chat Completions format:
        {"type": "function", "function": {"name": "...", "parameters": {...}}}

    Responses API format:
        {"type": "function", "name": "...", "parameters": {...}}

    Args:
        tools: List of tools in Chat Completions format

    Returns:
        List of tools in Responses API format
    """
    if not tools:
        return []

    converted = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            converted.append(
                {
                    "type": "function",
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters"),
                    # Note: strict=True is default in Responses API
                }
            )
        else:
            # Pass through non-function tools as-is
            converted.append(tool)
    return converted


def extract_responses_tool_calls(response: Any) -> list[dict] | None:
    """Extract tool calls from Responses API response.

    The Responses API returns tool calls as separate items in the output array
    with type="function_call".

    Args:
        response: The Responses API response object

    Returns:
        List of tool calls in a normalized format, or None if no tool calls
    """
    tool_calls = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )
    return tool_calls if tool_calls else None


def extract_usage(response: Any, is_responses_api: bool) -> tuple[int, int]:
    """Extract token usage from either API response.

    The Responses API uses different field names for token counts:
    - Chat Completions: prompt_tokens, completion_tokens
    - Responses API: input_tokens, output_tokens

    Args:
        response: The API response object
        is_responses_api: True if response is from Responses API

    Returns:
        Tuple of (prompt_tokens, completion_tokens)
    """
    if not response.usage:
        return 0, 0

    if is_responses_api:
        # Responses API uses different field names
        return (
            getattr(response.usage, "input_tokens", 0),
            getattr(response.usage, "output_tokens", 0),
        )
    else:
        # Chat Completions API
        return (
            getattr(response.usage, "prompt_tokens", 0),
            getattr(response.usage, "completion_tokens", 0),
        )


def extract_responses_content(response: Any) -> str:
    """Extract text content from Responses API response.

    Args:
        response: The Responses API response object

    Returns:
        The text content from the response, or empty string if none
    """
    # The SDK provides a helper property
    if hasattr(response, "output_text"):
        return response.output_text or ""

    # Fallback: manually extract from output items
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    return getattr(content, "text", "")
    return ""


def extract_responses_reasoning(response: Any) -> str | None:
    """Extract reasoning content from Responses API response.

    Reasoning models return their reasoning process in the response,
    which can be useful for debugging or display.

    Args:
        response: The Responses API response object

    Returns:
        The reasoning text, or None if not present
    """
    for item in response.output:
        if getattr(item, "type", None) == "reasoning":
            # Reasoning items may have summary or content
            summary = getattr(item, "summary", [])
            if summary:
                # Join summary items if present
                texts = []
                for s in summary:
                    if hasattr(s, "text"):
                        texts.append(s.text)
                if texts:
                    return "\n".join(texts)
    return None
