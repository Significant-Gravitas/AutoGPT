"""Helpers for OpenAI Responses API.

This module provides utilities for using OpenAI's Responses API, which is the
default for all modern OpenAI models. Legacy models (gpt-3.5-turbo) that do not
support the Responses API fall back to Chat Completions.
"""

from typing import Any

# Legacy models that do NOT support the Responses API.
# These must use chat.completions.create instead of responses.create.
CHAT_COMPLETIONS_ONLY_MODELS = frozenset(
    {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
    }
)


def requires_chat_completions(model: str) -> bool:
    """Check if model requires the legacy Chat Completions API (exact match).

    Args:
        model: The model identifier string (e.g., "gpt-3.5-turbo", "gpt-4o")

    Returns:
        True if the model requires chat.completions.create, False otherwise
    """
    return model in CHAT_COMPLETIONS_ONLY_MODELS


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
            entry: dict[str, Any] = {
                "type": "function",
                "name": func.get("name"),
                # Note: strict=True is default in Responses API
            }
            if func.get("description") is not None:
                entry["description"] = func["description"]
            if func.get("parameters") is not None:
                entry["parameters"] = func["parameters"]
            converted.append(entry)
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


def extract_responses_usage(response: Any) -> tuple[int, int]:
    """Extract token usage from Responses API response.

    The Responses API uses input_tokens/output_tokens (not prompt_tokens/completion_tokens).

    Args:
        response: The Responses API response object

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    if not response.usage:
        return 0, 0

    return (
        getattr(response.usage, "input_tokens", 0),
        getattr(response.usage, "output_tokens", 0),
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
