"""How ``structured_completion`` asks each provider for structured output.

JSON mode (``force_json_output``) constrains OpenAI, OpenRouter, Groq and
Ollama, but the native Anthropic Messages call ignores it. There the output
comes from one tool whose input schema is the response model, read back from
the tool call, as on the dream batch path. The tool is forced where the model
accepts that. Models that answer a forced ``tool_choice`` with a 400 (Opus
5.5 among them) get ``auto`` instead, a prompt line asking for the call
(``with_output_tool_instruction``), and their message text parsed when they
answer in prose after all.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.copilot.model_normalize import normalize_model_for_anthropic
from backend.util.llm.providers import ProviderLiteral, ProviderResponse
from backend.util.llm.tool_use import (
    auto_tool_choice,
    is_forced_tool_choice,
    pydantic_to_anthropic_tool,
    structured_tool_choice,
)

# Anthropic tool names take letters, digits, "_" and "-", 64 at most.
_TOOL_NAME_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_NAME_MAX_LENGTH = 64

# Appended to a batch phase tool's description when the tool is not forced
# (``tool_choice`` ``auto``): with the prompt line from
# ``with_output_tool_instruction`` it is what gets the tool called. Forced
# tools keep their description as written, and the sync path's description
# asks for one complete call in either mode.
OUTPUT_TOOL_CALL_ONCE = (
    "Call this tool exactly once, with the complete result as its input."
)


class StructuredRequest(BaseModel):
    """What ``call_provider`` is asked for beyond the prompt."""

    model_config = ConfigDict(frozen=True)

    model: str
    force_json_output: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None

    @property
    def forces_output_tool(self) -> bool:
        return is_forced_tool_choice(self.tool_choice)

    def with_output_tool_optional(self) -> StructuredRequest:
        """This request with the tool left to the model (``auto``): the
        retry for a model that turned the forced tool down."""
        return self.model_copy(update={"tool_choice": auto_tool_choice()})

    def prompt(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """*messages* as sent: when the output tool is offered but not
        forced, with the line asking the model to call it."""
        if not self.tools or self.forces_output_tool:
            return messages
        return with_output_tool_instruction(messages, self.tools[0]["name"])


def structured_request(
    provider: ProviderLiteral, model: str, response_model: type[BaseModel]
) -> StructuredRequest:
    """JSON mode, except on the native Anthropic API: there the model takes
    its native spelling and the call carries one tool built from
    *response_model*, forced where the model accepts a forced tool
    (``structured_tool_choice``).

    Raises ``ValueError`` for a non-Anthropic model on the ``anthropic``
    provider.
    """
    if provider != "anthropic":
        return StructuredRequest(model=model, force_json_output=True)
    native_model = normalize_model_for_anthropic(model)
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
        model=native_model,
        tools=[tool],
        tool_choice=structured_tool_choice(native_model, tool_name),
    )


def structured_payload(response: ProviderResponse) -> str:
    """The text to parse: the output tool's arguments when the model called
    it, the message text otherwise."""
    if response.tool_calls:
        return response.tool_calls[0].function.arguments
    return (response.content or "").strip()


def output_tool_name(response_model: type[BaseModel]) -> str:
    safe_name = _TOOL_NAME_UNSAFE_RE.sub("_", response_model.__name__)
    return f"emit_{safe_name}"[:_TOOL_NAME_MAX_LENGTH]


def with_output_tool_instruction(
    messages: list[dict[str, str]], tool_name: str
) -> list[dict[str, str]]:
    """*messages* with one line asking for the answer through *tool_name*,
    for a model the tool can't be forced on. The line closes the last user
    turn, after any "reply with only the JSON" wording, which it redirects
    into the tool call."""
    instruction = (
        f"Give your answer by calling the {tool_name} tool exactly once, with "
        "the complete result as its input, rather than writing it out as text."
    )
    if messages and messages[-1]["role"] == "user":
        *earlier, last = messages
        return [*earlier, {**last, "content": f"{last['content']}\n\n{instruction}"}]
    return [*messages, {"role": "user", "content": instruction}]
