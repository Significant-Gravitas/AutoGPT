"""Function schemas, choices and history names shared by both translators."""

import json

from backend.copilot.sdk.model_gateway.neutral import (
    NeutralRequest,
    ToolCallPart,
    ToolChoice,
    ToolSpec,
)
from backend.copilot.sdk.model_gateway.stream import Event
from backend.copilot.sdk.model_gateway.tool_names import upstream_tool_name


def tool_fields(request: NeutralRequest, *, responses: bool = False) -> Event:
    if not request.tools:
        return {}
    fields: Event = {
        "tools": [_function_tool(tool, responses) for tool in request.tools]
    }
    if responses or request.tool_choice is not None:
        fields["tool_choice"] = _choice(request.tool_choice, responses)
    if request.tool_choice is not None and not request.tool_choice.parallel:
        fields["parallel_tool_calls"] = False
    return fields


def history_names(request: NeutralRequest) -> list[str]:
    return [tool.name for tool in request.tools] + [
        part.name
        for message in request.messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


def arguments_json(arguments: object) -> str:
    return (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )


def _function_tool(tool: ToolSpec, responses: bool) -> Event:
    function: Event = {
        "name": upstream_tool_name(tool.name),
        "description": tool.description,
        "parameters": _parameters(tool.parameters),
    }
    return (
        {"type": "function", **function, "strict": False}
        if responses
        else {"type": "function", "function": function}
    )


def _parameters(schema: dict[str, object]) -> Event:
    cleaned = {key: value for key, value in schema.items() if key != "$schema"}
    cleaned.setdefault("type", "object")
    if cleaned["type"] == "object":
        cleaned.setdefault("properties", {})
    return cleaned


def _choice(choice: ToolChoice | None, responses: bool) -> object:
    if choice is None:
        return "auto"
    if choice.mode == "any":
        return "required"
    if choice.mode == "tool" and choice.name:
        function = {"name": upstream_tool_name(choice.name)}
        return (
            {"type": "function", **function}
            if responses
            else {"type": "function", "function": function}
        )
    return "none" if choice.mode == "none" else "auto"
