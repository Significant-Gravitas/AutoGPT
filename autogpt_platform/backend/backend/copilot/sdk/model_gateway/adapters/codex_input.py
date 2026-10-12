"""The CLI's conversation as a Codex Responses request body.

The rules are the ones CLIProxyAPI's Claude-to-Codex translator runs against
the same backend (re-implemented here, not copied): ``store: false`` with the
full history every call, encrypted reasoning back as ``reasoning`` items,
tool rounds as ``function_call``/``function_call_output`` pairs, images as
data-URL ``input_image`` parts. ``max_tokens``, ``temperature`` and stop
sequences are not sent: the Codex backend rejects them.
"""

from backend.copilot.sdk.model_gateway.adapters.codex_signature import unseal
from backend.copilot.sdk.model_gateway.adapters.openai_request import (
    arguments_json,
    tool_parameters,
)
from backend.copilot.sdk.model_gateway.neutral import (
    DocumentPart,
    ImagePart,
    NeutralMessage,
    NeutralRequest,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.stream import Event
from backend.copilot.sdk.model_gateway.tool_names import (
    name_map,
    upstream_call_id,
    upstream_tool_name,
)


def responses_payload(
    request: NeutralRequest, route: GatewayRoute, tag: str
) -> tuple[Event, dict[str, str]]:
    """The Responses body, and upstream tool name -> CLI tool name."""
    reasoning: Event = {"summary": "auto"}
    if route.effort is not None:
        reasoning["effort"] = route.effort
    payload: Event = {
        "model": route.model,
        "input": input_items(request.messages, tag),
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "reasoning": reasoning,
    }
    if request.system:
        payload["instructions"] = request.system
    if request.session_id:
        # One chat keeps one key, so the backend can route its calls to the
        # same prompt cache.
        payload["prompt_cache_key"] = request.session_id
    payload.update(_tool_fields(request))
    names = [tool.name for tool in request.tools]
    for message in request.messages:
        names.extend(p.name for p in message.parts if isinstance(p, ToolCallPart))
    return payload, name_map(names)


def input_items(messages: list[NeutralMessage], tag: str) -> list[Event]:
    items: list[Event] = []
    for message in messages:
        if message.role == "user":
            items.extend(_user_items(message))
        else:
            items.extend(_assistant_items(message, tag))
    return items


def _user_items(message: NeutralMessage) -> list[Event]:
    outputs: list[Event] = []
    content: list[Event] = []
    for part in message.parts:
        if isinstance(part, ToolResultPart):
            outputs.append(_function_call_output(part))
        elif isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "input_image", "image_url": part.url})
        elif isinstance(part, DocumentPart):
            content.append(
                {
                    "type": "input_file",
                    "file_data": part.data_url,
                    "filename": "document",
                }
            )
    if not content:
        return outputs
    return [*outputs, {"type": "message", "role": "user", "content": content}]


def _function_call_output(part: ToolResultPart) -> Event:
    output: str | list[Event] = part.text
    if part.images:
        output = [{"type": "input_text", "text": part.text}] if part.text else []
        output.extend({"type": "input_image", "image_url": i.url} for i in part.images)
    return {
        "type": "function_call_output",
        "call_id": upstream_call_id(part.call_id),
        "output": output,
    }


def _assistant_items(message: NeutralMessage, tag: str) -> list[Event]:
    items: list[Event] = []
    text: list[str] = []

    def flush_text() -> None:
        if text:
            content = [{"type": "output_text", "text": "".join(text)}]
            items.append({"type": "message", "role": "assistant", "content": content})
            text.clear()

    for part in message.parts:
        if isinstance(part, TextPart):
            text.append(part.text)
        elif isinstance(part, ToolCallPart):
            flush_text()
            items.append(_function_call(part))
        elif isinstance(part, ReasoningPart):
            encrypted = unseal(part.signature, tag)
            if encrypted is not None:
                flush_text()
                items.append(
                    {"type": "reasoning", "summary": [], "encrypted_content": encrypted}
                )
    flush_text()
    return items


def _function_call(part: ToolCallPart) -> Event:
    return {
        "type": "function_call",
        "call_id": upstream_call_id(part.id),
        "name": upstream_tool_name(part.name),
        "arguments": arguments_json(part.arguments),
    }


def _tool_fields(request: NeutralRequest) -> Event:
    if not request.tools:
        return {}
    fields: Event = {
        "tools": [
            {
                "type": "function",
                "name": upstream_tool_name(tool.name),
                "description": tool.description,
                "parameters": tool_parameters(tool.parameters),
                "strict": False,
            }
            for tool in request.tools
        ],
        "tool_choice": "auto",
    }
    choice = request.tool_choice
    if choice is None:
        return fields
    if choice.mode == "any":
        fields["tool_choice"] = "required"
    elif choice.mode == "none":
        fields["tool_choice"] = "none"
    elif choice.mode == "tool" and choice.name:
        fields["tool_choice"] = {
            "type": "function",
            "name": upstream_tool_name(choice.name),
        }
    if not choice.parallel:
        fields["parallel_tool_calls"] = False
    return fields
