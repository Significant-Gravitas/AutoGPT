"""Chat Completions payloads with tool results preceding user instructions."""

from backend.copilot.sdk.model_gateway.adapters.openai_tools import (
    arguments_json,
    history_names,
    tool_fields,
)
from backend.copilot.sdk.model_gateway.neutral import (
    DocumentPart,
    ImagePart,
    NeutralMessage,
    NeutralRequest,
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

_DOCUMENT_OMITTED = "[document omitted: this model server cannot read files]"
_IMAGE_FOLLOWS = "[the tool returned an image; it follows in the next message]"


def chat_payload(
    request: NeutralRequest, route: GatewayRoute
) -> tuple[Event, dict[str, str]]:
    """The Chat Completions body, and upstream tool name -> CLI tool name."""
    messages: list[Event] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for message in request.messages:
        if message.role == "user":
            messages.extend(_user_messages(message))
        else:
            messages.append(_assistant_message(message))
    payload: Event = {
        "model": route.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(tool_fields(request))
    payload.update(_sampling_fields(request, route))
    return payload, name_map(history_names(request))


def _user_messages(message: NeutralMessage) -> list[Event]:
    tool_messages: list[Event] = []
    content: list[Event] = []
    moved_images: list[Event] = []
    for part in message.parts:
        if isinstance(part, ToolResultPart):
            tool_messages.append(_tool_message(part))
            moved_images.extend(_image(image) for image in part.images)
        elif isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append(_image(part))
        elif isinstance(part, DocumentPart):
            content.append({"type": "text", "text": _DOCUMENT_OMITTED})
    content.extend(moved_images)
    if not content:
        return tool_messages
    return [*tool_messages, {"role": "user", "content": _user_content(content)}]


def _tool_message(part: ToolResultPart) -> Event:
    text = part.text or (_IMAGE_FOLLOWS if part.images else "")
    return {
        "role": "tool",
        "tool_call_id": upstream_call_id(part.call_id),
        "content": text,
    }


def _user_content(parts: list[Event]) -> str | list[Event]:
    """Plain text where that is all there is: every server takes a string."""
    if all(part["type"] == "text" for part in parts):
        return "\n\n".join(str(part["text"]) for part in parts)
    return parts


def _image(part: ImagePart) -> Event:
    return {"type": "image_url", "image_url": {"url": part.url}}


def _assistant_message(message: NeutralMessage) -> Event:
    text = "".join(p.text for p in message.parts if isinstance(p, TextPart))
    calls = [
        {
            "id": upstream_call_id(part.id),
            "type": "function",
            "function": {
                "name": upstream_tool_name(part.name),
                "arguments": arguments_json(part.arguments),
            },
        }
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    assistant: Event = {
        "role": "assistant",
        "content": text if text or not calls else None,
    }
    if calls:
        assistant["tool_calls"] = calls
    return assistant


def _sampling_fields(request: NeutralRequest, route: GatewayRoute) -> Event:
    fields: Event = {}
    if request.max_tokens is not None:
        fields["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        fields["temperature"] = request.temperature
    if request.top_p is not None:
        fields["top_p"] = request.top_p
    if request.stop:
        fields["stop"] = request.stop
    if route.effort is not None:
        fields["reasoning_effort"] = route.effort
    return fields
