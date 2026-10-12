"""Full-history Responses input with authenticated reasoning and store:false."""

from backend.copilot.sdk.model_gateway.adapters.codex_signature import (
    ReasoningIdentity,
    unseal,
)
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
    request: NeutralRequest, route: GatewayRoute, tag: ReasoningIdentity
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
    payload.update(tool_fields(request, responses=True))
    return payload, name_map(history_names(request))


def input_items(messages: list[NeutralMessage], tag: ReasoningIdentity) -> list[Event]:
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


def _assistant_items(message: NeutralMessage, tag: ReasoningIdentity) -> list[Event]:
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
