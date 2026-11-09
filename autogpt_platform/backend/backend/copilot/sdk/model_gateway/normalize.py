"""Read CLI Anthropic requests into a shared neutral form.

Drop billing metadata to preserve prefix caches; turn inline system notes into
user reminders and merge adjacent roles without splitting tool results."""

import json
from collections.abc import Mapping

from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.neutral import (
    DocumentPart,
    ImagePart,
    NeutralMessage,
    NeutralRequest,
    Part,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolChoiceMode,
    ToolResultPart,
    ToolSpec,
)

BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
_TOOL_CHOICE_MODES: dict[str, ToolChoiceMode] = {
    "auto": "auto",
    "any": "any",
    "none": "none",
    "tool": "tool",
}


def normalize_request(body: Mapping[str, object]) -> NeutralRequest:
    """Read one ``/v1/messages`` body; a malformed one is a 400."""
    return NeutralRequest(
        system=system_text(body.get("system")),
        messages=_messages(body.get("messages")),
        tools=_tools(body.get("tools")),
        tool_choice=_tool_choice(body.get("tool_choice")),
        max_tokens=_positive_int(body.get("max_tokens")),
        temperature=_number(body.get("temperature")),
        top_p=_number(body.get("top_p")),
        stop=[s for s in _list(body.get("stop_sequences")) if isinstance(s, str)],
        session_id=_session_id(body.get("metadata")),
    )


def reminder(text: str) -> str:
    return f"<system-reminder>\n{text}\n</system-reminder>"


def strip_billing_header(system: object) -> object:
    """``system`` without the CLI's billing line, in the shape it came in."""
    if isinstance(system, str):
        return "" if _is_billing_line(system) else system
    if not isinstance(system, list):
        return system
    return [
        block
        for block in system
        if not (isinstance(block, dict) and _is_billing_line(block.get("text")))
    ]


def system_text(system: object) -> str:
    kept = strip_billing_header(system)
    if isinstance(kept, str):
        return kept
    return "\n\n".join(_block_texts(kept))


def _is_billing_line(text: object) -> bool:
    return isinstance(text, str) and text.lstrip().startswith(BILLING_HEADER_PREFIX)


def _messages(raw: object) -> list[NeutralMessage]:
    if not isinstance(raw, list):
        raise GatewayError(400, "invalid_request_error", "messages must be an array")
    merged: list[NeutralMessage] = []
    for item in raw:
        message = _message(item)
        if message is None:
            continue
        if merged and merged[-1].role == message.role:
            merged[-1].parts.extend(message.parts)
        else:
            merged.append(message)
    return merged


def _message(item: object) -> NeutralMessage | None:
    if not isinstance(item, dict):
        raise GatewayError(400, "invalid_request_error", "each message is an object")
    role, content = item.get("role"), item.get("content")
    if role == "system":
        text = (
            content if isinstance(content, str) else "\n\n".join(_block_texts(content))
        )
        if not text:
            return None
        return NeutralMessage(role="user", parts=[TextPart(text=reminder(text))])
    if role not in ("user", "assistant"):
        raise GatewayError(
            400, "invalid_request_error", "roles are user, assistant or system"
        )
    parts = _parts(content)
    if not parts:
        return None
    return NeutralMessage(role="user" if role == "user" else "assistant", parts=parts)


def _parts(content: object) -> list[Part]:
    if isinstance(content, str):
        return [TextPart(text=content)]
    return [p for block in _list(content) if (p := _part(block)) is not None]


def _part(block: object) -> Part | None:
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    if kind == "text":
        return TextPart(text=str(block.get("text") or ""))
    if kind == "image":
        return _image(block.get("source"))
    if kind == "document":
        return _document(block.get("source"))
    if kind == "tool_use":
        return ToolCallPart(
            id=str(block.get("id") or ""),
            name=str(block.get("name") or ""),
            arguments=block.get("input") if block.get("input") is not None else {},
        )
    if kind == "tool_result":
        return _tool_result(block)
    if kind == "thinking":
        signature = block.get("signature")
        return ReasoningPart(
            text=str(block.get("thinking") or ""),
            signature=signature if isinstance(signature, str) else None,
        )
    return None


def _tool_result(block: Mapping[str, object]) -> ToolResultPart:
    content = block.get("content")
    parts: list[TextPart | ImagePart] = []
    if isinstance(content, str):
        parts.append(TextPart(text=content))
    for item in _list(content):
        part = _part(item)
        if isinstance(part, (TextPart, ImagePart)):
            parts.append(part)
    return ToolResultPart(
        call_id=str(block.get("tool_use_id") or ""),
        content=parts,
        is_error=block.get("is_error") is True,
    )


def _image(source: object) -> TextPart | ImagePart:
    if isinstance(source, dict):
        data, media_type = source.get("data"), source.get("media_type")
        if source.get("type") == "base64" and isinstance(data, str):
            return ImagePart(url=f"data:{media_type or 'image/png'};base64,{data}")
    return TextPart(text="[image omitted: its source cannot be forwarded]")


def inline_images(raw: object) -> object:
    """Rewrite image content, preserving tool inputs and all other body fields."""
    if not isinstance(raw, list):
        return raw
    return [
        (
            {**message, "content": _inline_content(message["content"])}
            if isinstance(message, dict) and "content" in message
            else message
        )
        for message in raw
    ]


def _inline_content(content: object) -> object:
    if not isinstance(content, list):
        return content
    return [_inline_block(block) for block in content]


def _inline_block(block: object) -> object:
    if not isinstance(block, dict):
        return block
    source = block.get("source")
    if (
        block.get("type") == "image"
        and isinstance(source, dict)
        and source.get("type") == "url"
    ):
        return {
            "type": "text",
            "text": "[image omitted: its source cannot be forwarded]",
        }
    if block.get("type") == "tool_result" and "content" in block:
        return {**block, "content": _inline_content(block["content"])}
    return block


def _document(source: object) -> TextPart | DocumentPart:
    if isinstance(source, dict):
        data, media_type = source.get("data"), source.get("media_type")
        if source.get("type") == "base64" and isinstance(data, str):
            return DocumentPart(
                media_type=str(media_type or "application/pdf"), data=data
            )
        if source.get("type") == "text" and isinstance(data, str):
            return TextPart(text=data)
    return TextPart(text="[document omitted: its source cannot be forwarded]")


def _tools(raw: object) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for tool in _list(raw):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            # A server tool (web search and the like) the upstream cannot run.
            continue
        specs.append(
            ToolSpec(
                name=tool["name"],
                description=str(tool.get("description") or ""),
                parameters={str(k): v for k, v in schema.items()},
            )
        )
    return specs


def _tool_choice(raw: object) -> ToolChoice | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    return ToolChoice(
        mode=_TOOL_CHOICE_MODES.get(str(raw.get("type")), "auto"),
        name=name if isinstance(name, str) else None,
        parallel=raw.get("disable_parallel_tool_use") is not True,
    )


def _session_id(metadata: object) -> str | None:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("user_id"), str):
        return None
    try:
        decoded = json.loads(metadata["user_id"])
    except ValueError:
        return None
    session = decoded.get("session_id") if isinstance(decoded, dict) else None
    return session if isinstance(session, str) and session else None


def _block_texts(blocks: object) -> list[str]:
    return [
        str(block["text"])
        for block in _list(blocks)
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    ]


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
