"""What Otto reads back from computer use: the tool list, results, errors."""

import json
from typing import Any

from backend.copilot.sdk.computer_use_approval import approval_arguments
from backend.copilot.sdk.computer_use_policy import LOCAL_PC_COMPUTER_TOOL

_SUMMARY_CHARS = 280
# A structured-content string this long that a text block already shows is
# sent once, not twice.
_DUPLICATE_STRING_CHARS = 200


def catalog_view(catalog: dict[str, Any], requested: list[str]) -> dict[str, Any]:
    tools = [tool for tool in catalog.get("tools") or [] if isinstance(tool, dict)]
    by_name = {tool.get("name"): tool for tool in tools}
    view: dict[str, Any] = {
        "driver": catalog.get("driver"),
        "instructions": catalog.get("instructions") or "",
        "tools": [
            {
                "name": tool.get("name"),
                "summary": _summary(tool.get("description")),
                **_approval(tool),
            }
            for tool in tools
        ],
        "how_to_call": (
            f"Call `{LOCAL_PC_COMPUTER_TOOL}` with `tool` and `arguments`. Pass "
            "`tools` here to get the input schemas of the tools you will use. A "
            "call that uses one of a tool's `approval_arguments` reaches the "
            "user's disk: it waits for the user to approve it on a card, so use "
            "one only when the task needs that file."
        ),
    }
    if requested:
        view["schemas"] = {
            name: {
                "description": by_name[name].get("description"),
                "input_schema": by_name[name].get("input_schema"),
            }
            for name in requested
            if name in by_name
        }
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            view["unknown_tools"] = unknown
    return view


def _approval(tool: dict[str, Any]) -> dict[str, Any]:
    named = approval_arguments(tool)
    return {"approval_arguments": named} if named else {}


def _summary(description: Any) -> str:
    text = str(description or "").strip()
    first = text.split("\n\n", 1)[0].replace("\n", " ")
    return first if len(first) <= _SUMMARY_CHARS else first[: _SUMMARY_CHARS - 1] + "…"


def mcp_result(result: dict[str, Any]) -> dict[str, Any]:
    """The driver's answer as MCP content: its text and images as they came,
    then its structured content without strings the text already shows."""
    content: list[dict[str, Any]] = []
    texts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
            content.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image" and isinstance(block.get("data"), str):
            content.append(
                {
                    "type": "image",
                    "data": block["data"],
                    "mimeType": str(block.get("mime_type") or "image/png"),
                }
            )
    structured = result.get("structured_content")
    if isinstance(structured, dict) and structured:
        shown = _without_duplicates(structured, texts)
        content.append({"type": "text", "text": json.dumps(shown, default=str)})
    if not content:
        content.append({"type": "text", "text": "(no output)"})
    return {"content": content, "isError": bool(result.get("is_error"))}


def _without_duplicates(structured: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    return {
        key: (
            "(shown in the text above)"
            if isinstance(value, str)
            and len(value) >= _DUPLICATE_STRING_CHARS
            and any(value.strip() in text for text in texts)
            else value
        )
        for key, value in structured.items()
    }


def ok(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
        "isError": False,
    }


def err(code: str, message: str, details: dict | None = None) -> dict[str, Any]:
    """A structured error: the LLM-friendly message and the code to branch on."""
    body: dict[str, Any] = {"code": code, "error": message}
    if details:
        body["details"] = details
    return {
        "content": [{"type": "text", "text": json.dumps(body, default=str)}],
        "isError": True,
    }
