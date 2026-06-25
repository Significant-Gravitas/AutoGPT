"""Pydantic → Anthropic tool definition conversion.

Anthropic's Messages API supports forced structured output via tool
use: provide a tool whose ``input_schema`` matches the desired output
shape, then set ``tool_choice={"type":"tool","name":<tool_name>}``
and the model is constrained to call exactly that tool with arguments
matching the schema — no preamble, no markdown, no "Looking at the
inputs, I need to..." prose. The model literally cannot emit anything
else.

This module turns any Pydantic ``BaseModel`` subclass into the tool
definition Anthropic expects + provides the forced ``tool_choice``
helper.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def pydantic_to_anthropic_tool(
    response_model: type[BaseModel],
    *,
    tool_name: str,
    description: str,
) -> dict[str, Any]:
    """Convert a Pydantic model class to an Anthropic tool definition.

    Anthropic accepts a tool spec of shape
    ``{"name": str, "description": str, "input_schema": dict}`` where
    ``input_schema`` is a JSON Schema describing the call's arguments.
    ``BaseModel.model_json_schema()`` returns a compatible JSON Schema
    with one quirk: nested models are emitted as ``$ref`` pointers into
    a ``$defs`` table. Anthropic's parser accepts ``$ref``/``$defs``
    forms in current API versions, but inlined schemas are more
    predictable across SDK versions — we inline them defensively.

    The returned dict is safe to pass straight into
    ``client.messages.create(tools=[<this>])`` or to include in a
    ``BatchRequest``'s params.
    """
    schema = response_model.model_json_schema()
    inlined = _inline_refs(schema)
    # Drop fields Anthropic doesn't use (title, $defs after inlining).
    inlined.pop("$defs", None)
    inlined.pop("title", None)
    return {
        "name": tool_name,
        "description": description,
        "input_schema": inlined,
    }


def force_tool_choice(tool_name: str) -> dict[str, Any]:
    """Build a ``tool_choice`` that forces Claude to call exactly this tool.

    With ``disable_parallel_tool_use=True`` the model emits exactly
    one tool_use block — no preamble, no parallel calls. This is what
    eliminates the JSON-parse failures we saw on the sanitize phase
    (where the model would emit chain-of-thought prose before the
    JSON and break our balanced-brace extractor).
    """
    return {
        "type": "tool",
        "name": tool_name,
        "disable_parallel_tool_use": True,
    }


# ---------------------------------------------------------------------------
# $ref inlining
# ---------------------------------------------------------------------------


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve all ``$ref`` indirections against ``$defs`` and inline them.

    Pydantic emits ``{"$ref": "#/$defs/Foo"}`` for nested models. This
    walks the schema, replaces each ``$ref`` with the referenced
    definition, and returns a fully inlined copy. Mutually recursive
    schemas would loop forever; the dream-pass schemas don't have
    that shape (validated by tests).
    """
    defs = schema.get("$defs", {})
    return _resolve(schema, defs)


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            if ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                target = defs.get(key)
                if target is not None:
                    # Resolve nested refs in the target before returning.
                    return _resolve(target, defs)
            # Unknown ref form — leave as-is rather than fabricate.
            return node
        return {k: _resolve(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    return node
