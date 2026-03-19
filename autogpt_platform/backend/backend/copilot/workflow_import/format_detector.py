"""Detect the source platform of a workflow JSON."""

import re
from typing import Any

from .models import SourcePlatform

_N8N_TYPE_RE = re.compile(r"^(n8n-nodes-base\.|@n8n/)")


def detect_format(json_data: dict[str, Any]) -> SourcePlatform:
    """Inspect a workflow JSON and determine which platform it came from.

    Args:
        json_data: The parsed JSON data from a workflow export file.

    Returns:
        The detected SourcePlatform.
    """
    # Zapier's "export all Zaps" (Zapfile.json) and "Powered by Zapier" API both
    # wrap results in {"data": [...]}. Unwrap to the first Zap for detection.
    candidate = unwrap_zapier_envelope(json_data) or json_data

    if _is_n8n(candidate):
        return SourcePlatform.N8N
    if _is_make(candidate):
        return SourcePlatform.MAKE
    if _is_zapier(candidate):
        return SourcePlatform.ZAPIER
    return SourcePlatform.UNKNOWN


def unwrap_zapier_envelope(data: dict[str, Any]) -> "dict[str, Any] | None":
    """Return the first Zap from a Zapier {'data': [...]} envelope, or None."""
    data_list = data.get("data")
    if isinstance(data_list, list) and data_list and isinstance(data_list[0], dict):
        return data_list[0]
    return None


def _is_n8n(data: dict[str, Any]) -> bool:
    """n8n workflows have a `nodes` array with items containing `type` fields
    matching patterns like `n8n-nodes-base.*` or `@n8n/*`, plus a `connections`
    object."""
    nodes = data.get("nodes")
    connections = data.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        return False
    if not nodes:
        return False
    # Check if at least one node has an n8n-style type
    return any(
        isinstance(n, dict)
        and isinstance(n.get("type"), str)
        and _N8N_TYPE_RE.match(n["type"])
        for n in nodes
    )


def _is_make(data: dict[str, Any]) -> bool:
    """Make.com scenarios have a `flow` array with items containing `module`
    fields in `service:action` URI format."""
    flow = data.get("flow")
    if not isinstance(flow, list) or not flow:
        return False
    # Check if at least one module has `service:action` pattern
    return any(
        isinstance(item, dict)
        and isinstance(item.get("module"), str)
        and ":" in item["module"]
        for item in flow
    )


_ZAPIER_ACTION_RE = re.compile(r"^(core:|uag:)")


def _is_zapier(data: dict[str, Any]) -> bool:
    """Zapier Zaps have a `steps` array. Two export shapes are supported:

    1. Old/community format: steps contain `app` + `action` string fields.
    2. Zapier API / Zapfile.json format: steps contain an `action` field with a
       `core:<id>` or `uag:<uuid>` prefix, plus optional `title`, `inputs`,
       `authentication` fields (no `app` key).
    """
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, str):
            continue
        # Old format: app field present
        if "app" in step:
            return True
        # API/export format: action prefixed with core: or uag:
        if _ZAPIER_ACTION_RE.match(action):
            return True
    return False
