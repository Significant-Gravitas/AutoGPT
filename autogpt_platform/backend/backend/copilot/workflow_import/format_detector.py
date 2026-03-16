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
    if _is_n8n(json_data):
        return SourcePlatform.N8N
    if _is_make(json_data):
        return SourcePlatform.MAKE
    if _is_zapier(json_data):
        return SourcePlatform.ZAPIER
    return SourcePlatform.UNKNOWN


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


def _is_zapier(data: dict[str, Any]) -> bool:
    """Zapier Zaps have a `steps` array with items containing `app` and
    `action` fields."""
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    return any(
        isinstance(step, dict) and "app" in step and "action" in step for step in steps
    )
