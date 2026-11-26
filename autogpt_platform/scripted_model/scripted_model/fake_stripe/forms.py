"""Stripe's request encoding: form fields with bracketed keys.

The SDK sends ``line_items[0][price_data][unit_amount]=500`` and
``expand[0]=payment_intent``, in a POST's body or a GET's query. ``decode``
turns them back into the nested dicts and lists they were. Values stay
strings, as Stripe receives them.
"""

from typing import Any
from urllib.parse import parse_qsl


def decode(encoded: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for key, value in parse_qsl(encoded, keep_blank_values=True):
        head, _, rest = key.partition("[")
        parts = [head, *rest.removesuffix("]").split("][")] if rest else [head]
        _assign(root, parts, value)
    listed = _listify(root)
    return listed if isinstance(listed, dict) else {}


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _assign(node: dict[str, Any], parts: list[str], value: str) -> None:
    *path, last = parts
    for part in path:
        child = node.get(part)
        if not isinstance(child, dict):
            child = node[part] = {}
        node = child
    # ``key[]`` appends, as curl and other clients send lists.
    node[last or str(len(node))] = value


def _listify(value: Any) -> Any:
    """Dicts keyed 0, 1, 2... were lists."""
    if not isinstance(value, dict):
        return value
    items = {key: _listify(child) for key, child in value.items()}
    if items and all(key.isdigit() for key in items):
        return [items[key] for key in sorted(items, key=int)]
    return items
