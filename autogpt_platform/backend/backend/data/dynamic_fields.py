"""
Utilities for handling dynamic field names with special delimiters.

Dynamic fields allow graphs to connect complex data structures using special delimiters:
- _#_ for dictionary keys (e.g., "values_#_name" → values["name"])
- _$_ for list indices (e.g., "items_$_0" → items[0])
- _@_ for object attributes (e.g., "obj_@_attr" → obj.attr)
"""

from typing import Any

from backend.util.mock import MockObject

# Dynamic field delimiters
LIST_SPLIT = "_$_"
DICT_SPLIT = "_#_"
OBJC_SPLIT = "_@_"

DYNAMIC_DELIMITERS = (LIST_SPLIT, DICT_SPLIT, OBJC_SPLIT)


def extract_base_field_name(field_name: str) -> str:
    """
    Extract the base field name from a dynamic field name by removing all dynamic suffixes.

    Examples:
        extract_base_field_name("values_#_name") → "values"
        extract_base_field_name("items_$_0") → "items"
        extract_base_field_name("obj_@_attr") → "obj"
        extract_base_field_name("regular_field") → "regular_field"

    Args:
        field_name: The field name that may contain dynamic delimiters

    Returns:
        The base field name without any dynamic suffixes
    """
    base_name = field_name
    for delimiter in DYNAMIC_DELIMITERS:
        if delimiter in base_name:
            base_name = base_name.split(delimiter)[0]
    return base_name


def is_dynamic_field(field_name: str) -> bool:
    """
    Check if a field name contains dynamic delimiters.

    Args:
        field_name: The field name to check

    Returns:
        True if the field contains any dynamic delimiters, False otherwise
    """
    return any(delimiter in field_name for delimiter in DYNAMIC_DELIMITERS)


def get_dynamic_field_description(field_name: str) -> str:
    """
    Generate a description for a dynamic field based on its structure.

    Args:
        field_name: The full dynamic field name (e.g., "values_#_name")

    Returns:
        A descriptive string explaining what this dynamic field represents
    """
    base_name = extract_base_field_name(field_name)

    if DICT_SPLIT in field_name:
        # Extract the key part after _#_
        parts = field_name.split(DICT_SPLIT)
        if len(parts) > 1:
            key = parts[1].split("_")[0] if "_" in parts[1] else parts[1]
            return f"Dictionary field '{key}' for base field '{base_name}' ({base_name}['{key}'])"
    elif LIST_SPLIT in field_name:
        # Extract the index part after _$_
        parts = field_name.split(LIST_SPLIT)
        if len(parts) > 1:
            index = parts[1].split("_")[0] if "_" in parts[1] else parts[1]
            return (
                f"List item {index} for base field '{base_name}' ({base_name}[{index}])"
            )
    elif OBJC_SPLIT in field_name:
        # Extract the attribute part after _@_
        parts = field_name.split(OBJC_SPLIT)
        if len(parts) > 1:
            # Get the full attribute name (everything after _@_)
            attr = parts[1]
            return f"Object attribute '{attr}' for base field '{base_name}' ({base_name}.{attr})"

    return f"Value for {field_name}"


def is_tool_pin(name: str) -> bool:
    """Check if a pin name represents a tool connection."""
    return name.startswith("tools_^_") or name == "tools"


def sanitize_pin_name(name: str) -> str:
    sanitized_name = extract_base_field_name(name)
    if is_tool_pin(sanitized_name):
        return "tools"
    return sanitized_name


# --------------------------------------------------------------------------- #
#  Dynamic field parsing and merging utilities
# --------------------------------------------------------------------------- #


def assert_safe_pin_name(name: str) -> None:
    """
    Raise ValueError if a (possibly flattened) pin name encodes an unsafe
    dynamic path. Called where a pin name is stored so one never persists, and
    a backstop for the use-site guards in `_assign`/`parse_execution_output`.
    """
    delim, pos = _next_delim(name)
    if delim is None:
        return  # plain field name, nothing dynamic to check
    tokens = _tokenise_bounded(name[pos:])
    if tokens is None:
        return  # not a valid dynamic path; stored/handled as a raw scalar key
    for d, ident in tokens:
        if d == OBJC_SPLIT and not is_safe_object_attribute(ident):
            raise ValueError(f"unsafe object attribute in pin name {name!r}")
        if d == LIST_SPLIT:
            _parse_list_index(ident)


def _next_delim(s: str) -> tuple[str | None, int]:
    """
    Return the *earliest* delimiter appearing in `s` and its index.

    If none present → (None, -1).
    """
    first: str | None = None
    pos = len(s)  # sentinel: larger than any real index
    for d in DYNAMIC_DELIMITERS:
        i = s.find(d)
        if 0 <= i < pos:
            first, pos = d, i
    return first, (pos if first else -1)


def _tokenise(path: str) -> list[tuple[str, str]] | None:
    """
    Convert the raw path string (starting with a delimiter) into
    [ (delimiter, identifier), … ] or None if the syntax is malformed.
    """
    tokens: list[tuple[str, str]] = []
    while path:
        # 1. Which delimiter starts this chunk?
        delim = next((d for d in DYNAMIC_DELIMITERS if path.startswith(d)), None)
        if delim is None:
            return None  # invalid syntax

        # 2. Slice off the delimiter, then up to the next delimiter (or EOS)
        path = path[len(delim) :]
        nxt_delim, pos = _next_delim(path)
        token, path = (
            path[: pos if pos != -1 else len(path)],
            path[pos if pos != -1 else len(path) :],
        )
        if token == "":
            return None  # empty identifier is invalid
        tokens.append((delim, token))
    return tokens


def parse_execution_output(
    output_item: tuple[str, Any],
    link_output_selector: str,
    sink_node_id: str | None = None,
    sink_pin_name: str | None = None,
) -> Any:
    """
    Retrieve a nested value out of `output` using the flattened `link_output_selector`.

    On any failure (wrong name, wrong type, out-of-range, bad path) returns **None**.

    ### Special Case: Tool pins
    For regular output pins, the `output_item`'s name will simply be the field name, and
    `link_output_selector` (= the `source_name` of the link) may provide a "selector"
    used to extract part of the output value and route it through the link
    to the next node.

    However, for tool pins, it is the other way around: the `output_item`'s name
    provides the routing information (`tools_^_{sink_node_id}_~_{field_name}`),
    and the `link_output_selector` is simply `"tools"`
    (or `"tools_^_{tool_name}_~_{field_name}"` for backward compatibility).

    Args:
        output_item: Tuple of (base_name, data) representing a block output entry.
        link_output_selector: The flattened field name to extract from the output data.
        sink_node_id: Sink node ID, used for tool use routing.
        sink_pin_name: Sink pin name, used for tool use routing.

    Returns:
        The value at the specified path, or `None` if not found/invalid.
    """
    output_pin_name, data = output_item

    # Special handling for tool pins
    if is_tool_pin(link_output_selector) and (  # "tools" or "tools_^_…"
        output_pin_name.startswith("tools_^_") and "_~_" in output_pin_name
    ):
        if not (sink_node_id and sink_pin_name):
            raise ValueError(
                "sink_node_id and sink_pin_name must be provided for tool pin routing"
            )

        # Extract routing information from emit key: tools_^_{node_id}_~_{field}
        selector = output_pin_name[8:]  # Remove "tools_^_" prefix
        target_node_id, target_input_pin = selector.split("_~_", 1)
        if target_node_id == sink_node_id and target_input_pin == sink_pin_name:
            return data
        else:
            return None

    # Exact match → whole object
    if link_output_selector == output_pin_name:
        return data

    # Must start with the expected name
    if not link_output_selector.startswith(output_pin_name):
        return None
    path = link_output_selector[len(output_pin_name) :]
    if not path:
        return None  # nothing left to parse

    try:
        tokens = _tokenise_bounded(path)
    except ValueError:
        return None
    if tokens is None:
        return None

    cur: Any = data
    for delim, ident in tokens:
        if delim == LIST_SPLIT:
            # list[index]
            try:
                idx = int(ident)
            except ValueError:
                return None
            if idx < 0 or not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]

        elif delim == DICT_SPLIT:
            if not isinstance(cur, dict) or ident not in cur:
                return None
            cur = cur[ident]

        elif delim == OBJC_SPLIT:
            # Instance data only: public names like `gi_frame` lead off the value.
            attrs = getattr(cur, "__dict__", {})
            if not is_safe_object_attribute(ident) or ident not in attrs:
                return None
            cur = attrs[ident]

        else:
            return None  # unreachable

    return cur


def _assign(
    container: Any,
    tokens: list[tuple[str, str]],
    value: Any,
    padding: "_PaddingBudget",
) -> Any:
    """
    Recursive helper that *returns* the (possibly new) container with
    `value` assigned along the remaining `tokens` path.
    """
    if not tokens:
        return value  # leaf reached

    delim, ident = tokens[0]
    rest = tokens[1:]

    # ---------- list ----------
    if delim == LIST_SPLIT:
        idx = _parse_list_index(ident)

        if container is None:
            container = []
        elif not isinstance(container, list):
            container = list(container) if hasattr(container, "__iter__") else []

        padding.spend(idx + 1 - len(container))
        while len(container) <= idx:
            container.append(None)
        container[idx] = _assign(container[idx], rest, value, padding)
        return container

    # ---------- dict ----------
    if delim == DICT_SPLIT:
        if container is None:
            container = {}
        elif not isinstance(container, dict):
            container = dict(container) if hasattr(container, "items") else {}
        container[ident] = _assign(container.get(ident), rest, value, padding)
        return container

    # ---------- object ----------
    if delim == OBJC_SPLIT:
        if not is_safe_object_attribute(ident):
            raise ValueError(f"unsafe object attribute {ident!r}")
        if container is None:
            container = MockObject()
        elif not hasattr(container, "__dict__"):
            # If it's not an object, create a new one
            container = MockObject()
        setattr(
            container,
            ident,
            _assign(getattr(container, ident, None), rest, value, padding),
        )
        return container

    return value  # unreachable


def merge_execution_input(data: dict[str, Any]) -> dict[str, Any]:
    """
    Reconstruct nested objects from a *flattened* dict of key → value.

    Raises ValueError on an unsafe or oversized dynamic path.

    Args:
        data: Dictionary with potentially flattened dynamic field keys

    Returns:
        Dictionary with nested objects reconstructed from flattened keys
    """
    merged: dict[str, Any] = {}
    padding = _PaddingBudget()

    for key, value in data.items():
        # Split off the base name (before the first delimiter, if any)
        delim, pos = _next_delim(key)
        if delim is None:
            merged[key] = value
            continue

        base, path = key[:pos], key[pos:]
        tokens = _tokenise_bounded(path)
        if tokens is None:
            # Invalid key; treat as scalar under the raw name
            merged[key] = value
            continue

        merged[base] = _assign(merged.get(base), tokens, value, padding)

    data.update(merged)
    return data


# An _@_ token is used directly as a setattr/getattr target, so it must be a
# plain public identifier; a dunder or leading-underscore name would let a
# flattened pin walk into an object's internals.
def is_safe_object_attribute(ident: str) -> bool:
    return ident.isidentifier() and not ident.startswith("_")


# A _$_ token pads a list up to its index, so both one index and all the padding
# one merge does are capped; no builder produces a legitimate pin near either.
MAX_DYNAMIC_LIST_INDEX = 10_000
MAX_DYNAMIC_LIST_PADDING = MAX_DYNAMIC_LIST_INDEX + 1

# Tokenising is quadratic in the path's length and each token is a level of
# recursion in _assign, so a path's size is capped before it is parsed.
MAX_DYNAMIC_PATH_LENGTH = 1024
MAX_DYNAMIC_PATH_TOKENS = 32


def _parse_list_index(ident: str) -> int:
    try:
        idx = int(ident)
    except ValueError:
        raise ValueError("index must be an integer")
    if idx < 0 or idx > MAX_DYNAMIC_LIST_INDEX:
        raise ValueError(f"list index out of range (0..{MAX_DYNAMIC_LIST_INDEX})")
    return idx


def _tokenise_bounded(path: str) -> list[tuple[str, str]] | None:
    if len(path) > MAX_DYNAMIC_PATH_LENGTH:
        raise ValueError(
            f"dynamic pin path longer than {MAX_DYNAMIC_PATH_LENGTH} characters"
        )
    tokens = _tokenise(path)
    if tokens is not None and len(tokens) > MAX_DYNAMIC_PATH_TOKENS:
        raise ValueError(
            f"dynamic pin path deeper than {MAX_DYNAMIC_PATH_TOKENS} levels"
        )
    return tokens


class _PaddingBudget:
    """List slots one merge_execution_input call may still pad, across all keys."""

    def __init__(self) -> None:
        self.left = MAX_DYNAMIC_LIST_PADDING

    def spend(self, slots: int) -> None:
        if slots <= 0:
            return
        if slots > self.left:
            raise ValueError(
                f"dynamic list padding exceeds {MAX_DYNAMIC_LIST_PADDING} slots"
            )
        self.left -= slots
