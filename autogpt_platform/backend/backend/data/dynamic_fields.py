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


# --------------------------------------------------------------------------- #
#  Dynamic field parsing and merging utilities
# --------------------------------------------------------------------------- #


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


def parse_execution_output(output: tuple[str, Any], name: str) -> Any:
    """
    Retrieve a nested value out of `output` using the flattened *name*.

    On any failure (wrong name, wrong type, out-of-range, bad path)
    returns **None**.

    Args:
        output: Tuple of (base_name, data) representing a block output entry
        name: The flattened field name to extract from the output data

    Returns:
        The value at the specified path, or None if not found/invalid
    """
    base_name, data = output

    # Exact match → whole object
    if name == base_name:
        return data

    # Must start with the expected name
    if not name.startswith(base_name):
        return None
    path = name[len(base_name) :]
    if not path:
        return None  # nothing left to parse

    tokens = _tokenise(path)
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
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]

        elif delim == DICT_SPLIT:
            if not isinstance(cur, dict) or ident not in cur:
                return None
            cur = cur[ident]

        elif delim == OBJC_SPLIT:
            if not hasattr(cur, ident):
                return None
            cur = getattr(cur, ident)

        else:
            return None  # unreachable

    return cur


def _assign(container: Any, tokens: list[tuple[str, str]], value: Any) -> Any:
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
        try:
            idx = int(ident)
        except ValueError:
            raise ValueError("index must be an integer")

        if container is None:
            container = []
        elif not isinstance(container, list):
            container = list(container) if hasattr(container, "__iter__") else []

        while len(container) <= idx:
            container.append(None)
        container[idx] = _assign(container[idx], rest, value)
        return container

    # ---------- dict ----------
    if delim == DICT_SPLIT:
        if container is None:
            container = {}
        elif not isinstance(container, dict):
            container = dict(container) if hasattr(container, "items") else {}
        container[ident] = _assign(container.get(ident), rest, value)
        return container

    # ---------- object ----------
    if delim == OBJC_SPLIT:
        if container is None:
            container = MockObject()
        elif not hasattr(container, "__dict__"):
            # If it's not an object, create a new one
            container = MockObject()
        setattr(
            container,
            ident,
            _assign(getattr(container, ident, None), rest, value),
        )
        return container

    return value  # unreachable


def merge_execution_input(data: dict[str, Any]) -> dict[str, Any]:
    """
    Reconstruct nested objects from a *flattened* dict of key → value.

    Raises ValueError on syntactically invalid list indices.

    Args:
        data: Dictionary with potentially flattened dynamic field keys

    Returns:
        Dictionary with nested objects reconstructed from flattened keys
    """
    merged: dict[str, Any] = {}

    for key, value in data.items():
        # Split off the base name (before the first delimiter, if any)
        delim, pos = _next_delim(key)
        if delim is None:
            merged[key] = value
            continue

        base, path = key[:pos], key[pos:]
        tokens = _tokenise(path)
        if tokens is None:
            # Invalid key; treat as scalar under the raw name
            merged[key] = value
            continue

        merged[base] = _assign(merged.get(base), tokens, value)

    data.update(merged)
    return data
