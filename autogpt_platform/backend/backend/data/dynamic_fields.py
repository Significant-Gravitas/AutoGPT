"""
Utilities for handling dynamic field names with special delimiters.

Dynamic fields allow graphs to connect complex data structures using special delimiters:
- _#_ for dictionary keys (e.g., "values_#_name" → values["name"])
- _$_ for list indices (e.g., "items_$_0" → items[0])  
- _@_ for object attributes (e.g., "obj_@_attr" → obj.attr)
"""

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


def sanitize_field_name(field_name: str) -> str:
    """
    Remove all dynamic field suffixes from a field name.
    This is an alias for extract_base_field_name but with clearer intent for sanitization.

    Args:
        field_name: The field name to sanitize

    Returns:
        The field name with all dynamic suffixes removed
    """
    return extract_base_field_name(field_name)
