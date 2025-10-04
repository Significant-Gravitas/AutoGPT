"""
Utilities for handling dynamic field names and delimiters in the AutoGPT Platform.

Dynamic fields allow graphs to connect complex data structures using special delimiters:
- _#_ for dictionary keys (e.g., "values_#_name" → values["name"])
- _$_ for list indices (e.g., "items_$_0" → items[0])  
- _@_ for object attributes (e.g., "obj_@_attr" → obj.attr)

This module provides utilities for:
- Extracting base field names from dynamic field names
- Generating proper schemas for base fields
- Creating helper functions for field sanitization
"""

from backend.data.dynamic_fields import DICT_SPLIT, LIST_SPLIT, OBJC_SPLIT

# All dynamic field delimiters
DYNAMIC_DELIMITERS = (DICT_SPLIT, LIST_SPLIT, OBJC_SPLIT)


def extract_base_field_name(field_name: str) -> str:
    """
    Extract the base field name from a dynamic field name.

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


def get_dynamic_field_description(
    base_field_name: str, original_field_name: str
) -> str:
    """
    Generate a description for a dynamic field based on its base field and structure.

    Args:
        base_field_name: The base field name (e.g., "values")
        original_field_name: The full dynamic field name (e.g., "values_#_name")

    Returns:
        A descriptive string explaining what this dynamic field represents
    """
    if DICT_SPLIT in original_field_name:
        key_part = (
            original_field_name.split(DICT_SPLIT, 1)[1].split(DICT_SPLIT[0])[0]
            if DICT_SPLIT in original_field_name
            else "key"
        )
        return f"Dictionary value for {base_field_name}['{key_part}']"
    elif LIST_SPLIT in original_field_name:
        index_part = (
            original_field_name.split(LIST_SPLIT, 1)[1].split(LIST_SPLIT[0])[0]
            if LIST_SPLIT in original_field_name
            else "index"
        )
        return f"List item for {base_field_name}[{index_part}]"
    elif OBJC_SPLIT in original_field_name:
        attr_part = (
            original_field_name.split(OBJC_SPLIT, 1)[1].split(OBJC_SPLIT[0])[0]
            if OBJC_SPLIT in original_field_name
            else "attr"
        )
        return f"Object attribute for {base_field_name}.{attr_part}"
    else:
        return f"Dynamic value for {base_field_name}"


def group_fields_by_base_name(field_names: list[str]) -> dict[str, list[str]]:
    """
    Group a list of field names by their base field names.

    Args:
        field_names: List of field names that may contain dynamic delimiters

    Returns:
        Dictionary mapping base field names to lists of original field names

    Example:
        group_fields_by_base_name([
            "values_#_name",
            "values_#_age",
            "items_$_0",
            "regular_field"
        ])
        → {
            "values": ["values_#_name", "values_#_age"],
            "items": ["items_$_0"],
            "regular_field": ["regular_field"]
        }
    """
    grouped = {}
    for field_name in field_names:
        base_name = extract_base_field_name(field_name)
        if base_name not in grouped:
            grouped[base_name] = []
        grouped[base_name].append(field_name)
    return grouped
