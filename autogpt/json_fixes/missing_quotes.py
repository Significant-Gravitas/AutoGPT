"""Fix quotes in a JSON string."""
import json
import re


def add_quotes_to_property_names(json_string: str) -> str:
    """
    Add quotes to property names in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with quotes added to property names.
    """

    def replace_func(match: re.Match) -> str:
        return f'"{match[1]}":'

    property_name_pattern = re.compile(r"(\w+):")
    corrected_json_string = property_name_pattern.sub(replace_func, json_string)

    try:
        json.loads(corrected_json_string)
        return corrected_json_string
    except json.JSONDecodeError as e:
        raise e
