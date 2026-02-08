"""Shared helpers for chat tools."""

from typing import Any

from .models import ErrorResponse


def error_response(
    message: str, session_id: str | None, **kwargs: Any
) -> ErrorResponse:
    """Create standardized error response.

    Args:
        message: Error message to display
        session_id: Current session ID
        **kwargs: Additional fields to pass to ErrorResponse

    Returns:
        ErrorResponse with the given message and session_id
    """
    return ErrorResponse(message=message, session_id=session_id, **kwargs)


def get_inputs_from_schema(
    input_schema: dict[str, Any],
    exclude_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract input field info from JSON schema.

    Args:
        input_schema: JSON schema dict with 'properties' and 'required'
        exclude_fields: Set of field names to exclude (e.g., credential fields)

    Returns:
        List of dicts with field info (name, title, type, description, required, default)
    """
    exclude = exclude_fields or set()
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    return [
        {
            "name": name,
            "title": schema.get("title", name),
            "type": schema.get("type", "string"),
            "description": schema.get("description", ""),
            "required": name in required,
            "default": schema.get("default"),
        }
        for name, schema in properties.items()
        if name not in exclude
    ]


def format_inputs_as_markdown(inputs: list[dict[str, Any]]) -> str:
    """Format input fields as a readable markdown list.

    Args:
        inputs: List of input dicts from get_inputs_from_schema

    Returns:
        Markdown-formatted string listing the inputs
    """
    if not inputs:
        return "No inputs required."

    lines = []
    for inp in inputs:
        required_marker = " (required)" if inp.get("required") else ""
        default = inp.get("default")
        default_info = f" [default: {default}]" if default is not None else ""
        description = inp.get("description", "")
        desc_info = f" - {description}" if description else ""

        lines.append(f"- **{inp['name']}**{required_marker}{default_info}{desc_info}")

    return "\n".join(lines)
