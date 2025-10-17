import json
import re
from typing import Any, Type, TypeGuard, TypeVar, overload

import jsonschema
import orjson
from fastapi.encoders import jsonable_encoder
from prisma import Json
from pydantic import BaseModel

from .type import type_match

__all__ = [
    "json",
    "dumps",
    "loads",
    "validate_with_jsonschema",
    "SafeJson",
    "convert_pydantic_to_json",
]

# Precompiled regex to remove PostgreSQL-incompatible control characters
# Removes \u0000-\u0008, \u000B-\u000C, \u000E-\u001F, \u007F (keeps tab \u0009, newline \u000A, carriage return \u000D)
POSTGRES_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]")


def to_dict(data) -> dict:
    if isinstance(data, BaseModel):
        data = data.model_dump()
    return jsonable_encoder(data)


def dumps(
    data: Any, *args: Any, indent: int | None = None, option: int = 0, **kwargs: Any
) -> str:
    """
    Serialize data to JSON string with automatic conversion of Pydantic models and complex types.

    This function converts the input data to a JSON-serializable format using FastAPI's
    jsonable_encoder before dumping to JSON. It handles Pydantic models, complex types,
    and ensures proper serialization.

    Parameters
    ----------
    data : Any
        The data to serialize. Can be any type including Pydantic models, dicts, lists, etc.
    *args : Any
        Additional positional arguments
    indent : int | None
        If not None, pretty-print with indentation
    option : int
        orjson option flags (default: 0)
    **kwargs : Any
        Additional keyword arguments. Supported: default, ensure_ascii, separators, indent

    Returns
    -------
    str
        JSON string representation of the data

    Examples
    --------
    >>> dumps({"name": "Alice", "age": 30})
    '{"name": "Alice", "age": 30}'

    >>> dumps(pydantic_model_instance, indent=2)
    '{\n  "field1": "value1",\n  "field2": "value2"\n}'
    """
    serializable_data = to_dict(data)

    # Handle indent parameter
    if indent is not None or kwargs.get("indent") is not None:
        option |= orjson.OPT_INDENT_2

    # orjson only accepts specific parameters, filter out stdlib json params
    # ensure_ascii: orjson always produces UTF-8 (better than ASCII)
    # separators: orjson uses compact separators by default
    supported_orjson_params = {"default"}
    orjson_kwargs = {k: v for k, v in kwargs.items() if k in supported_orjson_params}

    return orjson.dumps(serializable_data, option=option, **orjson_kwargs).decode(
        "utf-8"
    )


T = TypeVar("T")


@overload
def loads(data: str | bytes, *args, target_type: Type[T], **kwargs) -> T: ...


@overload
def loads(data: str | bytes, *args, **kwargs) -> Any: ...


def loads(
    data: str | bytes, *args, target_type: Type[T] | None = None, **kwargs
) -> Any:
    parsed = orjson.loads(data)

    if target_type:
        return type_match(parsed, target_type)
    return parsed


def validate_with_jsonschema(
    schema: dict[str, Any], data: dict[str, Any]
) -> str | None:
    """
    Validate the data against the schema.
    Returns the validation error message if the data does not match the schema.
    """
    try:
        jsonschema.validate(data, schema)
        return None
    except jsonschema.ValidationError as e:
        return str(e)


def is_list_of_basemodels(value: object) -> TypeGuard[list[BaseModel]]:
    return isinstance(value, list) and all(
        isinstance(item, BaseModel) for item in value
    )


def convert_pydantic_to_json(output_data: Any) -> Any:
    if isinstance(output_data, BaseModel):
        return output_data.model_dump()
    if is_list_of_basemodels(output_data):
        return [item.model_dump() for item in output_data]
    return output_data


def _sanitize_value(value: Any) -> Any:
    """
    Recursively sanitize values by removing PostgreSQL-incompatible control characters.

    This function walks through data structures and removes control characters from strings.
    It handles:
    - Strings: Remove control chars directly from the string
    - Lists: Recursively sanitize each element
    - Dicts: Recursively sanitize keys and values
    - Other types: Return as-is

    Args:
        value: The value to sanitize

    Returns:
        Sanitized version of the value with control characters removed
    """
    if isinstance(value, str):
        # Remove control characters directly from the string
        return POSTGRES_CONTROL_CHARS.sub("", value)
    elif isinstance(value, dict):
        # Recursively sanitize dictionary keys and values
        return {_sanitize_value(k): _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively sanitize list elements
        return [_sanitize_value(item) for item in value]
    elif isinstance(value, tuple):
        # Recursively sanitize tuple elements
        return tuple(_sanitize_value(item) for item in value)
    else:
        # For other types (int, float, bool, None, etc.), return as-is
        return value


def SafeJson(data: Any) -> Json:
    """
    Safely serialize data and return Prisma's Json type.
    Sanitizes control characters to prevent PostgreSQL 22P05 errors.

    This function:
    1. Converts Pydantic models to dicts
    2. Recursively removes PostgreSQL-incompatible control characters from strings
    3. Returns a Prisma Json object safe for database storage

    Args:
        data: Input data to sanitize and convert to Json

    Returns:
        Prisma Json object with control characters removed

    Examples:
        >>> SafeJson({"text": "Hello\\x00World"})  # null char removed
        >>> SafeJson({"path": "C:\\\\temp"})  # backslashes preserved
        >>> SafeJson({"data": "Text\\\\u0000here"})  # literal backslash-u preserved
    """
    # Convert Pydantic models to dict first
    if isinstance(data, BaseModel):
        data = data.model_dump(exclude_none=True)

    # Sanitize the data structure by removing control characters
    sanitized_data = _sanitize_value(data)

    # Return as Prisma Json type
    return Json(sanitized_data)
