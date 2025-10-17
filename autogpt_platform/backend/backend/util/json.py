import logging
import re
from typing import Any, Type, TypeVar, overload

import jsonschema
import orjson
from fastapi.encoders import jsonable_encoder as to_dict
from prisma import Json

from .truncate import truncate
from .type import type_match

logger = logging.getLogger(__name__)

# Precompiled regex to remove PostgreSQL-incompatible control characters
# Removes \u0000-\u0008, \u000B-\u000C, \u000E-\u001F, \u007F (keeps tab \u0009, newline \u000A, carriage return \u000D)
POSTGRES_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]")


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


def _sanitize_string(value: str) -> str:
    """Remove PostgreSQL-incompatible control characters from string."""
    return POSTGRES_CONTROL_CHARS.sub("", value)


def sanitize_json(data: Any) -> Any:
    try:
        # Use two-pass approach for consistent string sanitization:
        # 1. First convert to basic JSON-serializable types (handles Pydantic models)
        # 2. Then sanitize strings in the result
        basic_result = to_dict(data)
        return to_dict(basic_result, custom_encoder={str: _sanitize_string})
    except Exception as e:
        # Log the failure and fall back to string representation
        logger.error(
            "SafeJson fallback to string representation due to serialization error: %s (%s). "
            "Data type: %s, Data preview: %s",
            type(e).__name__,
            truncate(str(e), 200),
            type(data).__name__,
            truncate(str(data), 100),
        )

        # Ultimate fallback: convert to string representation and sanitize
        return _sanitize_string(str(data))


class SafeJson(Json):
    """
    Safely serialize data and return Prisma's Json type.
    Sanitizes control characters to prevent PostgreSQL 22P05 errors.

    This function:
    1. Converts Pydantic models to dicts (recursively using to_dict)
    2. Recursively removes PostgreSQL-incompatible control characters from strings
    3. Returns a Prisma Json object safe for database storage

    Uses to_dict (jsonable_encoder) with a custom encoder to handle both Pydantic
    conversion and control character sanitization in a two-pass approach.

    Args:
        data: Input data to sanitize and convert to Json

    Returns:
        Prisma Json object with control characters removed

    Examples:
        >>> SafeJson({"text": "Hello\\x00World"})  # null char removed
        >>> SafeJson({"path": "C:\\\\temp"})  # backslashes preserved
        >>> SafeJson({"data": "Text\\\\u0000here"})  # literal backslash-u preserved
    """

    def __init__(self, data: Any):
        super().__init__(sanitize_json(data))
