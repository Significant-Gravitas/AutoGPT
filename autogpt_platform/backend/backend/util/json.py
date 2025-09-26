import json
from typing import Any, Type, TypeGuard, TypeVar, overload

import jsonschema
import orjson
from fastapi.encoders import jsonable_encoder
from prisma import Json
from pydantic import BaseModel

from .type import type_match


def to_dict(data) -> dict:
    if isinstance(data, BaseModel):
        data = data.model_dump()
    return jsonable_encoder(data)


def dumps(data: Any, *args: Any, **kwargs: Any) -> str:
    """
    Serialize data to JSON string with automatic conversion of Pydantic models and complex types.

    This function converts the input data to a JSON-serializable format using FastAPI's
    jsonable_encoder before dumping to JSON. It handles Pydantic models, complex types,
    and ensures proper serialization. Uses orjson for high performance.

    Parameters
    ----------
    data : Any
        The data to serialize. Can be any type including Pydantic models, dicts, lists, etc.
    *args : Any
        Additional positional arguments (ignored with orjson)
    **kwargs : Any
        Additional keyword arguments (limited support with orjson)

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

    # orjson is faster but has limited options support
    option = 0
    if kwargs.get("indent") is not None:
        option |= orjson.OPT_INDENT_2
    # orjson.dumps returns bytes, so we decode to str
    return orjson.dumps(serializable_data, option=option).decode("utf-8")


T = TypeVar("T")


@overload
def loads(data: str | bytes, *args, target_type: Type[T], **kwargs) -> T: ...


@overload
def loads(data: str | bytes, *args, **kwargs) -> Any: ...


def loads(
    data: str | bytes, *args, target_type: Type[T] | None = None, **kwargs
) -> Any:
    # orjson can handle both str and bytes directly
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


def _sanitize_null_bytes(data: Any) -> Any:
    """
    Recursively sanitize null bytes from data structures to prevent PostgreSQL 22P05 errors.
    PostgreSQL cannot store null bytes (\u0000) in text fields.
    """
    if isinstance(data, str):
        return data.replace("\u0000", "")
    elif isinstance(data, dict):
        return {key: _sanitize_null_bytes(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_sanitize_null_bytes(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_sanitize_null_bytes(item) for item in data)
    else:
        # For other types (int, float, bool, None, etc.), return as-is
        return data


def SafeJson(data: Any) -> Json:
    """
    Safely serialize data and return Prisma's Json type.
    Sanitizes null bytes to prevent PostgreSQL 22P05 errors.
    """
    if isinstance(_sanitize_null_bytes(data), BaseModel):
        return Json(
            _sanitize_null_bytes(data).model_dump(
                mode="json",
                warnings="error",
                exclude_none=True,
                fallback=lambda v: None,
            )
        )
    # Round-trip through JSON to ensure proper serialization with fallback for non-serializable values
    json_string = dumps(_sanitize_null_bytes(data), default=lambda v: None)
    return Json(json.loads(json_string))
