import json
from typing import Any, Type, TypeGuard, TypeVar, overload

import jsonschema
from fastapi.encoders import jsonable_encoder
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
    and ensures proper serialization.

    Parameters
    ----------
    data : Any
        The data to serialize. Can be any type including Pydantic models, dicts, lists, etc.
    *args : Any
        Additional positional arguments passed to json.dumps()
    **kwargs : Any
        Additional keyword arguments passed to json.dumps() (e.g., indent, separators)

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
    return json.dumps(to_dict(data), *args, **kwargs)


T = TypeVar("T")


@overload
def loads(data: str | bytes, *args, target_type: Type[T], **kwargs) -> T: ...


@overload
def loads(data: str | bytes, *args, **kwargs) -> Any: ...


def loads(
    data: str | bytes, *args, target_type: Type[T] | None = None, **kwargs
) -> Any:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    parsed = json.loads(data, *args, **kwargs)
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
