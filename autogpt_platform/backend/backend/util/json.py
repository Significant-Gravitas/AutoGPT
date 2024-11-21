import json
from typing import Any, Type, TypeVar, overload

import jsonschema
from fastapi.encoders import jsonable_encoder

from .type import type_match


def to_dict(data) -> dict:
    return jsonable_encoder(data)


def dumps(data) -> str:
    return json.dumps(jsonable_encoder(data))


T = TypeVar("T")


@overload
def loads(data: str, *args, target_type: Type[T], **kwargs) -> T: ...


@overload
def loads(data: str, *args, **kwargs) -> Any: ...


def loads(data: str, *args, target_type: Type[T] | None = None, **kwargs) -> Any:
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
