import json
import logging
import signal
from contextlib import contextmanager
from typing import Any, Type, TypeGuard, TypeVar, overload

import jsonschema
import regex
from fastapi.encoders import jsonable_encoder
from prisma import Json
from pydantic import BaseModel

from .type import type_match

logger = logging.getLogger(__name__)


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


def SafeJson(data: Any) -> Json:
    """Safely serialize data and return Prisma's Json type."""
    if isinstance(data, BaseModel):
        return Json(
            data.model_dump(
                mode="json",
                warnings="error",
                exclude_none=True,
                fallback=lambda v: None,
            )
        )
    # Round-trip through JSON to ensure proper serialization with fallback for non-serializable values
    json_string = dumps(data, default=lambda v: None)
    return Json(json.loads(json_string))


# ================ PARSING ================ #


JSON_REGEX = regex.compile(
    r"""
    (?P<value>
        (?P<object> \{\s*
            (?:
                (?P<member>(?&string) \s*:\s* (?&value))
                ( \s*,\s* (?&member) )*
            )?
        \s*\})
        | (?P<array> \[\s*
            (
                (?&value)
                ( \s*,\s* (?&value) )*
            )?
        \s*\])
        | (?P<string>   " [^"\\]* (?: \\. | [^"\\]* )* ")
        | (?P<number>   (?P<integer> -? (?: 0 | [1-9][0-9]* ))
                        (?: \. [0-9]* )?
                        (?: [eE] [-+]? [0-9]+ )?
        )
        | true
        | false
        | null
    )
    """,
    flags=regex.VERBOSE | regex.UNICODE,
)

JSON_OBJECT_REGEX = regex.compile(
    r"""
    (?P<object> \{\s*
        (?:
            (?P<member>(?&string) \s*:\s*
                (?P<value>
                    (?&object)
                    | (?P<array>    \[\s* ((?&value) (\s*,\s* (?&value))*)? \s*\])
                    | (?P<string>   " [^"\\]* (?: \\. | [^"\\]* )* ")
                    | (?P<number>   (?P<integer> -? (?: 0 | [1-9][0-9]* ))
                                    (?: \. [0-9]* )?
                                    (?: [eE] [-+]? [0-9]+ )?
                    )
                    | true
                    | false
                    | null
                )
            )
            ( \s*,\s* (?&member) )*
        )?
    \s*\})
    """,
    flags=regex.VERBOSE | regex.UNICODE,
)


JSON_ARRAY_REGEX = regex.compile(
    r"""
    (?P<array> \[\s*
        (
            (?P<value>
                (?P<object> \{\s*
                    (?:
                        (?P<member>(?&string) \s*:\s* (?&value))
                        ( \s*,\s* (?&member) )*
                    )?
                \s*\})
                | (?&array)
                | (?P<string>   " [^"\\]* (?: \\. | [^"\\]* )* ")
                | (?P<number>   (?P<integer> -? (?: 0 | [1-9][0-9]* ))
                                (?: \. [0-9]* )?
                                (?: [eE] [-+]? [0-9]+ )?
                )
                | true
                | false
                | null
            )
            ( \s*,\s* (?&value) )*
        )?
    \s*\])
    """,
    flags=regex.VERBOSE | regex.UNICODE,
)


def find_objects_in_text(text: str, timeout_seconds: int = 5) -> list[str]:
    """Find all JSON objects in a text string with timeout protection."""
    try:
        with _regex_timeout(timeout_seconds):
            json_matches = JSON_OBJECT_REGEX.findall(text)
            return [match[0] for match in json_matches]
    except TimeoutError:
        logger.warning("Regex for finding JSON objects timed out")
        return []


def find_arrays_in_text(text: str, timeout_seconds: int = 5) -> list[str]:
    """Find all JSON arrays in a text string with timeout protection."""
    try:
        with _regex_timeout(timeout_seconds):
            json_matches = JSON_ARRAY_REGEX.findall(text)
            return [match[0] for match in json_matches]
    except TimeoutError:
        logger.warning("Regex for finding JSON arrays timed out")
        return []


@contextmanager
def _regex_timeout(seconds: int = 5):
    """Context manager to timeout regex operations that might hang."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Regex operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
