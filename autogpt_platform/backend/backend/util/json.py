import itertools
import logging
import math
import re
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from threading import Lock
from typing import Any, Type, TypeVar, overload

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
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

# Sentinel value to detect when fallback is not provided
_NO_FALLBACK = object()


@overload
def loads(
    data: str | bytes, *args, target_type: Type[T], fallback: T | None = None, **kwargs
) -> T:
    pass


@overload
def loads(data: str | bytes, *args, fallback: Any = None, **kwargs) -> Any:
    pass


def loads(
    data: str | bytes,
    *args,
    target_type: Type[T] | None = None,
    fallback: Any = _NO_FALLBACK,
    **kwargs,
) -> Any:
    """Parse JSON with optional fallback on decode errors.

    Args:
        data: JSON string or bytes to parse
        target_type: Optional type to validate/cast result to
        fallback: Value to return on JSONDecodeError. If not provided, raises.
        **kwargs: Additional arguments (unused, for compatibility)

    Returns:
        Parsed JSON data, or fallback value if parsing fails

    Raises:
        orjson.JSONDecodeError: Only if fallback is not provided

    Examples:
        >>> loads('{"valid": "json"}')
        {'valid': 'json'}
        >>> loads('invalid json', fallback=None)
        None
        >>> loads('invalid json', fallback={})
        {}
        >>> loads('invalid json')  # raises orjson.JSONDecodeError
    """
    try:
        parsed = orjson.loads(data)
    except orjson.JSONDecodeError:
        if fallback is not _NO_FALLBACK:
            return fallback
        raise

    if target_type:
        return type_match(parsed, target_type)
    return parsed


_SchemaTypeFingerprint = tuple[tuple[int, type[Any]], ...]
_SchemaCacheKey = tuple[bytes, _SchemaTypeFingerprint]

_VALIDATOR_CACHE: OrderedDict[_SchemaCacheKey, Any] = OrderedDict()
# The 565 built-in Block classes can each contribute a normal and dry-run
# schema; 2,048 covers that working set while leaving room for user schemas.
_VALIDATOR_CACHE_MAX_ENTRIES = 2048
_VALIDATOR_CACHE_MAX_KEY_BYTES = 32 * 1024
_VALIDATOR_CACHE_LOCK = Lock()


def _compiled_validator(schema: dict[str, Any]):
    """Return a validator for `schema`, compiled once per distinct schema.

    `jsonschema.validate()` re-resolves the validator class and re-validates the
    schema against its meta-schema on every single call; both depend only on the
    schema, so both are cached here.

    The key is the schema's serialized content, not its identity: a graph's
    input schema is rebuilt as a fresh dict on every node execution, so identity
    keying would never hit and would pin every schema object the process has
    ever seen. Key order is deliberately *not* normalised -- the error strings
    this function returns embed a repr of the failing sub-schema, so reordering
    keys would change them.

    """
    key = _schema_cache_key(schema)
    if key is None:
        return _new_validator(schema)

    with _VALIDATOR_CACHE_LOCK:
        cached = _VALIDATOR_CACHE.get(key)
        if cached is not None:
            _VALIDATOR_CACHE.move_to_end(key)
            return cached

    return _remember(key, _new_validator(schema))


def _schema_cache_key(schema: dict[str, Any]) -> _SchemaCacheKey | None:
    """Return an injective, size-bounded key for safely cacheable schemas."""
    try:
        key = orjson.dumps(schema)
    except TypeError:
        return None
    if len(key) > _VALIDATOR_CACHE_MAX_KEY_BYTES:
        return None
    type_fingerprint = _schema_type_fingerprint(schema)
    if type_fingerprint is None:
        return None
    return key, type_fingerprint


def _schema_type_fingerprint(value: Any) -> _SchemaTypeFingerprint | None:
    """Describe string subclasses while rejecting unsafe serialization."""
    pending = [value]
    seen_containers: set[int] = set()
    string_subclasses: list[tuple[int, type[Any]]] = []
    string_position = 0
    while pending:
        current = pending.pop()
        current_type = type(current)
        if current is None or current_type in (int, bool):
            continue
        if issubclass(current_type, str):
            if current_type is not str:
                if not issubclass(current_type, Enum):
                    return None
                string_subclasses.append((string_position, current_type))
            string_position += 1
            continue
        if current_type is float:
            if not math.isfinite(current):
                return None
            continue
        if current_type not in (list, dict):
            return None
        identity = id(current)
        if identity in seen_containers:
            return None
        seen_containers.add(identity)
        if current_type is dict:
            # orjson.dumps rejects non-exact-string keys before this walk.
            pending.extend(reversed(tuple(current.values())))
        else:
            pending.extend(reversed(current))
    return tuple(string_subclasses)


def _new_validator(schema: dict[str, Any]):
    # Compile against a private copy: a jsonschema validator memoises the
    # sub-schemas it walks, so a retained one must not alias a dict the caller
    # still holds and could mutate afterwards. deepcopy also preserves key
    # order, which the error messages depend on.
    schema_copy = deepcopy(schema)
    validator_cls = jsonschema.validators.validator_for(schema_copy)
    validator_cls.check_schema(schema_copy)
    return validator_cls(schema_copy)


def _remember(key: _SchemaCacheKey, value: Any) -> Any:
    """Publish and return one value, evicting the least-recently used entry.

    The keys are caller-supplied schemas, which on this platform come from
    user-authored graphs, so the cache has to be bounded.
    """
    with _VALIDATOR_CACHE_LOCK:
        cached = _VALIDATOR_CACHE.get(key)
        if cached is not None:
            _VALIDATOR_CACHE.move_to_end(key)
            return cached
        if len(_VALIDATOR_CACHE) >= _VALIDATOR_CACHE_MAX_ENTRIES:
            _VALIDATOR_CACHE.popitem(last=False)
        _VALIDATOR_CACHE[key] = value
        return value


def validate_with_jsonschema(
    schema: dict[str, Any], data: dict[str, Any]
) -> str | None:
    """
    Validate the data against the schema.
    Returns the validation error message if the data does not match the schema.
    """
    errors = _compiled_validator(schema).iter_errors(data)

    # Valid data is the common case, and it does not need best_match at all.
    first_error = next(errors, None)
    if first_error is None:
        return None

    # `jsonschema.validate()` raises best_match(iter_errors(...)), which is not
    # the same error Validator.validate() raises, so match it exactly. chain the
    # already-consumed first error back in rather than re-running iter_errors.
    return str(
        jsonschema.exceptions.best_match(itertools.chain((first_error,), errors))
    )


def sanitize_string(value: str) -> str:
    """Remove PostgreSQL-incompatible control characters from string.

    Strips \\x00-\\x08, \\x0B-\\x0C, \\x0E-\\x1F, \\x7F while keeping tab,
    newline, and carriage return.  Use this before inserting free-form text
    into PostgreSQL text/varchar columns.
    """
    return POSTGRES_CONTROL_CHARS.sub("", value)


def sanitize_json(data: Any) -> Any:
    try:
        # Use two-pass approach for consistent string sanitization:
        # 1. First convert to basic JSON-serializable types (handles Pydantic models)
        # 2. Then sanitize strings in the result
        basic_result = to_dict(data)
        return to_dict(basic_result, custom_encoder={str: sanitize_string})
    except Exception as e:
        # Log the failure and fall back to string representation
        logger.error(
            "SafeJson fallback to string representation due to serialization error: %s (%s). "
            "Data type: %s",
            type(e).__name__,
            truncate(str(e), 200),
            type(data).__name__,
        )

        # Ultimate fallback: convert to string representation and sanitize
        return sanitize_string(str(data))


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
