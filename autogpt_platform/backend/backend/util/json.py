import itertools
import logging
import math
import re
import sys
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from enum import Enum
from threading import Lock
from typing import Any, Type, TypeVar, overload

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import orjson
from fastapi.encoders import jsonable_encoder
from prisma import Json
from referencing import Registry

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


def to_dict(data: Any, **kwargs: Any) -> Any:
    """`jsonable_encoder` that also turns numpy scalars into plain Python values."""
    # A numpy value can only exist once numpy is imported, so this never imports it.
    if numpy := sys.modules.get("numpy"):
        kwargs["custom_encoder"] = {
            **(kwargs.get("custom_encoder") or {}),
            numpy.generic: numpy.generic.item,
        }
    return jsonable_encoder(data, **kwargs)


def same_json(left: Any, right: Any) -> bool:
    """Whether two values serialize to the same JSON. Python's ``==`` counts
    ``True`` as ``1`` and ``1.0`` as ``1``; what a block or MCP server is sent
    does not."""
    option = orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS
    return dumps(left, option=option) == dumps(right, option=option)


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
# Built-in blocks contribute normal and dry-run schemas. The bounded LRU
# leaves room for user schemas without retaining arbitrarily large inputs.
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
    ever seen. Key order is deliberately *not* normalised -- it sets the order
    jsonschema checks properties in, so reordering keys could change which
    error is reported.

    """
    key = _schema_cache_key(schema)
    if key is None:
        return _new_validator(schema)

    with _VALIDATOR_CACHE_LOCK:
        cached = _VALIDATOR_CACHE.get(key)
        if cached is not None:
            _VALIDATOR_CACHE.move_to_end(key)
            return cached

    schema_copy = deepcopy(schema)
    snapshot_key = _schema_cache_key(schema_copy)
    validator = _new_validator(schema_copy)
    if snapshot_key is None:
        return validator
    return _remember(snapshot_key, validator)


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
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    # Resolve `$ref` only within the schema and the bundled meta-schemas.
    return validator_cls(schema, registry=Registry())


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
    error = _best_error(schema, data)
    return describe_validation_error(error) if error is not None else None


def _best_error(
    schema: dict[str, Any], data: dict[str, Any]
) -> jsonschema.ValidationError | None:
    """The error `jsonschema.validate()` would raise, or None."""
    errors = _compiled_validator(schema).iter_errors(data)

    # Valid data is the common case, and it does not need best_match at all.
    first_error = next(errors, None)
    if first_error is None:
        return None

    # `jsonschema.validate()` raises best_match(iter_errors(...)), which is not
    # the same error Validator.validate() raises, so match it exactly. chain the
    # already-consumed first error back in rather than re-running iter_errors.
    return jsonschema.exceptions.best_match(itertools.chain((first_error,), errors))


VALIDATION_ERROR_MAX_LENGTH = 300

_JSON_TYPE_NAMES: dict[type, str] = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    tuple: "array",
    dict: "object",
}

_PATH_PART_MAX_LENGTH = 40
_PLAIN_PATH_PART = re.compile(r"[A-Za-z0-9_-]+")
# Keywords whose next schema path part is a name, not a keyword.
_NAMING_KEYWORDS = frozenset(
    {
        "properties",
        "patternProperties",
        "dependentSchemas",
        "dependencies",
        "$defs",
        "definitions",
    }
)

_BOUND_SIGNS: dict[str, str] = {
    "minimum": ">=",
    "exclusiveMinimum": ">",
    "maximum": "<=",
    "exclusiveMaximum": "<",
}

# Draft 4 marks an exclusive bound with a boolean beside it.
_DRAFT_4_EXCLUSIVE_FLAGS: dict[str, str] = {
    "minimum": "exclusiveMinimum",
    "maximum": "exclusiveMaximum",
}

_COUNT_LIMITS: dict[str, tuple[str, str]] = {
    "minLength": ("at least", "character"),
    "maxLength": ("at most", "character"),
    "minItems": ("at least", "item"),
    "maxItems": ("at most", "item"),
    "minProperties": ("at least", "property"),
    "maxProperties": ("at most", "property"),
}


def describe_validation_error(
    error: jsonschema.ValidationError, field_name: str | None = None
) -> str:
    """A short message for a failed check: where it failed and what was
    expected.

    ``str(error)`` carries the whole schema and the given value, which can run
    to hundreds of lines; this carries neither. ``field_name`` names the field
    when the schema was only that field's.
    """
    path: list[str | int] = [field_name] if field_name is not None else []
    path.extend(error.absolute_path)
    location = _format_error_path(path)
    message = _describe_failed_check(error)
    if _checks_a_key(error.absolute_schema_path):
        message = f"a key {message}"
    if location:
        # Keep the failed check and the end of the path, nearest the failure.
        room = VALIDATION_ERROR_MAX_LENGTH - len(message) - 2
        if len(location) > room:
            keep = max(room - 3, 0)
            location = "..." + location[len(location) - keep :]
        message = f"{location}: {message}"
    if len(message) > VALIDATION_ERROR_MAX_LENGTH:
        message = message[: VALIDATION_ERROR_MAX_LENGTH - 3] + "..."
    return message


def _format_error_path(path: list[str | int]) -> str:
    location = ""
    for index, part in enumerate(path):
        if isinstance(part, int):
            location += f"[{part}]"
            continue
        # A key of a free-form dict comes from the given value, so keep it
        # short, and quote it unless it is a plain name.
        name = _short_name(part)
        if _PLAIN_PATH_PART.fullmatch(name):
            location += f".{name}" if index else name
        else:
            # orjson rejects lone surrogates, so spell them out first.
            name = name.encode("utf-8", "backslashreplace").decode("utf-8")
            location += f"[{orjson.dumps(name).decode()}]"
    return location


def _short_name(name: Any) -> str:
    name = str(name)
    if len(name) > _PATH_PART_MAX_LENGTH:
        return name[: _PATH_PART_MAX_LENGTH - 3] + "..."
    return name


def _checks_a_key(schema_path: Iterable[str | int]) -> bool:
    """Whether the failed check came from `propertyNames`, which checks the
    keys of an object rather than its values."""
    after_naming_keyword = False
    for part in schema_path:
        if after_naming_keyword:
            after_naming_keyword = False
        elif part == "propertyNames":
            return True
        else:
            after_naming_keyword = part in _NAMING_KEYWORDS
    return False


def _json_type_name(value: Any) -> str:
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def _expected_types(types: Any) -> list[str] | None:
    """The type names a `type` check allows, or None when they are not all
    plain names (draft 3 also allows schemas there)."""
    names = types if isinstance(types, list) else [types]
    if names and all(isinstance(name, str) for name in names):
        return names
    return None


def _count(number: int, noun: str) -> str:
    if number == 1:
        return f"1 {noun}"
    plural = noun[:-1] + "ies" if noun.endswith("y") else noun + "s"
    return f"{number} {plural}"


def _describe_failed_check(error: jsonschema.ValidationError) -> str:
    # Typed loosely: jsonschema marks these attributes as possibly unset.
    keyword: Any = error.validator
    expected: Any = error.validator_value
    given: Any = error.instance
    schema: Any = error.schema
    context: list[jsonschema.ValidationError] = list(error.context or [])

    if keyword == "type":
        if allowed := _expected_types(expected):
            return f"expected {' or '.join(allowed)}, got {_json_type_name(given)}"
        return "does not match any of the allowed forms"
    if keyword in ("anyOf", "oneOf"):
        if context and all(
            sub.validator == "type" and not sub.relative_path for sub in context
        ):
            allowed_per_form = [_expected_types(sub.validator_value) for sub in context]
            if all(allowed_per_form):
                types = [name for names in allowed_per_form for name in names or []]
                unique_types = " or ".join(dict.fromkeys(types))
                return f"expected {unique_types}, got {_json_type_name(given)}"
        if keyword == "oneOf" and not context:
            return "matches more than one of the allowed forms"
        return "does not match any of the allowed forms"
    if keyword == "required" and isinstance(expected, list):
        present = given if isinstance(given, dict) else {}
        missing = [name for name in expected if name not in present]
        if missing:
            return f"{_short_name(missing[0])!r} is a required property"
    if keyword in ("required", "dependentRequired", "dependencies"):
        return "is missing a required property"
    if keyword == "enum":
        return "is not one of the allowed values"
    if keyword == "const":
        return "is not the required value"
    if keyword in _BOUND_SIGNS and isinstance(expected, (int, float)):
        sign = _BOUND_SIGNS[keyword]
        if (
            keyword in _DRAFT_4_EXCLUSIVE_FLAGS
            and isinstance(schema, dict)
            and schema.get(_DRAFT_4_EXCLUSIVE_FLAGS[keyword]) is True
        ):
            sign = sign.rstrip("=")
        return f"must be {sign} {expected}"
    if keyword == "multipleOf" and isinstance(expected, (int, float)):
        return f"must be a multiple of {expected}"
    if keyword in _COUNT_LIMITS and isinstance(expected, int):
        bound, noun = _COUNT_LIMITS[keyword]
        return f"must have {bound} {_count(expected, noun)}"
    if keyword == "uniqueItems":
        return "must not contain duplicate items"
    if keyword == "pattern":
        return "does not match the required pattern"
    if keyword == "format" and isinstance(expected, str):
        return f"is not a valid {expected}"
    if keyword in ("additionalProperties", "unevaluatedProperties"):
        return "has properties that are not allowed"
    if keyword in ("additionalItems", "items", "unevaluatedItems"):
        return "has items that are not allowed"
    if keyword == "minContains" and isinstance(expected, int):
        return f"must contain at least {_count(expected, 'matching item')}"
    if keyword == "maxContains" and isinstance(expected, int):
        return f"must contain at most {_count(expected, 'matching item')}"
    if keyword == "contains":
        return "does not contain enough of the required items"
    if keyword == "not":
        return "matches a form that is not allowed"
    if keyword is None:
        # A `false` schema, which allows no value at all.
        return "is not allowed"
    return "is not valid"


def sanitize_string(value: str) -> str:
    """Remove PostgreSQL-incompatible control characters from string.

    Strips \\x00-\\x08, \\x0B-\\x0C, \\x0E-\\x1F, \\x7F while keeping tab,
    newline, and carriage return.  Use this before inserting free-form text
    into PostgreSQL text/varchar columns.
    """
    return POSTGRES_CONTROL_CHARS.sub("", value)


def nesting_depth(data: Any) -> int:
    """How many levels of lists and dicts ``data`` contains: 0 for a scalar,
    1 for a flat list or dict, 2 for a list of dicts, and so on.

    Walks iteratively, so it also measures data too deep for ``to_dict`` or
    ``orjson`` to serialize.
    """
    deepest = 0
    pending: list[tuple[Any, int]] = [(data, 0)]
    while pending:
        current, depth = pending.pop()
        if isinstance(current, dict):
            children = current.values()
        elif isinstance(current, (list, tuple)):
            children = current
        else:
            continue
        depth += 1
        deepest = max(deepest, depth)
        pending.extend((child, depth) for child in children)
    return deepest


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
