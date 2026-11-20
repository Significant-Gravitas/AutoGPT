"""The one rendering of a validation error that a response or a log line carries.

Every reader under ``backend/api`` goes through ``bounded_validation_errors``,
or ``loggable_validation_errors`` or ``describe_validation_error`` for a log
line; ``backend/util/architecture_validation_errors_test.py`` enforces that.
"""

import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, get_args

import pydantic
from fastapi.exceptions import RequestValidationError
from pydantic_core import PydanticKnownError
from pydantic_core.core_schema import ErrorType

from backend.util.url_redaction import REDACTED

MAX_REPORTED_VALIDATION_ERRORS = 20
MAX_VALIDATION_ERROR_CHARS = 200
MAX_VALIDATION_ERROR_LOC_PARTS = 16
# FastAPI's first location part says where in the request a value was.
_REQUEST_PARTS = frozenset({"body", "query", "path", "header", "cookie"})
# What a log may show of pydantic's own messages: the schema's side of an error
# (what was expected, a pattern, a limit, an encoding, a class or field name, a
# discriminator and its tags), and the numbers pydantic counts from the input
# (a length, an offset), which can't carry its text. Every other value in an
# error's context comes from the refused input, such as a parser's account of
# it, an exception's text or a validator's message, and is masked. An error
# pydantic doesn't define has a message that can say anything, so a log leaves
# it out. That trusts the schema's side to come from the schema:
# architecture_pydantic_errors_test.py keeps code from raising a built-in
# error with a context of its own.
_SCHEMA_CONTEXT = frozenset(
    {
        "attribute",
        "class",
        "class_name",
        "decimal_places",
        "discriminator",
        "encoding",
        "expected",
        "expected_plural",
        "expected_schemes",
        "expected_tags",
        "expected_version",
        "field_type",
        "ge",
        "gt",
        "le",
        "lt",
        "max_digits",
        "max_length",
        "method_name",
        "min_length",
        "multiple_of",
        "pattern",
        "tz_expected",
        "whole_digits",
        # Counted from the input, but a count or an offset can't carry a
        # credential, and both help debugging.
        "actual_length",
        "tz_actual",
    }
)
_BUILT_IN_TYPES = frozenset(get_args(ErrorType))
CUSTOM_MESSAGE_NOT_LOGGED = "Not logged: a custom error's message can quote the input"
_UNREADABLE = "Not logged: the errors are of a shape the log can't read"


def bounded_validation_errors(
    exc: RequestValidationError | pydantic.ValidationError,
) -> list[dict[str, Any]]:
    """The first ``MAX_REPORTED_VALIDATION_ERRORS`` of ``exc``'s errors, each
    reduced to ``type``, ``loc`` and ``msg`` with every string capped. The three
    fields are an allowlist, so a field a later pydantic adds is not reported."""
    return _bounded(exc.errors(), _capped_part, lambda error: error.get("msg", ""))


def loggable_validation_errors(
    exc: RequestValidationError | pydantic.ValidationError,
) -> list[dict[str, Any]]:
    """``bounded_validation_errors`` for a log line, which the caller's own data
    stays out of. A location part that no loaded pydantic model declares, such
    as a mapping's key, is ``*``, and so is a key a model refused as extra,
    whatever another model calls its fields. pydantic's own messages are kept,
    with any part the refused input gives them, such as a rejected tag, a
    parser's account of it or a validator's message, masked; a custom error's
    message, which can say anything, is left out. Field names, indices and the
    reasons are kept."""
    names = _declared_names(len(sys.modules))
    try:
        return _bounded(
            exc.errors(),
            lambda part: (
                part
                if isinstance(part, int) or (isinstance(part, str) and part in names)
                else "*"
            ),
            _message_without_input,
            _loggable_location,
        )
    except Exception:
        # A hand-built error of a shape pydantic never makes; the log line
        # must not fail the response it accompanies.
        return [{"type": "unreadable", "loc": [], "msg": _UNREADABLE}]


def describe_validation_error(exc: pydantic.ValidationError) -> str:
    """A log line's account of ``exc``: how many errors, in what, and each as
    ``loggable_validation_errors`` renders it."""
    count = exc.error_count()
    return (
        f"{count} validation error{'' if count == 1 else 's'} for {exc.title}: "
        f"{loggable_validation_errors(exc)}"
    )


def _bounded(
    errors: Sequence[Mapping[str, Any]],
    loc_part: Callable[[Any], str | int],
    message: Callable[[Mapping[str, Any]], str],
    location: Callable[[Mapping[str, Any]], Sequence[Any]] = (
        lambda error: error.get("loc", ())
    ),
) -> list[dict[str, Any]]:
    """Only the errors reported are rendered, however many ``errors`` holds."""
    reported: list[dict[str, Any]] = [
        {
            "type": _capped(error.get("type", "")),
            "loc": [
                loc_part(part)
                for part in location(error)[:MAX_VALIDATION_ERROR_LOC_PARTS]
            ],
            "msg": _capped(message(error)),
        }
        for error in errors[:MAX_REPORTED_VALIDATION_ERRORS]
    ]
    if omitted := len(errors) - len(reported):
        reported.append(
            {
                "type": "too_many_errors",
                "loc": [],
                "msg": f"{omitted} more validation errors omitted",
            }
        )
    return reported


def _loggable_location(error: Mapping[str, Any]) -> Sequence[Any]:
    """A location, with the key a model refused as extra as ``*``."""
    location = error.get("loc", ())
    if not isinstance(location, (tuple, list)):
        return ()
    if error.get("type") == "extra_forbidden":
        return (*location[:-1], "*")
    return location


def _message_without_input(error: Mapping[str, Any]) -> str:
    error_type = error.get("type")
    if not isinstance(error_type, str) or error_type not in _BUILT_IN_TYPES:
        return CUSTOM_MESSAGE_NOT_LOGGED
    context = error.get("ctx") or {}
    if not isinstance(context, Mapping):
        return CUSTOM_MESSAGE_NOT_LOGGED
    shown = {
        key: value if key in _SCHEMA_CONTEXT else REDACTED
        for key, value in context.items()
    }
    try:
        # Rendered even when nothing is masked: a hand-built error's own
        # message can quote the input.
        return PydanticKnownError(cast(ErrorType, error_type), shown or None).message()
    except Exception:
        # A context pydantic-core can't render, which a later version could
        # bring; the type and location still say what failed.
        return CUSTOM_MESSAGE_NOT_LOGGED


@functools.lru_cache(maxsize=1)
def _declared_names(module_count: int) -> frozenset[str]:
    """The name, field names and aliases of every pydantic model loaded, read
    again once ``module_count``, the number of modules loaded, changes. A model
    built later without a module, as ``create_model`` builds one, is missed
    until then, and its names are logged as ``*``."""
    names = set(_REQUEST_PARTS)
    unread: list[type[pydantic.BaseModel]] = [pydantic.BaseModel]
    seen: set[type[pydantic.BaseModel]] = set()
    while unread:
        for model in unread.pop().__subclasses__():
            if model in seen:
                continue
            seen.add(model)
            unread.append(model)
            names.add(model.__name__)
            for name, field in model.__pydantic_fields__.items():
                names.add(name)
                names.update(
                    alias
                    for alias in (field.alias, field.validation_alias)
                    if isinstance(alias, str)
                )
    return frozenset(names)


def _capped_part(part: Any) -> str | int:
    return part if isinstance(part, int) else _capped(part)


def _capped(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= MAX_VALIDATION_ERROR_CHARS:
        return text
    return text[:MAX_VALIDATION_ERROR_CHARS] + "…"
