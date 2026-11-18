"""The one rendering of a validation error that a response or a log line carries.

Every reader under ``backend/api`` goes through ``bounded_validation_errors``,
or ``loggable_validation_errors`` or ``describe_validation_error`` for a log
line; ``backend/util/architecture_validation_errors_test.py`` enforces that.
"""

import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, get_args

import pydantic
from fastapi.exceptions import RequestValidationError
from pydantic_core.core_schema import ErrorType

from backend.util.url_redaction import REDACTED

MAX_REPORTED_VALIDATION_ERRORS = 20
MAX_VALIDATION_ERROR_CHARS = 200
MAX_VALIDATION_ERROR_LOC_PARTS = 16
# FastAPI's first location part says where in the request a value was.
_REQUEST_PARTS = frozenset({"body", "query", "path", "header", "cookie"})
# pydantic's own messages say what was expected, not what was sent, so a log
# keeps them. The exceptions: a union's rejected tag, logged from its context
# instead, and the messages of a validator, an attribute read and a custom
# error, which can quote the refused value in any form (escaped, sliced,
# re-cased), so a log leaves them out.
_INPUT_FREE_TYPES = frozenset(get_args(ErrorType)) - {
    "union_tag_invalid",
    "value_error",
    "assertion_error",
    "get_attribute_error",
}
VALIDATOR_MESSAGE_NOT_LOGGED = "Not logged: a validator's message can quote the input"


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
    but a union's rejected tag is left out of its message, and a validator's or
    a custom error's message, which can quote the refused value in any form, is
    left out whole. Field names, indices and the reasons are kept."""
    names = _declared_names(len(sys.modules))
    return _bounded(
        exc.errors(),
        lambda part: part if isinstance(part, int) or part in names else "*",
        _message_without_input,
        _loggable_location,
    )


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
    if error.get("type") == "extra_forbidden":
        return (*location[:-1], "*")
    return location


def _message_without_input(error: Mapping[str, Any]) -> str:
    error_type = error.get("type")
    if error_type in _INPUT_FREE_TYPES:
        return error.get("msg", "")
    if error_type == "union_tag_invalid":
        context = error.get("ctx") or {}
        return (
            f"Input tag '{REDACTED}' found using {context.get('discriminator')} "
            f"does not match any of the expected tags: {context.get('expected_tags')}"
        )
    return VALIDATOR_MESSAGE_NOT_LOGGED


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
