"""The one rendering of a validation error that a response or a log line carries.

Every reader under ``backend/api`` goes through ``bounded_validation_errors``,
or ``loggable_validation_errors`` or ``describe_validation_error`` for a log
line; ``backend/util/architecture_validation_errors_test.py`` enforces that.
"""

import functools
import itertools
import sys
from collections import deque
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
# A shorter refused value is masked only where a message quotes it: masked
# anywhere, it could hide the limit a message names.
_MIN_MASKED_INPUT = 8
_QUOTES = ("'", '"', "`")
# How much of a refused value is read for text a message might quote.
_MAX_INPUT_NODES = 256
# pydantic's own messages say what was expected, not what was sent. A union's
# rejected tag is the exception, and is logged from its context instead; a
# validator's own message, or a custom error's, can say anything.
_INPUT_FREE_TYPES = frozenset(get_args(ErrorType)) - {
    "union_tag_invalid",
    "value_error",
    "assertion_error",
}


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
    whatever another model calls its fields. A union's rejected tag is left out
    of its message, other text of the refused value that a message quotes is
    masked, and a message that can quote it is left out when the value is too
    large to read whole. Field names, indices and the reasons are kept."""
    names = _declared_names(len(sys.modules))
    errors = [
        (
            {**error, "loc": (*error.get("loc", ())[:-1], "*")}
            if error.get("type") == "extra_forbidden"
            else error
        )
        for error in exc.errors()
    ]
    return _bounded(
        errors,
        lambda part: part if isinstance(part, int) or part in names else "*",
        _message_without_input,
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
) -> list[dict[str, Any]]:
    reported: list[dict[str, Any]] = [
        {
            "type": _capped(error.get("type", "")),
            "loc": [
                loc_part(part)
                for part in error.get("loc", ())[:MAX_VALIDATION_ERROR_LOC_PARTS]
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


def _message_without_input(error: Mapping[str, Any]) -> str:
    error_type = error.get("type")
    if error_type == "union_tag_invalid":
        context = error.get("ctx") or {}
        return (
            f"Input tag '{REDACTED}' found using {context.get('discriminator')} "
            f"does not match any of the expected tags: {context.get('expected_tags')}"
        )
    texts, complete = _texts_in(error.get("input"))
    if not complete and error_type not in _INPUT_FREE_TYPES:
        return "Not logged: the refused value is too large to check for credentials"
    message = error.get("msg", "")
    for text in sorted(texts, key=len, reverse=True):
        if len(text) >= _MIN_MASKED_INPUT:
            message = message.replace(text, REDACTED)
        else:
            for quote in _QUOTES:
                message = message.replace(
                    f"{quote}{text}{quote}", f"{quote}{REDACTED}{quote}"
                )
    return message


def _texts_in(value: Any) -> tuple[set[str], bool]:
    """The strings in ``value``, its keys long enough to be a secret included,
    read breadth-first and no further than ``_MAX_INPUT_NODES`` values; and
    whether that was all of it."""
    found: set[str] = set()
    pending: deque[tuple[Any, bool]] = deque([(value, False)])
    # Children are queued only while the budget has room, so a large refused
    # mapping costs no more than the values it is read for.
    room = _MAX_INPUT_NODES - 1
    complete = True
    while pending:
        item, is_key = pending.popleft()
        if isinstance(item, str):
            if item and (not is_key or len(item) >= _MIN_MASKED_INPUT):
                found.add(item)
            continue
        if isinstance(item, Mapping):
            children = itertools.chain.from_iterable(
                ((key, True), (child, False)) for key, child in item.items()
            )
            size = 2 * len(item)
        elif isinstance(item, (list, tuple)):
            children = ((child, False) for child in item)
            size = len(item)
        else:
            continue
        taken = list(itertools.islice(children, room))
        complete = complete and len(taken) == size
        room -= len(taken)
        pending.extend(taken)
    return found, complete


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
