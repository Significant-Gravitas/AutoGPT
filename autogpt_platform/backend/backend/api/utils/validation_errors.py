"""The one rendering of a validation error that a response or a log line carries.

Every reader under ``backend/api`` goes through ``bounded_validation_errors``,
or ``loggable_validation_errors`` or ``describe_validation_error`` for a log
line; ``backend/util/architecture_validation_errors_test.py`` enforces that.
"""

import functools
import sys
from typing import Any

import pydantic
from fastapi.exceptions import RequestValidationError

from backend.util.url_redaction import REDACTED

MAX_REPORTED_VALIDATION_ERRORS = 20
MAX_VALIDATION_ERROR_CHARS = 200
MAX_VALIDATION_ERROR_LOC_PARTS = 16
# FastAPI's first location part says where in the request a value was.
_REQUEST_PARTS = frozenset({"body", "query", "path", "header", "cookie"})
# A shorter refused value is masked nowhere in a message: it can't be a secret,
# and masking it could hide the limit a message names.
_MIN_MASKED_INPUT = 8


def bounded_validation_errors(
    exc: RequestValidationError | pydantic.ValidationError,
) -> list[dict[str, Any]]:
    """The first ``MAX_REPORTED_VALIDATION_ERRORS`` of ``exc``'s errors, each
    reduced to ``type``, ``loc`` and ``msg`` with every string capped. The three
    fields are an allowlist, so a field a later pydantic adds is not reported."""
    errors = exc.errors()
    reported: list[dict[str, Any]] = [
        {
            "type": _capped(error.get("type", "")),
            "loc": [
                _capped_part(part)
                for part in error.get("loc", ())[:MAX_VALIDATION_ERROR_LOC_PARTS]
            ],
            "msg": _capped(error.get("msg", "")),
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


def loggable_validation_errors(
    exc: RequestValidationError | pydantic.ValidationError,
) -> list[dict[str, Any]]:
    """``bounded_validation_errors`` for a log line, which the caller's own data
    stays out of. A location part that no loaded pydantic model declares, such
    as a mapping's key, is ``*``, and a message that quotes the value it refused
    has it masked; field names, indices and the reasons are kept."""
    names = _declared_names(len(sys.modules))
    reported = bounded_validation_errors(exc)
    for report, error in zip(reported, exc.errors()[:MAX_REPORTED_VALIDATION_ERRORS]):
        report["loc"] = [
            part if isinstance(part, int) or part in names else "*"
            for part in report["loc"]
        ]
        refused = error.get("input")
        if isinstance(refused, str) and len(refused) >= _MIN_MASKED_INPUT:
            report["msg"] = _capped(error.get("msg", "").replace(refused, REDACTED))
    return reported


def describe_validation_error(exc: pydantic.ValidationError) -> str:
    """A log line's account of ``exc``: how many errors, in what, and each as
    ``loggable_validation_errors`` renders it."""
    count = exc.error_count()
    return (
        f"{count} validation error{'' if count == 1 else 's'} for {exc.title}: "
        f"{loggable_validation_errors(exc)}"
    )


@functools.lru_cache(maxsize=1)
def _declared_names(module_count: int) -> frozenset[str]:
    """The name, field names and aliases of every pydantic model loaded, read
    again once ``module_count``, the number of modules loaded, changes."""
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
