"""The one rendering of a validation error that a response or a log line carries.

Every reader under ``backend/api`` goes through ``bounded_validation_errors``;
``backend/util/architecture_test.py`` enforces that.
"""

from typing import Any

import pydantic
from fastapi.exceptions import RequestValidationError

MAX_REPORTED_VALIDATION_ERRORS = 20
MAX_VALIDATION_ERROR_CHARS = 200
MAX_VALIDATION_ERROR_LOC_PARTS = 16


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


def _capped_part(part: Any) -> str | int:
    return part if isinstance(part, int) else _capped(part)


def _capped(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= MAX_VALIDATION_ERROR_CHARS:
        return text
    return text[:MAX_VALIDATION_ERROR_CHARS] + "…"
