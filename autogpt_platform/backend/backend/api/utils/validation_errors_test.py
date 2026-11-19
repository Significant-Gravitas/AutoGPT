"""The bounds `bounded_validation_errors` puts on what a 422 reports and logs,
and the mounted apps that must answer through it."""

import logging
from collections.abc import Iterator, Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Literal
from uuid import UUID

import fastapi
import fastapi.testclient
import pydantic
import pytest
import starlette.routing
from fastapi.exceptions import RequestValidationError
from pydantic_core import PydanticCustomError

# Private: pydantic-core lists its errors only here. If an upgrade moves it,
# this module fails to collect, and the renderer's allowlist is then rechecked
# against the new version's errors.
from pydantic_core._pydantic_core import list_all_errors

from backend.api.external.v2.errors import add_v2_exception_handlers
from backend.api.utils import validation_errors
from backend.api.utils.exceptions import add_exception_handlers
from backend.api.utils.validation_errors import (
    CUSTOM_MESSAGE_NOT_LOGGED,
    MAX_REPORTED_VALIDATION_ERRORS,
    MAX_VALIDATION_ERROR_CHARS,
    MAX_VALIDATION_ERROR_LOC_PARTS,
    bounded_validation_errors,
    loggable_validation_errors,
)
from backend.util.url_redaction import REDACTED

OVER_CAP = "x" * (MAX_VALIDATION_ERROR_CHARS + 100)
TOO_MANY = [0] * (MAX_REPORTED_VALIDATION_ERRORS + 5)

# Every mounted app of the REST API process, and the module that owns the
# bounded handler each is expected to use.
MOUNTED_APPS = {
    "/": "backend.api.utils.exceptions",
    "/external-api": "backend.api.utils.exceptions",
    "/external-api/v1": "backend.api.utils.exceptions",
    "/api/v2": "backend.api.external.v2.errors",
    # Not FastAPI: the MCP SDK answers its own errors, through no handler of ours.
    "/api/v2/mcp": None,
}


class _Ids(pydantic.BaseModel):
    ids: list[str]
    name: str


def _errors_for(body: object) -> list[dict]:
    with pytest.raises(pydantic.ValidationError) as caught:
        _Ids.model_validate(body)
    return bounded_validation_errors(caught.value)


def test_the_rejected_value_is_not_reported() -> None:
    """An error is reported as its ``type``, ``loc`` and ``msg`` only."""
    reported = _errors_for({"filler": OVER_CAP})

    assert [error["loc"] for error in reported] == [["ids"], ["name"]]
    assert all(set(error) == {"type", "loc", "msg"} for error in reported)
    assert OVER_CAP not in str(reported)


class _CountedReads(Mapping[str, str]):
    """A large mapping that counts how many of its keys were read."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.read = 0

    def __getitem__(self, key: str) -> str:
        return "a value as long as a credential"

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[str]:
        for index in range(self.size):
            self.read += 1
            yield f"key-{index}"


def test_a_large_refused_mapping_is_not_read_for_the_log() -> None:
    """A missing field's error keeps the whole mapping it was missing from."""
    refused = _CountedReads(100_000)
    error = pydantic.ValidationError.from_exception_data(
        "_Ids", [{"type": "missing", "loc": ("ids",), "input": refused}]
    )

    assert loggable_validation_errors(error)[0]["loc"] == ["ids"]
    assert refused.read == 0


def test_a_validators_own_message_stays_out_of_the_log() -> None:
    """A validator, a custom error or an attribute read can quote what was sent
    in any form, so its message isn't logged; pydantic's own messages, which
    say what was expected, are."""
    secret = "line one\nS3cr3t-line-two"
    error = pydantic.ValidationError.from_exception_data(
        "_Ids",
        [
            {
                "type": "value_error",
                "loc": ("name",),
                "input": secret,
                "ctx": {"error": ValueError(f"bad token {secret!r}")},
            },
            {
                "type": PydanticCustomError(
                    "token_refused", "{token} refused", {"token": secret}
                ),
                "loc": ("token",),
                "input": secret,
            },
            {"type": "missing", "loc": ("ids",), "input": {}},
        ],
    )

    logged = loggable_validation_errors(error)
    assert [entry["msg"] for entry in logged] == [
        f"Value error, {REDACTED}",
        CUSTOM_MESSAGE_NOT_LOGGED,
        "Field required",
    ]
    # The response keeps the validator's message.
    assert "S3cr3t-line-two" in bounded_validation_errors(error)[0]["msg"]


# No word a message could hold, and characters no message uses.
_SENTINEL = "Ꙭqz7§XpW"
_EXAMPLES = {error["type"]: error for error in list_all_errors()}


def _logs_part_of(text: str) -> bool:
    return (
        "Ꙭ" in text
        or "§" in text
        or any(_SENTINEL[i : i + 4] in text for i in range(len(_SENTINEL) - 3))
    )


@pytest.mark.parametrize("error_type", sorted(_EXAMPLES))
def test_no_built_in_message_logs_what_its_context_takes_from_the_input(
    error_type: str,
) -> None:
    """Whatever an error's context holds besides the schema's side, such as a
    parser's account of the refused value, is masked in the log line."""
    example = dict(_EXAMPLES[error_type].get("example_context") or {})
    context = {
        key: value if key in validation_errors._SCHEMA_CONTEXT else _SENTINEL
        for key, value in example.items()
    }
    details = {"type": error_type, "loc": ("x",), "input": _SENTINEL}
    error = pydantic.ValidationError.from_exception_data(
        "_Ids", [{**details, "ctx": context} if context else details]
    )

    [logged] = loggable_validation_errors(error)
    assert not _logs_part_of(logged["msg"]), logged["msg"]


class _Parsed(pydantic.BaseModel):
    uuid: UUID
    when: datetime
    day: date
    at: time
    took: timedelta
    url: pydantic.AnyUrl
    blob: pydantic.Json[dict]
    amount: Decimal
    count: int
    ratio: float
    host: pydantic.IPvAnyAddress
    data: pydantic.Base64Bytes
    mode: Literal["a", "b"]


@pytest.mark.parametrize("field", list(_Parsed.model_fields))
def test_a_parse_error_logs_no_part_of_the_refused_value(field: str) -> None:
    with pytest.raises(pydantic.ValidationError) as caught:
        _Parsed.model_validate({field: _SENTINEL})

    logged = [
        entry
        for entry in loggable_validation_errors(caught.value)
        if entry["loc"][:1] == [field]
    ]
    assert logged
    assert not any(_logs_part_of(entry["msg"]) for entry in logged), logged


@pytest.mark.parametrize(
    "error",
    [
        {"type": "missing", "loc": ("body", ["a", "list"]), "msg": "x"},
        {"type": "missing", "loc": ("body", {"a": 1}), "msg": "x"},
        {"type": ["not", "hashable"], "loc": ("body",), "msg": "x"},
        {"type": "union_tag_invalid", "loc": ("body",), "msg": "x", "ctx": ["a"]},
        {"type": "missing", "loc": None, "msg": "x"},
        "not a mapping",
    ],
)
def test_the_log_reads_any_shape_of_error_without_failing(error: object) -> None:
    """A hand-built error can take shapes pydantic never makes; the log line
    must not turn the 422 it accompanies into a 500."""
    logged = loggable_validation_errors(RequestValidationError([error]))
    assert logged and all(set(entry) == {"type", "loc", "msg"} for entry in logged)


@pytest.mark.parametrize("error_type", ["missing", "string_type", "value_error"])
def test_a_hand_built_errors_own_message_is_not_logged(error_type: str) -> None:
    """A built-in error's log message comes from pydantic's template, never
    from the error's own ``msg``, which a hand-built one can fill with the
    input; one whose template needs a context it lacks is left out."""
    error = {"type": error_type, "loc": ("body", "name"), "msg": f"bad {_SENTINEL}"}

    [logged] = loggable_validation_errors(RequestValidationError([error]))

    assert not _logs_part_of(logged["msg"]), logged["msg"]
    assert logged["msg"]


def test_a_message_with_nothing_to_mask_logs_as_pydantic_wrote_it() -> None:
    class Limited(pydantic.BaseModel):
        name: str
        count: int = pydantic.Field(gt=5)

    with pytest.raises(pydantic.ValidationError) as caught:
        Limited.model_validate({"count": 3})

    assert [entry["msg"] for entry in loggable_validation_errors(caught.value)] == [
        error["msg"] for error in caught.value.errors()
    ]


def test_an_attribute_read_logs_no_part_of_what_it_raised() -> None:
    """With ``from_attributes``, an attribute that raises puts its exception's
    text in pydantic's own message."""

    class Read(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(from_attributes=True)
        name: str

    class Source:
        @property
        def name(self) -> str:
            raise ValueError(f"bad {_SENTINEL}")

    with pytest.raises(pydantic.ValidationError) as caught:
        Read.model_validate(Source())

    [logged] = loggable_validation_errors(caught.value)
    assert logged["type"] == "get_attribute_error"
    assert not _logs_part_of(logged["msg"]), logged["msg"]


def test_a_failing_iterator_logs_no_part_of_what_it_raised() -> None:
    """An iterator that yields a valid item and then raises puts the
    exception's text in pydantic's own message."""

    class Listed(pydantic.BaseModel):
        items: list[int]

    def items() -> Iterator[int]:
        yield 1
        raise ValueError(f"bad {_SENTINEL}")

    with pytest.raises(pydantic.ValidationError) as caught:
        Listed.model_validate({"items": items()})

    [logged] = loggable_validation_errors(caught.value)
    assert logged["type"] == "iteration_error"
    assert not _logs_part_of(logged["msg"]), logged["msg"]


def test_only_the_reported_errors_locations_are_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request with thousands of keys a model refuses costs no more than the
    errors a log line reports."""
    rewritten: list[object] = []
    location = validation_errors._loggable_location

    def counted(error: Mapping[str, object]) -> object:
        rewritten.append(error)
        return location(error)

    monkeypatch.setattr(validation_errors, "_loggable_location", counted)

    class Strict(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")

    with pytest.raises(pydantic.ValidationError) as caught:
        Strict.model_validate({f"extra{i}": i for i in range(10_000)})
    logged = loggable_validation_errors(caught.value)

    assert len(rewritten) == MAX_REPORTED_VALIDATION_ERRORS
    assert logged[-1]["msg"] == "9980 more validation errors omitted"


def test_a_list_field_reports_a_bounded_number_of_errors() -> None:
    reported = _errors_for({"ids": TOO_MANY, "name": "n"})

    assert len(reported) == MAX_REPORTED_VALIDATION_ERRORS + 1
    assert [error["loc"] for error in reported[:3]] == [
        ["ids", 0],
        ["ids", 1],
        ["ids", 2],
    ]
    assert reported[-1]["msg"] == "5 more validation errors omitted"


def test_a_request_derived_string_is_capped() -> None:
    """Every string in an error, each part of ``loc`` included, is capped."""
    error = {"type": "x" * 300, "loc": ("body", OVER_CAP, 7), "msg": "y" * 300}

    (reported,) = bounded_validation_errors(RequestValidationError([error]))

    assert reported["loc"] == ["body", "x" * MAX_VALIDATION_ERROR_CHARS + "…", 7]
    assert reported["msg"] == "y" * MAX_VALIDATION_ERROR_CHARS + "…"
    assert reported["type"] == "x" * MAX_VALIDATION_ERROR_CHARS + "…"


def test_a_deep_loc_is_capped() -> None:
    error = {"type": "t", "loc": tuple(range(100)), "msg": "m"}

    (reported,) = bounded_validation_errors(RequestValidationError([error]))

    assert reported["loc"] == list(range(MAX_VALIDATION_ERROR_LOC_PARTS))


def _rest_client() -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.post("/ids")
    async def ids(body: _Ids) -> None: ...

    add_exception_handlers(app)
    return fastapi.testclient.TestClient(app)


def test_the_rest_api_bounds_its_422(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger="backend.api.utils.exceptions"):
        response = _rest_client().post(
            "/ids", json={"ids": TOO_MANY, "filler": OVER_CAP}
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) == MAX_REPORTED_VALIDATION_ERRORS + 1
    assert len(response.content) < 3_000
    assert OVER_CAP not in caplog.text
    assert len(caplog.text) < 3_000
    assert "string_type" in caplog.text


def _v2_client() -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.post("/ids")
    async def ids(body: _Ids) -> None: ...

    add_v2_exception_handlers(app)
    return fastapi.testclient.TestClient(app)


def test_the_v2_api_bounds_its_422(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="backend.api.external.v2.errors"):
        response = _v2_client().post("/ids", json={"ids": TOO_MANY, "filler": OVER_CAP})

    assert response.status_code == 422
    errors = response.json()["error"]["details"]["errors"]
    assert len(errors) == MAX_REPORTED_VALIDATION_ERRORS + 1
    assert len(response.content) < 3_000
    assert OVER_CAP not in caplog.text
    assert len(caplog.text) < 3_000


def test_every_mounted_app_bounds_its_validation_errors() -> None:
    """A mounted app does not inherit its parent's exception handlers, so each
    registers its own."""
    from backend.api.rest_api import app

    found = dict(_mounted_apps(app))

    assert set(found) == set(MOUNTED_APPS), "a new mounted app needs a handler too"
    for path, mounted in found.items():
        if MOUNTED_APPS[path] is None:
            continue
        for exception in (RequestValidationError, pydantic.ValidationError):
            handler = mounted.exception_handlers.get(exception)
            assert handler is not None, f"{path} has no {exception.__name__} handler"
            assert handler.__module__ == MOUNTED_APPS[path], (
                f"{path} handles {exception.__name__} with "
                f"{handler.__module__}.{handler.__qualname__}, which is not a "
                "reader of `bounded_validation_errors`"
            )


def _mounted_apps(app: fastapi.FastAPI, prefix: str = ""):
    yield prefix or "/", app
    for route in app.routes:
        if not isinstance(route, starlette.routing.Mount):
            continue
        if isinstance(route.app, fastapi.FastAPI):
            yield from _mounted_apps(route.app, prefix + route.path)
        else:
            yield prefix + route.path, route.app
