"""The bounds `bounded_validation_errors` puts on what a 422 reports and logs,
and the mounted apps that must answer through it."""

import logging

import fastapi
import fastapi.testclient
import pydantic
import pytest
import starlette.routing
from fastapi.exceptions import RequestValidationError

from backend.api.external.v2.errors import add_v2_exception_handlers
from backend.api.utils.exceptions import add_exception_handlers
from backend.api.utils.validation_errors import (
    MAX_REPORTED_VALIDATION_ERRORS,
    MAX_VALIDATION_ERROR_CHARS,
    MAX_VALIDATION_ERROR_LOC_PARTS,
    bounded_validation_errors,
)

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
