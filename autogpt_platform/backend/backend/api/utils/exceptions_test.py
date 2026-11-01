"""What an error response tells the caller."""

import fastapi
import fastapi.testclient
import pydantic
import pytest

from backend.api.utils.exceptions import (
    MAX_VALIDATION_ERRORS_SHOWN,
    MAX_VALIDATION_INPUT_PREVIEW_CHARS,
    add_exception_handlers,
)


class _Items(pydantic.BaseModel):
    items: list[int] = pydantic.Field(max_length=3)


class _Numbers(pydantic.BaseModel):
    numbers: list[int]


def _client(*, server_error_detail: bool) -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("Client is not connected to the query engine")

    @app.get("/bad")
    async def bad() -> None:
        raise ValueError("limit must be positive")

    @app.post("/items")
    async def items(body: _Items) -> None:
        return None

    @app.post("/numbers")
    async def numbers(body: _Numbers) -> None:
        return None

    add_exception_handlers(app, server_error_detail=server_error_detail)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("server_error_detail", [False, True])
def test_a_server_error_names_its_cause_only_where_asked(
    server_error_detail: bool,
) -> None:
    """Third parties call v1, and the text can name internal services."""
    response = _client(server_error_detail=server_error_detail).get("/boom")

    assert response.status_code == 500
    assert ("query engine" in response.json()["detail"]) is server_error_detail


def test_a_client_error_still_says_what_was_wrong() -> None:
    response = _client(server_error_detail=False).get("/bad")

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be positive"


def test_an_oversized_request_gets_a_small_validation_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every item fails, but the body and the log line name only the first
    errors, each with a short preview of the value."""
    value = "x" * 10_000
    response = _client(server_error_detail=False).post(
        "/numbers", json={"numbers": [value] * 2_000}
    )

    assert response.status_code == 422
    body = response.json()
    assert len(body["detail"]) == MAX_VALIDATION_ERRORS_SHOWN
    assert body["omitted_errors"] == 2_000 - MAX_VALIDATION_ERRORS_SHOWN
    first = body["detail"][0]
    assert first["loc"] == ["body", "numbers", 0]
    assert first["type"] == "int_parsing"
    assert len(first["input"]) <= MAX_VALIDATION_INPUT_PREVIEW_CHARS
    assert len(response.content) < 50_000
    [line] = [r.getMessage() for r in caplog.records if "Validation failed" in r.getMessage()]
    assert len(line) < 50_000
    assert f"and {2_000 - MAX_VALIDATION_ERRORS_SHOWN} more errors" in line


def test_a_validation_error_still_previews_the_value_it_failed_on() -> None:
    response = _client(server_error_detail=False).post(
        "/items", json={"items": [987654321] * 50}
    )

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", "items"]
    assert error["type"] == "too_long"
    assert error["input"].startswith("[987654321, 987654321")
    assert len(error["input"]) <= MAX_VALIDATION_INPUT_PREVIEW_CHARS
    assert "omitted_errors" not in response.json()
