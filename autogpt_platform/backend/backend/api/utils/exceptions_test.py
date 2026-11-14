"""What an error response tells the caller."""

import logging

import fastapi
import fastapi.testclient
import pydantic
import pytest

from backend.api.utils.exceptions import add_exception_handlers


def _client(*, server_error_detail: bool) -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("Client is not connected to the query engine")

    @app.get("/bad")
    async def bad() -> None:
        raise ValueError("limit must be positive")

    @app.get("/page")
    async def page(limit: int) -> None:
        pass

    @app.get("/built")
    async def built() -> None:
        _Page.model_validate({"limit": "many"})

    add_exception_handlers(app, server_error_detail=server_error_detail)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


class _Page(pydantic.BaseModel):
    limit: int


@pytest.mark.parametrize("server_error_detail", [False, True])
def test_a_server_error_names_its_cause_only_where_asked(
    server_error_detail: bool,
) -> None:
    """Third parties call v1, and the text can name internal services."""
    response = _client(server_error_detail=server_error_detail).get("/boom")

    assert response.status_code == 500
    assert ("query engine" in response.json()["detail"]) is server_error_detail


@pytest.mark.parametrize(
    "path, level",
    [("/page", logging.WARNING), ("/built", logging.ERROR)],
    ids=["request-misses-the-schema", "our-code-builds-a-bad-model"],
)
def test_a_validation_failure_is_logged_as_an_error_only_when_it_is_ours(
    path: str, level: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Sentry raises an event for every ERROR line, so a malformed request must not."""
    with caplog.at_level(logging.WARNING, logger="backend.api.utils.exceptions"):
        response = _client(server_error_detail=False).get(path)

    assert response.status_code == 422
    assert [
        r.levelno for r in caplog.records if r.name == "backend.api.utils.exceptions"
    ] == [level]


def test_a_client_error_still_says_what_was_wrong() -> None:
    response = _client(server_error_detail=False).get("/bad")

    assert response.status_code == 400
    assert response.json()["detail"] == "limit must be positive"
