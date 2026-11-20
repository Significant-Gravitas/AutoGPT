"""What an error response tells the caller."""

import contextlib
import logging

import fastapi
import fastapi.testclient
import pytest

from backend.api.utils import exceptions
from backend.api.utils.exceptions import add_exception_handlers
from backend.util.service import UnhealthyServiceError


def _client(*, server_error_detail: bool) -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("Client is not connected to the query engine")

    @app.get("/bad")
    async def bad() -> None:
        raise ValueError("limit must be positive")

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


def test_a_failed_health_check_is_a_503_logged_once(caplog) -> None:
    """A 503, not the 400 every ``ValueError`` gets; and the error, which logs
    itself when raised, is not logged again by the handler."""
    app = fastapi.FastAPI()

    @app.get("/health")
    async def health() -> None:
        raise UnhealthyServiceError("Database is not connected", log=False)

    add_exception_handlers(app)
    client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)
    with contextlib.closing(client), caplog.at_level(logging.WARNING):
        response = client.get("/health")

    assert response.status_code == 503
    assert [r for r in caplog.records if r.name == exceptions.logger.name] == []
