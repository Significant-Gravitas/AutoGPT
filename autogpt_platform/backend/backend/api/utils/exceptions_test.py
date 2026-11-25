"""What an error response tells the caller."""

import contextlib
import logging

import fastapi
import fastapi.testclient
import pydantic
import pytest

from backend.api.rest_api import app as rest_app
from backend.api.utils import exceptions
from backend.api.utils.exceptions import add_exception_handlers
from backend.util.exceptions import MissingConfigError
from backend.util.service import UnhealthyServiceError


def _client(**handler_options: bool) -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("Client is not connected to the query engine")

    @app.get("/bad")
    async def bad() -> None:
        raise ValueError("limit must be positive")

    @app.get("/unconfigured")
    async def unconfigured() -> None:
        raise MissingConfigError("AYRSHARE_API_KEY is not configured")

    @app.get("/page")
    async def page(limit: int) -> None:
        pass

    @app.get("/built")
    async def built() -> None:
        _Page.model_validate({"limit": "many"})

    add_exception_handlers(app, **handler_options)
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


def test_a_malformed_requests_warning_keeps_the_clients_keys_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A warning is still logged, so it carries no more than the error did."""
    app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)

    @app.post("/limits")
    async def set_limits(limits: _Limits) -> None:
        pass

    with caplog.at_level(logging.WARNING, logger="backend.api.utils.exceptions"):
        response = fastapi.testclient.TestClient(app).post(
            "/limits", json={"per_key": {_CLIENT_KEY: "many"}}
        )

    assert response.json()["detail"][0]["loc"] == ["body", "per_key", _CLIENT_KEY]
    [record] = [r for r in caplog.records if r.name == "backend.api.utils.exceptions"]
    assert record.levelno == logging.WARNING
    assert _CLIENT_KEY not in record.getMessage()
    assert "'loc': ['body', 'per_key', '*']" in record.getMessage()


# A mapping's key is the client's data, a name no loaded model declares.
_CLIENT_KEY = "acct-7Hq2Lx9Wd4"


class _Limits(pydantic.BaseModel):
    per_key: dict[str, int]


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


@pytest.mark.parametrize(
    ("handler_options", "detail"),
    [
        ({"server_error_detail": True}, "AYRSHARE_API_KEY is not configured"),
        # The external APIs: hidden like any other server error.
        ({"server_error_detail": False}, "Internal server error"),
        # REST: other server errors hidden, the missing setting still named.
        (
            {"server_error_detail": False, "expose_missing_config_detail": True},
            "AYRSHARE_API_KEY is not configured",
        ),
    ],
)
def test_a_missing_setting_is_named_unless_the_app_hides_it(
    handler_options: dict[str, bool], detail: str
) -> None:
    response = _client(**handler_options).get("/unconfigured")

    assert response.status_code == 503
    assert response.json()["detail"] == detail
