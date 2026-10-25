"""``handle_internal_http_error`` response bodies (#15651, #15660)."""

import fastapi
import fastapi.testclient
import pytest
from prisma.errors import PrismaError

from backend.api.rest_api import app as rest_app
from backend.api.rest_api import handle_internal_http_error
from backend.util.exceptions import MissingConfigError, NotFoundError

SECRET = "/data/workspaces/ws/file/name.txt"


def _client(exc: Exception, handler) -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()
    app.add_exception_handler(type(exc), handler)

    @app.get("/boom")
    async def boom():
        raise exc

    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "exc",
    [
        OSError(36, "File name too long", SECRET),
        RuntimeError(f"connect to db-internal:5432 failed, see {SECRET}"),
    ],
)
def test_5xx_detail_is_generic(exc: Exception):
    response = _client(exc, handle_internal_http_error(500)).get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error"
    assert SECRET not in response.text
    assert body["message"] == "Failed to process GET /boom"
    assert body["hint"] == "Check server logs and dependent services."


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ValueError("Folder name is required"), 400),
        (NotFoundError("Graph #1 not found"), 404),
    ],
)
def test_4xx_keeps_exception_message(exc: Exception, status: int):
    response = _client(exc, handle_internal_http_error(status)).get("/boom")

    assert response.status_code == status
    assert response.json()["detail"] == str(exc)


def test_5xx_can_opt_in_to_exposing_detail():
    exc = MissingConfigError("AYRSHARE_API_KEY is not configured")

    response = _client(exc, handle_internal_http_error(503, expose_detail=True)).get(
        "/boom"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "AYRSHARE_API_KEY is not configured"


# Engine text of the NUL-byte failure from #15660, as Prisma raises it.
_PRISMA_NUL_ERROR = (
    "Error occurred during query execution:\nConnectorError(ConnectorError "
    '{ user_facing_error: None, kind: QueryError(PostgresError { code: "22021", '
    'message: "invalid byte sequence for encoding \\"UTF8\\": 0x00" }) })'
)


@pytest.mark.parametrize(
    "exc",
    [
        PrismaError(_PRISMA_NUL_ERROR),
        RuntimeError(f"unexpected: {SECRET}"),
    ],
)
def test_production_5xx_handlers_hide_exception_text(exc: Exception):
    """The handlers rest_api actually registers for PrismaError and the
    catch-all Exception answer 500 without the exception's text."""
    handler_type = PrismaError if isinstance(exc, PrismaError) else Exception
    handler = rest_app.exception_handlers[handler_type]

    response = _client(exc, handler).get("/boom")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    for leaked in ("22021", "ConnectorError", "0x00", SECRET):
        assert leaked not in response.text


def test_production_missing_config_handler_keeps_message():
    exc = MissingConfigError("AYRSHARE_API_KEY is not configured")

    response = _client(exc, rest_app.exception_handlers[MissingConfigError]).get(
        "/boom"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "AYRSHARE_API_KEY is not configured"


def test_production_value_error_handler_keeps_message():
    exc = ValueError("Search text must not contain NUL characters")

    response = _client(exc, rest_app.exception_handlers[ValueError]).get("/boom")

    assert response.status_code == 400
    assert response.json()["detail"] == str(exc)
