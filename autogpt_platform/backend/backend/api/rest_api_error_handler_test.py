"""``handle_internal_http_error`` response bodies (#15651)."""

import fastapi
import fastapi.testclient
import pytest

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
