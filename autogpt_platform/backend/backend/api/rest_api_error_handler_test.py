"""What the REST app's own error handlers send back (#15651, #15660).

`api/utils/exceptions_test.py` covers the options; this checks the ones
`rest_api` registers.
"""

import fastapi
import fastapi.testclient
import httpx
import pytest
from prisma.errors import PrismaError

from backend.api.rest_api import app as rest_app
from backend.util.exceptions import MissingConfigError, NotFoundError

SECRET = "/data/workspaces/ws/file/name.txt"

# Engine text of the NUL-byte failure from #15660, as Prisma raises it.
_PRISMA_NUL_ERROR = (
    "Error occurred during query execution:\nConnectorError(ConnectorError "
    '{ user_facing_error: None, kind: QueryError(PostgresError { code: "22021", '
    'message: "invalid byte sequence for encoding \\"UTF8\\": 0x00" }) })'
)


def _raise_through_rest_handlers(exc: Exception) -> httpx.Response:
    app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    return fastapi.testclient.TestClient(app, raise_server_exceptions=False).get(
        "/boom"
    )


@pytest.mark.parametrize(
    "exc",
    [
        PrismaError(_PRISMA_NUL_ERROR),
        OSError(36, "File name too long", SECRET),
        RuntimeError(f"connect to db-internal:5432 failed, see {SECRET}"),
    ],
)
def test_a_server_error_does_not_say_what_failed(exc: Exception) -> None:
    response = _raise_through_rest_handlers(exc)

    assert response.status_code == 500
    assert response.json() == {
        "message": "Failed to process GET /boom",
        "detail": "Internal server error",
        "hint": "Check server logs and dependent services.",
    }
    for leaked in ("22021", "ConnectorError", "db-internal", SECRET):
        assert leaked not in response.text


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (MissingConfigError("AYRSHARE_API_KEY is not configured"), 503),
        (ValueError("Search text must not contain NUL characters"), 400),
        (NotFoundError("Graph #1 not found"), 404),
    ],
)
def test_a_missing_setting_or_a_client_error_keeps_its_message(
    exc: Exception, status: int
) -> None:
    response = _raise_through_rest_handlers(exc)

    assert response.status_code == status
    assert response.json()["detail"] == str(exc)
