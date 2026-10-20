"""One-click unsubscribe, answered through the app's own error handlers."""

import base64
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.rest_api import app as rest_app
from backend.data.user import generate_unsubscribe_link
from backend.util.exceptions import DatabaseError

from .postmark import router

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
app.include_router(router)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

USER_ID = "user-1"


@pytest.fixture
def update_preferences(mocker: pytest_mock.MockFixture) -> AsyncMock:
    mocker.patch(
        "backend.data.user.get_user_by_id",
        AsyncMock(return_value=MagicMock(id=USER_ID, email="user@example.com")),
    )
    return mocker.patch(
        "backend.data.user.update_user_notification_preference", AsyncMock()
    )


def test_a_valid_token_unsubscribes_its_user(update_preferences: AsyncMock):
    response = _unsubscribe(_token_for(USER_ID))

    assert response.status_code == 200
    update_preferences.assert_awaited_once()
    assert update_preferences.await_args.args[0] == USER_ID


@pytest.mark.parametrize(
    "token",
    [
        "abc",
        "!!!!",
        base64.urlsafe_b64encode(b"no-separator").decode(),
        base64.urlsafe_b64encode(b"\xff\xfe:x").decode(),
        base64.urlsafe_b64encode(f"{USER_ID}:not-its-signature".encode()).decode(),
        base64.urlsafe_b64encode(f"{USER_ID}:é".encode()).decode(),
    ],
    ids=[
        "bad-padding",
        "not-base64",
        "no-separator",
        "not-utf8",
        "wrong-signature",
        "non-ascii-signature",
    ],
)
def test_a_token_we_did_not_issue_is_a_bad_request(
    token: str, update_preferences: AsyncMock
):
    response = _unsubscribe(token)

    assert response.status_code == 400
    assert token not in response.text
    update_preferences.assert_not_awaited()


def test_a_database_failure_stays_a_server_error(update_preferences: AsyncMock):
    update_preferences.side_effect = DatabaseError("connection reset")

    response = _unsubscribe(_token_for(USER_ID))

    assert response.status_code == 500


def _unsubscribe(token: str):
    return client.post("/unsubscribe", params={"token": token})


def _token_for(user_id: str) -> str:
    return parse_qs(urlparse(generate_unsubscribe_link(user_id)).query)["token"][0]
