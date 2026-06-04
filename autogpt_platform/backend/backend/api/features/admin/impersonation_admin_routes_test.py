from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.util.metrics import DiscordChannel

from .impersonation_admin_routes import router as impersonation_router

app = fastapi.FastAPI()
app.include_router(impersonation_router)

client = fastapi.testclient.TestClient(app)

_MOCK_MODULE = "backend.api.features.admin.impersonation_admin_routes"

_TARGET_USER_ID = "target-user-id"
_ADMIN_EMAIL = "admin@agpt.co"
_TARGET_EMAIL = "target@example.com"


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Make every request in this module authenticate as an admin by default."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _set_token(mocker: pytest_mock.MockerFixture, token: str) -> None:
    """Swap the module's settings for a tiny stub exposing only what's read."""
    mocker.patch(
        f"{_MOCK_MODULE}.settings",
        SimpleNamespace(secrets=SimpleNamespace(discord_bot_token=token)),
    )


def _patch_emails(mocker: pytest_mock.MockerFixture) -> None:
    mocker.patch(
        f"{_MOCK_MODULE}.get_user_email_by_id",
        new_callable=AsyncMock,
        side_effect=lambda uid: (
            _ADMIN_EMAIL if uid != _TARGET_USER_ID else _TARGET_EMAIL
        ),
    )


def _patch_alert(mocker: pytest_mock.MockerFixture, **kwargs) -> AsyncMock:
    return mocker.patch(
        f"{_MOCK_MODULE}.discord_send_alert", new_callable=AsyncMock, **kwargs
    )


def test_notify_alert_delivered_allows(mocker: pytest_mock.MockerFixture) -> None:
    """Token set + Discord confirms delivery -> 200 alerted=True, sent to PLATFORM."""
    _set_token(mocker, "fake-token")
    _patch_emails(mocker)
    mock_alert = _patch_alert(mocker, return_value="Message sent")

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 200
    assert response.json() == {"alerted": True}
    mock_alert.assert_awaited_once()
    # second positional arg is the channel
    assert mock_alert.await_args.args[1] == DiscordChannel.PLATFORM


def test_notify_no_token_skips_alert_allows(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """No bot token -> skip the alert and allow the swap (200 alerted=False)."""
    _set_token(mocker, "")
    mock_alert = _patch_alert(mocker, return_value="Message sent")

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 200
    assert response.json() == {"alerted": False}
    mock_alert.assert_not_awaited()


def test_notify_channel_not_found_blocks(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Misconfigured channel returns a non-success status (no raise) -> 502 block."""
    _set_token(mocker, "fake-token")
    _patch_emails(mocker)
    _patch_alert(mocker, return_value="Channel not found: local-alerts")

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 502


def test_notify_send_raises_blocks(mocker: pytest_mock.MockerFixture) -> None:
    """Discord send raising (e.g. bad token / login failure) -> 502 block."""
    _set_token(mocker, "fake-token")
    _patch_emails(mocker)
    _patch_alert(mocker, side_effect=ValueError("Login error occurred"))

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 502


def test_notify_requires_admin_role(mock_jwt_user) -> None:
    """Non-admin callers are rejected with 403."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 403
