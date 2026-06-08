import asyncio
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


def test_notify_actor_is_jwt_admin_not_impersonation_header(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The audit actor comes from the JWT, never the X-Act-As-User-Id header.

    Regression: using get_user_id (impersonation-aware) here would let a lingering
    impersonation header name the impersonated user as the actor.
    """
    _set_token(mocker, "fake-token")
    _patch_emails(mocker)
    mock_alert = _patch_alert(mocker, return_value="Message sent")

    response = client.post(
        "/admin/impersonation/notify",
        json={"target_user_id": _TARGET_USER_ID},
        headers={"X-Act-As-User-Id": "spoofed-impersonated-id"},
    )

    assert response.status_code == 200
    content = mock_alert.await_args.args[0]
    assert "spoofed-impersonated-id" not in content


def test_notify_alert_timeout_blocks(mocker: pytest_mock.MockerFixture) -> None:
    """A Discord send that exceeds the timeout is bounded and blocks the swap."""
    _set_token(mocker, "fake-token")
    _patch_emails(mocker)

    async def slow_alert(*_args, **_kwargs):
        await asyncio.sleep(1)
        return "Message sent"

    mocker.patch(f"{_MOCK_MODULE}.discord_send_alert", new=slow_alert)
    mocker.patch(f"{_MOCK_MODULE}._DISCORD_ALERT_TIMEOUT_SECONDS", 0.05)

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 502


def test_notify_missing_target_user_id_returns_422() -> None:
    """A request body without target_user_id fails validation."""
    response = client.post("/admin/impersonation/notify", json={})
    assert response.status_code == 422


def test_notify_email_lookup_failure_is_non_fatal(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """A failed email lookup degrades to 'unknown' but still delivers (200)."""
    _set_token(mocker, "fake-token")
    mocker.patch(
        f"{_MOCK_MODULE}.get_user_email_by_id",
        new_callable=AsyncMock,
        side_effect=Exception("DB connection lost"),
    )
    mock_alert = _patch_alert(mocker, return_value="Message sent")

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 200
    assert response.json() == {"alerted": True}
    content = mock_alert.await_args.args[0]
    assert "test-admin@example.com" in content  # admin email from the JWT
    assert "unknown" in content  # target lookup failed, degraded gracefully


def test_notify_admin_email_comes_from_jwt_without_db_lookup(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The admin email is read from the JWT; only the target is looked up in DB."""
    _set_token(mocker, "fake-token")
    mock_email = mocker.patch(
        f"{_MOCK_MODULE}.get_user_email_by_id",
        new_callable=AsyncMock,
        return_value=_TARGET_EMAIL,
    )
    mock_alert = _patch_alert(mocker, return_value="Message sent")

    response = client.post(
        "/admin/impersonation/notify", json={"target_user_id": _TARGET_USER_ID}
    )

    assert response.status_code == 200
    content = mock_alert.await_args.args[0]
    assert "test-admin@example.com" in content
    # Admin email came from the JWT, so the DB is only hit for the target.
    mock_email.assert_awaited_once_with(_TARGET_USER_ID)
