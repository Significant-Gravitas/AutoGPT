"""Tests for push notification routes."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.push.routes import router

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_get_vapid_public_key(mocker):
    mock_settings = MagicMock()
    mock_settings.secrets.vapid_public_key = "test-vapid-public-key-base64url"
    mocker.patch(
        "backend.api.features.push.routes._settings",
        mock_settings,
    )

    response = client.get("/vapid-key")

    assert response.status_code == 200
    data = response.json()
    assert data["public_key"] == "test-vapid-public-key-base64url"


def test_get_vapid_public_key_empty(mocker):
    mock_settings = MagicMock()
    mock_settings.secrets.vapid_public_key = ""
    mocker.patch(
        "backend.api.features.push.routes._settings",
        mock_settings,
    )

    response = client.get("/vapid-key")

    assert response.status_code == 200
    data = response.json()
    assert data["public_key"] == ""


def test_subscribe_push(mocker, test_user_id):
    mock_upsert = mocker.patch(
        "backend.api.features.push.routes.upsert_push_subscription",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/subscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
            "keys": {
                "p256dh": "test-p256dh-key",
                "auth": "test-auth-key",
            },
            "user_agent": "Mozilla/5.0 Test",
        },
    )

    assert response.status_code == 204
    mock_upsert.assert_awaited_once_with(
        user_id=test_user_id,
        endpoint="https://fcm.googleapis.com/fcm/send/abc123",
        p256dh="test-p256dh-key",
        auth="test-auth-key",
        user_agent="Mozilla/5.0 Test",
    )


def test_subscribe_push_without_user_agent(mocker, test_user_id):
    mock_upsert = mocker.patch(
        "backend.api.features.push.routes.upsert_push_subscription",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/subscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
            "keys": {
                "p256dh": "test-p256dh-key",
                "auth": "test-auth-key",
            },
        },
    )

    assert response.status_code == 204
    mock_upsert.assert_awaited_once_with(
        user_id=test_user_id,
        endpoint="https://fcm.googleapis.com/fcm/send/abc123",
        p256dh="test-p256dh-key",
        auth="test-auth-key",
        user_agent=None,
    )


def test_subscribe_push_missing_keys():
    response = client.post(
        "/subscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
        },
    )

    assert response.status_code == 422


def test_subscribe_push_missing_endpoint():
    response = client.post(
        "/subscribe",
        json={
            "keys": {
                "p256dh": "test-p256dh-key",
                "auth": "test-auth-key",
            },
        },
    )

    assert response.status_code == 422


def test_subscribe_push_rejects_empty_crypto_keys():
    response = client.post(
        "/subscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
            "keys": {"p256dh": "", "auth": ""},
        },
    )

    assert response.status_code == 422


def test_subscribe_push_rejects_oversized_endpoint():
    response = client.post(
        "/subscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/" + "x" * 3000,
            "keys": {"p256dh": "k", "auth": "a"},
        },
    )

    assert response.status_code == 422


def test_unsubscribe_push(mocker, test_user_id):
    mock_delete = mocker.patch(
        "backend.api.features.push.routes.delete_push_subscription",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/unsubscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
        },
    )

    assert response.status_code == 204
    mock_delete.assert_awaited_once_with(
        test_user_id,
        "https://fcm.googleapis.com/fcm/send/abc123",
    )


def test_unsubscribe_push_missing_endpoint():
    response = client.post(
        "/unsubscribe",
        json={},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "untrusted_endpoint",
    [
        "https://localhost/evil",
        "https://127.0.0.1/evil",
        "https://169.254.169.254/latest/meta-data/",
        "https://internal-service.local/api",
        "https://attacker.example.com/push",
        "http://fcm.googleapis.com/fcm/send/abc",
        "file:///etc/passwd",
    ],
)
def test_subscribe_push_rejects_untrusted_endpoints(mocker, untrusted_endpoint):
    mock_upsert = mocker.patch(
        "backend.api.features.push.routes.upsert_push_subscription",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/subscribe",
        json={
            "endpoint": untrusted_endpoint,
            "keys": {
                "p256dh": "test-p256dh-key",
                "auth": "test-auth-key",
            },
        },
    )

    assert response.status_code == 400
    mock_upsert.assert_not_awaited()


def test_subscribe_push_surfaces_cap_as_400(mocker):
    mocker.patch(
        "backend.api.features.push.routes.upsert_push_subscription",
        new_callable=AsyncMock,
        side_effect=ValueError("Subscription limit of 20 per user reached"),
    )

    response = client.post(
        "/subscribe",
        json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
            "keys": {
                "p256dh": "test-p256dh-key",
                "auth": "test-auth-key",
            },
        },
    )

    assert response.status_code == 400
    assert "Subscription limit" in response.json()["detail"]
