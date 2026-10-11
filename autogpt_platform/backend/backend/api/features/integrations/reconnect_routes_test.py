"""A connect resumes the scheduled chats waiting on its provider (SECRT-2879)."""

from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from pydantic import SecretStr

from backend.api.features.integrations.router import router
from backend.data.model import OAuth2Credentials, OAuthState

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

ROUTER = "backend.api.features.integrations.router"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@pytest.fixture(autouse=True)
def setup_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _oauth_callback(state: OAuthState | None):
    handler = MagicMock()
    handler.handle_default_scopes.side_effect = lambda scopes: scopes
    handler.exchange_code_for_tokens = AsyncMock(
        return_value=OAuth2Credentials(
            provider="google",
            title="My Google",
            access_token=SecretStr("ya29.access-token"),
            refresh_token=SecretStr("1//refresh-token"),
            scopes=SCOPES,
            username="alice@gmail.com",
        )
    )
    with (
        patch(f"{ROUTER}._get_provider_oauth_handler", return_value=handler),
        patch(f"{ROUTER}.creds_manager") as mock_mgr,
    ):
        mock_mgr.store.verify_state_token = AsyncMock(return_value=state)
        mock_mgr.store.get_creds_by_provider = AsyncMock(return_value=[])
        mock_mgr.create = AsyncMock()
        return client.post(
            "/google/callback", json={"code": "auth-code", "state_token": "state"}
        )


def _create_api_key(create: AsyncMock):
    with patch(f"{ROUTER}.creds_manager") as mock_mgr:
        mock_mgr.create = create
        return client.post(
            "/openai/credentials",
            json={
                "id": "new-cred",
                "provider": "openai",
                "type": "api_key",
                "title": "New Key",
                "api_key": "sk-new",
            },
        )


def test_an_oauth_connect_resumes_what_waits_on_the_provider(
    resume_after_reconnect: MagicMock, test_user_id: str
):
    state = OAuthState(
        token="state", provider="google", expires_at=9999999999, scopes=SCOPES
    )

    assert _oauth_callback(state).status_code == 200

    resume_after_reconnect.assert_called_once_with(test_user_id, "google")


def test_a_failed_oauth_connect_resumes_nothing(resume_after_reconnect: MagicMock):
    assert _oauth_callback(None).status_code == 400

    resume_after_reconnect.assert_not_called()


def test_an_api_key_resumes_what_waits_on_the_provider(
    resume_after_reconnect: MagicMock, test_user_id: str
):
    assert _create_api_key(AsyncMock()).status_code == 201

    resume_after_reconnect.assert_called_once_with(test_user_id, "openai")


def test_an_api_key_that_was_not_stored_resumes_nothing(
    resume_after_reconnect: MagicMock,
):
    response = _create_api_key(AsyncMock(side_effect=RuntimeError("db down")))

    assert response.status_code == 500
    resume_after_reconnect.assert_not_called()
