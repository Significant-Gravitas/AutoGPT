from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth import get_optional_user_id
from pydantic import SecretStr

from backend.api.features.integrations.codex import (
    CODEX_LOGIN_STATE_KEY,
    CodexDeviceLogin,
    CodexDeviceLoginState,
    build_device_login_cancel_url,
    build_device_login_url,
    render_device_login_page,
    revoke_codex_credentials,
)
from backend.api.features.integrations.router import router
from backend.data.model import OAuth2Credentials, OAuthState
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexRateLimitsSnapshot,
    CodexRateLimitWindow,
)

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture(autouse=True)
def setup_auth(mocker):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": TEST_USER_ID,
        "role": "user",
        "email": "test@example.com",
    }
    app.dependency_overrides[get_optional_user_id] = lambda: TEST_USER_ID
    mocker.patch(
        "backend.api.features.integrations.router.has_codex_access_for_discovery",
        new=AsyncMock(return_value=True),
    )
    mocker.patch(
        "backend.api.features.integrations.router.enforce_codex_access_http",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.features.integrations.codex.enforce_codex_access_http",
        new=AsyncMock(),
    )
    yield
    app.dependency_overrides.clear()


def _credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="codex-credential",
        provider="codex",
        title="ChatGPT for Codex",
        username="user@example.com",
        access_token=SecretStr("access-secret"),
        refresh_token=SecretStr("refresh-secret"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr("provider-secret"),
        provider_state_version=1,
    )


def _oauth_state(login_id: str = "login-123") -> OAuthState:
    return OAuthState(
        token="state-123",
        provider="codex",
        expires_at=4_000_000_000,
        scopes=[],
        state_metadata={CODEX_LOGIN_STATE_KEY: login_id},
    )


def test_codex_login_reuses_generic_oauth_contract():
    login = CodexDeviceLogin(
        login_id="login-123",
        verification_url="https://auth.openai.com/codex/device",
        user_code="ABCD-EFGH",
    )
    with (
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
        patch("backend.api.features.integrations.router.settings") as settings,
    ):
        settings.config.frontend_base_url = "http://localhost:3000"
        settings.config.codex_login_timeout_seconds = 900
        coordinator.start = AsyncMock(return_value=login)
        manager.store.store_state_token = AsyncMock(
            return_value=("state-123", "unused-challenge")
        )

        response = client.get("/codex/login")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state_token"] == "state-123"
    assert payload["login_url"].startswith(
        "http://localhost:3000/api/proxy/api/integrations/codex/device-login/login-123#"
    )
    assert payload["cancel_url"] == (
        "/api/proxy/api/integrations/codex/device-login/login-123/cancel"
    )
    assert "state=state-123" in payload["login_url"].split("#", 1)[1]
    assert "user_code=ABCD-EFGH" in payload["login_url"].split("#", 1)[1]
    manager.store.store_state_token.assert_awaited_once_with(
        TEST_USER_ID,
        "codex",
        [],
        expires_in_seconds=960,
        credential_id=None,
        state_metadata={CODEX_LOGIN_STATE_KEY: "login-123"},
    )


def test_codex_login_start_failure_is_sanitized(caplog):
    with (
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.settings") as settings,
    ):
        settings.config.frontend_base_url = "http://localhost:3000"
        coordinator.start = AsyncMock(
            side_effect=RuntimeError(
                "user_code=ABCD-EFGH verification_url=https://secret.example"
            )
        )

        response = client.get("/codex/login")

    assert response.status_code == 503
    assert response.json()["detail"] == ("ChatGPT sign-in is temporarily unavailable")
    assert "ABCD-EFGH" not in response.text
    assert "ABCD-EFGH" not in caplog.text
    assert "secret.example" not in response.text
    assert "secret.example" not in caplog.text


def test_codex_login_preserves_state_store_error_when_cancel_fails():
    login = CodexDeviceLogin(
        login_id="login-123",
        verification_url="https://auth.openai.com/codex/device",
        user_code="ABCD-EFGH",
    )
    with (
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
        patch("backend.api.features.integrations.router.settings") as settings,
    ):
        settings.config.frontend_base_url = "http://localhost:3000"
        settings.config.codex_login_timeout_seconds = 900
        coordinator.start = AsyncMock(return_value=login)
        coordinator.cancel = AsyncMock(side_effect=RuntimeError("cancel failed"))
        manager.store.store_state_token = AsyncMock(
            side_effect=RuntimeError("state store failed")
        )

        with pytest.raises(RuntimeError, match="state store failed"):
            client.get("/codex/login")

    coordinator.cancel.assert_awaited_once_with(TEST_USER_ID, "login-123")


def test_codex_login_rejects_user_without_required_plan():
    gate = AsyncMock(
        side_effect=fastapi.HTTPException(
            status_code=402,
            detail="A Max plan or higher is required to use ChatGPT.",
        )
    )
    with (
        patch(
            "backend.api.features.integrations.router.enforce_codex_access_http",
            new=gate,
        ),
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
    ):
        coordinator.start = AsyncMock()
        manager.store.store_state_token = AsyncMock()

        response = client.get("/codex/login")

    assert response.status_code == 402
    assert response.json()["detail"] == (
        "A Max plan or higher is required to use ChatGPT."
    )
    gate.assert_awaited_once_with(TEST_USER_ID)
    coordinator.start.assert_not_awaited()
    manager.store.store_state_token.assert_not_awaited()


def test_codex_callback_persists_one_safe_credential():
    credentials = _credentials()
    with (
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
    ):
        manager.store.verify_state_token = AsyncMock(return_value=_oauth_state())
        coordinator.complete = AsyncMock(return_value=credentials)

        response = client.post(
            "/codex/callback",
            json={"code": "login-123", "state_token": "state-123"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "codex-credential",
        "provider": "codex",
        "type": "oauth2",
        "title": "ChatGPT for Codex",
        "scopes": [],
        "username": "user@example.com",
        "host": None,
        "is_managed": False,
    }
    raw_response = response.text
    assert "access-secret" not in raw_response
    assert "refresh-secret" not in raw_response
    assert "provider-secret" not in raw_response


def test_codex_callback_rejects_user_without_required_plan_before_state_use():
    gate = AsyncMock(
        side_effect=fastapi.HTTPException(
            status_code=402,
            detail="A Max plan or higher is required to use ChatGPT.",
        )
    )
    with (
        patch(
            "backend.api.features.integrations.router.enforce_codex_access_http",
            new=gate,
        ),
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
    ):
        coordinator.complete = AsyncMock()
        manager.store.verify_state_token = AsyncMock()

        response = client.post(
            "/codex/callback",
            json={"code": "login-123", "state_token": "state-123"},
        )

    assert response.status_code == 402
    gate.assert_awaited_once_with(TEST_USER_ID)
    manager.store.verify_state_token.assert_not_awaited()
    coordinator.complete.assert_not_awaited()


def test_codex_callback_rejects_login_id_mismatch():
    with (
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
    ):
        manager.store.verify_state_token = AsyncMock(return_value=_oauth_state())
        coordinator.complete = AsyncMock()

        response = client.post(
            "/codex/callback",
            json={"code": "other-login", "state_token": "state-123"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Codex login completion"
    coordinator.complete.assert_not_awaited()


def test_codex_callback_rejects_non_ascii_login_id_without_server_error():
    with (
        patch(
            "backend.api.features.integrations.router.codex_login_coordinator"
        ) as coordinator,
        patch("backend.api.features.integrations.router.creds_manager") as manager,
    ):
        manager.store.verify_state_token = AsyncMock(return_value=_oauth_state())
        coordinator.complete = AsyncMock()

        response = client.post(
            "/codex/callback",
            json={"code": "lógin-123", "state_token": "state-123"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid Codex login completion"
    coordinator.complete.assert_not_awaited()


def test_client_cannot_post_codex_credentials():
    with patch("backend.api.features.integrations.router.creds_manager") as manager:
        manager.create = AsyncMock()
        response = client.post(
            "/codex/credentials",
            json={
                "id": "injected",
                "provider": "codex",
                "type": "oauth2",
                "title": "Injected",
                "access_token": "stolen-access",
                "refresh_token": "stolen-refresh",
                "scopes": [],
                "refresh_strategy": "provider_runtime",
                "provider_state": "stolen-provider-state",
                "provider_state_version": 1,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Codex credentials must be created through ChatGPT sign-in"
    )
    manager.create.assert_not_awaited()


def test_device_login_page_reads_fragment_and_contains_no_tokens():
    page = render_device_login_page("test-nonce")

    assert "<script>alert(1)</script>" not in page
    assert "ABCD-&lt;EFGH&gt;" not in page
    assert "URLSearchParams" in page
    assert "access_token" not in page
    assert "refresh_token" not in page
    assert "/status" in page
    assert "/cancel" in page
    assert "pagehide" in page
    assert "/auth/integrations/oauth_callback" in page
    assert "try {\n          loginID = decodeURIComponent" in page
    assert "response.status === 404" in page
    assert "pollDeadline" in page


def test_device_login_status_enforces_user_ownership():
    with (
        patch(
            "backend.api.features.integrations.codex.codex_login_coordinator"
        ) as coordinator,
    ):
        coordinator.get = AsyncMock(return_value=None)
        response = client.get("/codex/device-login/another-users-login/status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Codex login not found"


def test_device_login_status_returns_only_nonsecret_state():
    state = CodexDeviceLoginState(
        status="completed",
    )
    with (
        patch(
            "backend.api.features.integrations.codex.codex_login_coordinator"
        ) as coordinator,
    ):
        coordinator.get = AsyncMock(return_value=state)
        response = client.get("/codex/device-login/login-123/status")

    assert response.status_code == 200
    assert response.json() == {"status": "completed", "error": None}
    assert "ABCD-EFGH" not in response.text


def test_device_login_cancel_is_best_effort_when_runtime_cancel_fails():
    with patch(
        "backend.api.features.integrations.codex.codex_login_coordinator"
    ) as coordinator:
        coordinator.cancel = AsyncMock(side_effect=RuntimeError("dead runtime"))
        response = client.post("/codex/device-login/login-123/cancel")

    assert response.status_code == 204
    assert response.content == b""


def test_device_login_cancel_returns_not_found_for_unknown_login():
    with patch(
        "backend.api.features.integrations.codex.codex_login_coordinator"
    ) as coordinator:
        coordinator.cancel = AsyncMock(return_value=False)
        response = client.post("/codex/device-login/missing/cancel")

    assert response.status_code == 404


def test_provider_discovery_includes_codex_when_user_has_access():
    with (
        patch("backend.blocks.load_all_blocks"),
        patch(
            "backend.api.features.integrations.router.get_all_provider_names",
            return_value=["codex", "github"],
        ),
        patch(
            "backend.api.features.integrations.router.get_provider_description",
            return_value=None,
        ),
        patch(
            "backend.api.features.integrations.router.get_supported_auth_types",
            return_value=[],
        ),
    ):
        response = client.get("/providers")

    assert response.status_code == 200
    assert [provider["name"] for provider in response.json()] == ["codex", "github"]


def test_provider_discovery_omits_codex_when_user_lacks_access():
    access = AsyncMock(return_value=False)
    with (
        patch("backend.blocks.load_all_blocks"),
        patch(
            "backend.api.features.integrations.router.has_codex_access_for_discovery",
            new=access,
        ),
        patch(
            "backend.api.features.integrations.router.get_all_provider_names",
            return_value=["codex", "github"],
        ),
        patch(
            "backend.api.features.integrations.router.get_provider_description",
            return_value=None,
        ),
        patch(
            "backend.api.features.integrations.router.get_supported_auth_types",
            return_value=[],
        ),
    ):
        response = client.get("/providers")

    assert response.status_code == 200
    assert [provider["name"] for provider in response.json()] == ["github"]
    access.assert_awaited_once_with(TEST_USER_ID)


def test_provider_discovery_remains_public_and_omits_codex_anonymously():
    access = AsyncMock(return_value=True)
    app.dependency_overrides[get_optional_user_id] = lambda: None
    with (
        patch("backend.blocks.load_all_blocks"),
        patch(
            "backend.api.features.integrations.router.has_codex_access_for_discovery",
            new=access,
        ),
        patch(
            "backend.api.features.integrations.router.get_all_provider_names",
            return_value=["codex", "github"],
        ),
        patch(
            "backend.api.features.integrations.router.get_provider_description",
            return_value=None,
        ),
        patch(
            "backend.api.features.integrations.router.get_supported_auth_types",
            return_value=[],
        ),
    ):
        response = client.get("/providers")

    assert response.status_code == 200
    assert [provider["name"] for provider in response.json()] == ["github"]
    access.assert_not_awaited()


def test_device_login_page_has_no_store_and_restrictive_browser_headers():
    state = CodexDeviceLoginState(status="pending")
    with (
        patch(
            "backend.api.features.integrations.codex.codex_login_coordinator"
        ) as coordinator,
    ):
        coordinator.get = AsyncMock(return_value=state)
        response = client.get("/codex/device-login/login-123")

    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "nonce=" in response.text


def test_build_device_login_url_uses_same_origin_proxy():
    url = build_device_login_url(
        "https://platform.example",
        CodexDeviceLogin(
            login_id="login/id",
            verification_url="https://auth.openai.com/codex/device",
            user_code="ABCD EFGH",
        ),
        "state token",
    )

    assert url == (
        "https://platform.example/api/proxy/api/integrations/codex/"
        "device-login/login%2Fid#state=state+token&verification_url="
        "https%3A%2F%2Fauth.openai.com%2Fcodex%2Fdevice&user_code=ABCD+EFGH"
    )


def test_build_device_login_cancel_url_uses_same_origin_proxy():
    url = build_device_login_cancel_url(
        CodexDeviceLogin(
            login_id="login/id",
            verification_url="https://auth.openai.com/codex/device",
            user_code="ABCD EFGH",
        ),
    )

    assert url == ("/api/proxy/api/integrations/codex/device-login/login%2Fid/cancel")


def test_codex_account_uses_user_owned_credential_lease():
    lease = MagicMock()
    lease.credentials = _credentials()
    lease.release = AsyncMock()
    account = CodexAccountSnapshot(
        connected=True,
        requires_openai_auth=False,
        account_type="chatgpt",
        email="user@example.com",
        plan_type="plus",
    )
    with (
        patch(
            "backend.api.features.integrations.codex.codex_credentials_manager"
        ) as manager,
        patch(
            "backend.api.features.integrations.codex._get_codex_transport"
        ) as get_transport,
    ):
        manager.acquire_lease = AsyncMock(return_value=lease)
        get_transport.return_value.account = AsyncMock(return_value=account)

        response = client.get("/codex/credentials/codex-credential/account")

    assert response.status_code == 200
    assert response.json() == account.model_dump(mode="json")
    manager.acquire_lease.assert_awaited_once_with(TEST_USER_ID, "codex-credential")
    get_transport.return_value.account.assert_awaited_once_with(lease)
    lease.release.assert_awaited_once()


def test_codex_rate_limits_returns_safe_snapshot():
    lease = MagicMock()
    lease.credentials = _credentials()
    lease.release = AsyncMock()
    rate_limits = CodexRateLimitsSnapshot(
        plan_type="plus",
        primary=CodexRateLimitWindow(
            used_percent=42,
            window_duration_mins=300,
            resets_at=4_000_000_000,
        ),
        has_credits=True,
    )
    with (
        patch(
            "backend.api.features.integrations.codex.codex_credentials_manager"
        ) as manager,
        patch(
            "backend.api.features.integrations.codex._get_codex_transport"
        ) as get_transport,
    ):
        manager.acquire_lease = AsyncMock(return_value=lease)
        get_transport.return_value.rate_limits = AsyncMock(return_value=rate_limits)

        response = client.get("/codex/credentials/codex-credential/rate-limits")

    assert response.status_code == 200
    assert response.json() == rate_limits.model_dump(mode="json")
    assert "access-secret" not in response.text
    assert "refresh-secret" not in response.text
    lease.release.assert_awaited_once()


def test_codex_delete_logs_out_then_deletes_even_if_logout_fails():
    credentials = _credentials()
    with (
        patch("backend.api.features.integrations.router.creds_manager") as manager,
        patch(
            "backend.api.features.integrations.router.remove_all_webhooks_for_credentials",
            new=AsyncMock(),
        ),
        patch(
            "backend.api.features.integrations.router.revoke_codex_credentials",
            new=AsyncMock(return_value=False),
        ) as revoke,
    ):
        manager.store.get_creds_by_id = AsyncMock(return_value=credentials)
        manager.delete = AsyncMock()

        response = client.delete("/codex/credentials/codex-credential?force=true")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "revoked": False}
    revoke.assert_awaited_once_with(manager, TEST_USER_ID, "codex-credential")
    manager.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_logout_uses_and_releases_exclusive_lease():
    lease = MagicMock()
    lease.credentials = _credentials()
    lease.delete = AsyncMock()
    lease.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=lease)
    with patch(
        "backend.api.features.integrations.codex._get_codex_transport"
    ) as get_transport:
        get_transport.return_value.logout = AsyncMock()

        revoked = await revoke_codex_credentials(
            manager, TEST_USER_ID, "codex-credential"
        )

    assert revoked is True
    manager.acquire_lease.assert_awaited_once_with(TEST_USER_ID, "codex-credential")
    get_transport.return_value.logout.assert_awaited_once_with(lease)
    lease.delete.assert_awaited_once()
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_codex_logout_failure_is_redacted_and_still_releases(caplog):
    lease = MagicMock()
    lease.credentials = _credentials()
    lease.delete = AsyncMock()
    lease.release = AsyncMock()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=lease)
    with patch(
        "backend.api.features.integrations.codex._get_codex_transport"
    ) as get_transport:
        get_transport.return_value.logout = AsyncMock(
            side_effect=RuntimeError("access-secret refresh-secret")
        )

        revoked = await revoke_codex_credentials(
            manager, TEST_USER_ID, "codex-credential"
        )

    assert revoked is False
    assert "access-secret" not in caplog.text
    assert "refresh-secret" not in caplog.text
    lease.delete.assert_awaited_once()
    lease.release.assert_awaited_once()
