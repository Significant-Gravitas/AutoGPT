from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import fastapi
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.testclient import TestClient

from backend.data.llm_provider_settings import PersistedLlmProviderSettings
from backend.util.llm.config import resolve_chat_config
from backend.util.llm.diagnostics import ProviderDiagnosticResult
from backend.util.llm.runtime_config import merge_effective_llm_config

from .llm_settings_routes import router

app = fastapi.FastAPI()
app.include_router(router, prefix="/api/admin")
client = TestClient(app)


def _effective():
    env = {
        "CHAT_PROVIDER": "deepseek",
        "CHAT_API_KEY": "secret-value",
        "CHAT_MODEL": "deepseek-v4-flash",
    }
    return merge_effective_llm_config(
        resolve_chat_config(env),
        environment=env,
    )


def _persisted() -> PersistedLlmProviderSettings:
    now = datetime.now(UTC)
    return PersistedLlmProviderSettings(
        enabled=True,
        provider="deepseek",
        use_local=False,
        base_url="https://api.deepseek.com",
        encrypted_api_key="existing-ciphertext",
        model="deepseek-v4-flash",
        title_model="deepseek-v4-flash",
        fast_standard_model="deepseek-v4-flash",
        fast_advanced_model="deepseek-v4-pro",
        thinking_standard_model="deepseek-v4-flash",
        thinking_advanced_model="deepseek-v4-pro",
        claude_agent_fallback_model="deepseek-v4-pro",
        request_timeout_s=20,
        max_retries=1,
        local_request_timeout_s=20,
        store_embedding_model="text-embedding-3-small",
        created_at=now,
        updated_at=now,
    )


def test_get_settings_masks_api_key(mock_jwt_admin) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    with (
        patch(
            "backend.api.features.admin.llm_settings_routes.get_llm_provider_settings",
            new=AsyncMock(return_value=_persisted()),
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes.resolve_effective_llm_config",
            new=AsyncMock(return_value=_effective()),
        ),
    ):
        response = client.get("/api/admin/llm/settings")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["api_key_configured"] is True
    assert data["api_key_masked"] == "********"
    assert "secret-value" not in response.text


def test_put_empty_api_key_keeps_existing_ciphertext(mock_jwt_admin) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    saved = _persisted()
    upsert = AsyncMock(return_value=saved)
    with (
        patch(
            "backend.api.features.admin.llm_settings_routes.get_llm_provider_settings",
            new=AsyncMock(return_value=saved),
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes._resolve_for_request",
            new=AsyncMock(return_value=(_effective(), object())),
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes.resolve_effective_llm_config",
            new=AsyncMock(return_value=_effective()),
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes.upsert_llm_provider_settings",
            new=upsert,
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes.encrypt_llm_api_key"
        ) as encrypt,
    ):
        response = client.put(
            "/api/admin/llm/settings",
            json={"api_key": "", "request_timeout_s": 20},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert upsert.await_args.kwargs["encrypted_api_key"] == "existing-ciphertext"
    encrypt.assert_not_called()


def test_test_endpoint_uses_unsaved_config(mock_jwt_admin) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    effective = _effective()
    diagnostic = ProviderDiagnosticResult(
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        base_url_host="api.deepseek.com",
        config_source="override",
        latency_ms=12,
        success=True,
        response="ok",
    )
    diagnose = AsyncMock(return_value=diagnostic)
    with (
        patch(
            "backend.api.features.admin.llm_settings_routes._resolve_for_request",
            new=AsyncMock(return_value=(effective, object())),
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes.diagnose_chat_provider",
            new=diagnose,
        ),
        patch(
            "backend.api.features.admin.llm_settings_routes.upsert_llm_provider_settings"
        ) as upsert,
    ):
        response = client.post(
            "/api/admin/llm/settings/test",
            json={"provider": "deepseek", "model": "deepseek-v4-pro"},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["response"] == "ok"
    assert diagnose.await_args.kwargs["config"] is effective
    upsert.assert_not_called()


def test_invalid_provider_is_rejected(mock_jwt_admin) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    response = client.put(
        "/api/admin/llm/settings",
        json={"provider": "invalid-provider"},
    )
    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_provider_presets_include_direct_deepseek(mock_jwt_admin) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    response = client.get("/api/admin/llm/providers")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    deepseek = next(item for item in response.json() if item["provider"] == "deepseek")
    assert deepseek["base_url"] == "https://api.deepseek.com"
    assert deepseek["models"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert deepseek["pricing_status"] == "unknown"
