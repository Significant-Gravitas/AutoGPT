from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.api.features.partner_embed import llm_route


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


@pytest.mark.asyncio
async def test_defaults_to_platform_transport(monkeypatch):
    monkeypatch.delenv("PARTNER_EMBED_CODEX_CREDENTIAL_ID", raising=False)

    assert await llm_route.resolve_embed_llm_route("user-1") == ("platform", None)


@pytest.mark.asyncio
async def test_uses_valid_user_owned_codex_credential(monkeypatch, mocker):
    monkeypatch.setenv("PARTNER_EMBED_CODEX_CREDENTIAL_ID", "credential-1")
    credentials = SimpleNamespace(provider="codex", type="oauth2")
    get = AsyncMock(return_value=credentials)
    mocker.patch(
        "backend.api.features.partner_embed.llm_route.IntegrationCredentialsManager.get",
        get,
    )
    validate = mocker.patch(
        "backend.api.features.partner_embed.llm_route.bundle_from_credentials"
    )

    result = await llm_route.resolve_embed_llm_route("user-1")

    assert result == ("codex", "credential-1")
    get.assert_awaited_once_with("user-1", "credential-1")
    validate.assert_called_once_with(credentials)


@pytest.mark.asyncio
async def test_rejects_credential_not_owned_by_partner_user(monkeypatch, mocker):
    monkeypatch.setenv("PARTNER_EMBED_CODEX_CREDENTIAL_ID", "credential-1")
    mocker.patch(
        "backend.api.features.partner_embed.llm_route.IntegrationCredentialsManager.get",
        AsyncMock(return_value=None),
    )

    with pytest.raises(HTTPException) as error:
        await llm_route.resolve_embed_llm_route("user-1")

    assert error.value.status_code == 503
    assert error.value.detail == "partner_chat_credential_unavailable"


@pytest.mark.asyncio
async def test_rejects_wrong_credential_type(monkeypatch, mocker):
    monkeypatch.setenv("PARTNER_EMBED_CODEX_CREDENTIAL_ID", "credential-1")
    mocker.patch(
        "backend.api.features.partner_embed.llm_route.IntegrationCredentialsManager.get",
        AsyncMock(return_value=SimpleNamespace(provider="openai", type="api_key")),
    )

    with pytest.raises(HTTPException) as error:
        await llm_route.resolve_embed_llm_route("user-1")

    assert error.value.status_code == 503
