import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.transport import (
    CodexCredentialBusyError,
    CodexTransport,
)


def _credentials(*, legacy: bool = False) -> OAuth2Credentials:
    token = (
        "eyJhbGciOiJub25lIn0." "eyJleHAiOjk5OTk5OTk5OTksImVtYWlsIjoiYUBiLmMifQ." "sig"
    )
    credentials = credentials_from_bundle(
        CodexAuthBundleV1(
            tokens=CodexAuthTokensV1(
                id_token=SecretStr(token),
                access_token=SecretStr(token),
                refresh_token=SecretStr("refresh"),
            ),
            codex_runtime_version="http",
        )
    ).model_copy(update={"id": "cred-id"})
    if legacy:
        return credentials.model_copy(update={"refresh_strategy": "provider_runtime"})
    return credentials


@pytest.mark.asyncio
async def test_runtime_snapshot_does_not_hold_an_exclusive_credential_lease() -> None:
    manager = MagicMock()
    manager.get = AsyncMock(return_value=_credentials())
    manager.acquire_lease = AsyncMock()
    transport = CodexTransport()
    session = MagicMock()

    with (
        patch(
            "backend.integrations.codex.transport.IntegrationCredentialsManager",
            return_value=manager,
        ),
        patch.object(transport, "_session_for", return_value=session),
    ):
        lease = await transport.acquire_runtime_lease(
            "user-a", "cred-id", lock_timeout_seconds=1
        )

    assert lease.credentials.id == "cred-id"
    manager.get.assert_awaited_once_with("user-a", "cred-id")
    manager.acquire_lease.assert_not_awaited()
    await lease.release()


@pytest.mark.asyncio
async def test_runtime_snapshot_times_out_during_credential_contention() -> None:
    blocker = asyncio.Event()

    async def blocked_get(_user_id: str, _credential_id: str):
        await blocker.wait()

    manager = MagicMock()
    manager.get = AsyncMock(side_effect=blocked_get)

    with patch(
        "backend.integrations.codex.transport.IntegrationCredentialsManager",
        return_value=manager,
    ):
        with pytest.raises(CodexCredentialBusyError, match="codex_credential_busy"):
            await CodexTransport().acquire_runtime_lease(
                "user-a", "cred-id", lock_timeout_seconds=0.01
            )


@pytest.mark.asyncio
async def test_legacy_runtime_credential_is_migrated_before_use() -> None:
    legacy = _credentials(legacy=True)
    migrated = _credentials()
    legacy_lease = MagicMock()
    legacy_lease.credentials = legacy
    legacy_lease.checkpoint = AsyncMock()
    legacy_lease.release = AsyncMock()
    manager = MagicMock()
    manager.get = AsyncMock(side_effect=[legacy, migrated])
    manager.acquire_lease = AsyncMock(return_value=legacy_lease)
    transport = CodexTransport()

    with (
        patch(
            "backend.integrations.codex.transport.IntegrationCredentialsManager",
            return_value=manager,
        ),
        patch.object(transport, "_session_for", return_value=MagicMock()),
    ):
        lease = await transport.acquire_runtime_lease(
            "user-a", "cred-id", lock_timeout_seconds=1
        )

    checkpointed = legacy_lease.checkpoint.await_args.args[0]
    assert checkpointed.refresh_strategy == "oauth_handler"
    legacy_lease.release.assert_awaited_once()
    assert lease.credentials.refresh_strategy == "oauth_handler"
