import asyncio
import concurrent.futures
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.codex import transport as transport_module
from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.models import (
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexModelInfo,
    CodexRateLimitsSnapshot,
)
from backend.integrations.codex.transport import (
    CodexCredentialBusyError,
    CodexCredentialLease,
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


def test_transport_singleton_initializes_once_across_worker_threads() -> None:
    first_constructor_entered = threading.Event()
    release_first_constructor = threading.Event()
    construction_count = 0
    constructed: list[object] = []

    def construct(**_kwargs):
        nonlocal construction_count
        construction_count += 1
        instance = object()
        constructed.append(instance)
        if construction_count == 1:
            first_constructor_entered.set()
            assert release_first_constructor.wait(timeout=2)
        return instance

    with (
        patch.object(transport_module, "_transport", None),
        patch.object(transport_module, "Settings", return_value=MagicMock()),
        patch.object(transport_module, "CodexTransport", side_effect=construct),
        concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(transport_module.get_codex_transport)
        assert first_constructor_entered.wait(timeout=1)
        second = pool.submit(transport_module.get_codex_transport)

        with pytest.raises(concurrent.futures.TimeoutError):
            second.result(timeout=0.05)

        release_first_constructor.set()
        first_result = first.result(timeout=1)
        second_result = second.result(timeout=1)

    assert construction_count == 1
    assert first_result is constructed[0]
    assert second_result is first_result


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


def _invocation_result() -> CodexInvocationResult:
    return CodexInvocationResult(
        response_id="response-1",
        final_response="done",
        status="completed",
    )


def _session_with_rate_limits() -> MagicMock:
    session = MagicMock()
    session.credential_id = "cred-id"
    session.rate_limits = CodexRateLimitsSnapshot(plan_type="pro")
    session.invoke = AsyncMock(return_value=_invocation_result())
    return session


@pytest.mark.asyncio
async def test_direct_invoke_records_the_latest_rate_limits() -> None:
    credentials = _credentials()
    session = _session_with_rate_limits()
    transport = CodexTransport()
    lease = CodexCredentialLease(credentials, session)
    model = CodexModelInfo(
        model="gpt-5.6-sol",
        display_name="GPT-5.6 Sol",
        is_default=True,
        hidden=False,
        default_reasoning_effort="medium",
        supported_reasoning_efforts=["medium"],
    )

    with (
        patch.object(transport, "_session_for", return_value=session),
        patch(
            "backend.integrations.codex.transport.fetch_models",
            new=AsyncMock(return_value=[model]),
        ),
    ):
        await transport.invoke(lease, CodexInvocationRequest(prompt="hello"))

    assert (await transport.rate_limits(lease)).plan_type == "pro"


@pytest.mark.asyncio
async def test_direct_agent_invoke_records_the_latest_rate_limits() -> None:
    credentials = _credentials()
    session = _session_with_rate_limits()
    transport = CodexTransport()
    lease = CodexCredentialLease(credentials, session)

    with patch.object(transport, "_session_for", return_value=session):
        await transport.invoke_agent(
            lease,
            CodexInvocationRequest(prompt="hello"),
            [],
            AsyncMock(),
        )

    assert (await transport.rate_limits(lease)).plan_type == "pro"
