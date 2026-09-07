import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.api.external import credential_access
from backend.api.external.v1 import integrations, routes
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.db_accessors import LiveResourceLeaseGuard


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    yield


@pytest.fixture
def auth():
    return APIAuthorizationInfo(
        user_id="user",
        scopes=list(APIKeyPermission),
        type="api_key",
        created_at=datetime.now(UTC),
        organization_id="org",
        team_id_restriction="team",
    )


@pytest.fixture
def access(mocker):
    state = SimpleNamespace(request_active=False, lease_active=False, allowed=True)
    lease_db = MagicMock(is_live_resource_lease_active=AsyncMock(return_value=True))

    @asynccontextmanager
    async def barrier(*_args, **_kwargs):
        state.request_active = True
        try:
            yield state.allowed
        finally:
            state.request_active = False

    @asynccontextmanager
    async def lease(*_args, **_kwargs):
        state.lease_active = True
        try:
            yield LiveResourceLeaseGuard(lease_db, "lease")
        finally:
            state.lease_active = False

    mocker.patch.object(routes, "live_resource_access_barrier", new=barrier)
    mocker.patch.object(routes, "live_resource_lease", new=lease)
    mocker.patch.object(integrations, "live_resource_permission_barrier", new=barrier)
    mocker.patch.object(credential_access, "live_resource_lease", new=lease)
    mocker.patch.object(
        credential_access, "live_resource_permission_barrier", new=barrier
    )
    state.lease_db, state.lease, state.barrier = lease_db, lease, barrier
    return state


@pytest.mark.asyncio
async def test_direct_block_holds_dedicated_lease_without_request_transaction(
    access, auth, mocker
):
    async def execute(*_args):
        assert not access.request_active
        assert access.lease_active
        return {"result": ["ok"]}

    mocker.patch.object(routes, "_execute_graph_block_live", side_effect=execute)
    assert await routes.execute_graph_block("block", {}, auth) == {"result": ["ok"]}
    assert not access.lease_active


@pytest.mark.asyncio
async def test_direct_block_lease_loss_cancels_external_work(access, auth, mocker):
    cancelled = asyncio.Event()

    async def execute(*_args):
        access.lease_db.is_live_resource_lease_active.return_value = False
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    mocker.patch.object(routes, "_execute_graph_block_live", side_effect=execute)
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(routes.execute_graph_block("block", {}, auth), 1)
    assert exc.value.status_code == 403
    assert cancelled.is_set()
    assert not access.lease_active


@pytest.fixture
def oauth(mocker):
    state = SimpleNamespace(
        callback_url="https://app.example/callback",
        scopes=["read"],
        code_verifier="verifier",
        state_metadata={},
    )
    handler = MagicMock(handle_default_scopes=MagicMock(return_value=["read"]))
    credentials = SimpleNamespace(
        id="credential",
        provider="github",
        type="oauth2",
        title="GitHub",
        scopes=["read"],
        username="user",
    )
    handler.exchange_code_for_tokens = AsyncMock(return_value=credentials)
    manager = MagicMock(
        store=MagicMock(verify_state_token=AsyncMock(return_value=state)),
        create=AsyncMock(),
    )
    mocker.patch.object(integrations, "creds_manager", manager)
    mocker.patch.object(
        integrations, "_get_oauth_handler_for_external", return_value=handler
    )
    return SimpleNamespace(handler=handler, credentials=credentials, manager=manager)


@pytest.mark.asyncio
async def test_oauth_exchange_releases_request_transaction(access, oauth, auth):
    async def exchange(*_args):
        assert not access.request_active
        assert access.lease_active
        return oauth.credentials

    oauth.handler.exchange_code_for_tokens.side_effect = exchange
    response = await integrations.complete_oauth(
        "github",
        integrations.OAuthCompleteRequest(code="code", state_token="state"),
        auth,
    )
    assert response.credentials_id == "credential"
    oauth.manager.create.assert_awaited_once_with("user", oauth.credentials)


@pytest.mark.asyncio
async def test_oauth_management_permission_denial_prevents_exchange(
    access, oauth, auth
):
    access.allowed = False
    with pytest.raises(HTTPException) as exc:
        await integrations.complete_oauth(
            "github",
            integrations.OAuthCompleteRequest(code="code", state_token="state"),
            auth,
        )
    assert exc.value.status_code == 403
    oauth.handler.exchange_code_for_tokens.assert_not_awaited()
    oauth.manager.create.assert_not_awaited()
    assert not access.lease_active


@pytest.mark.asyncio
async def test_oauth_lease_loss_cancels_exchange_before_saving(access, oauth, auth):
    cancelled = asyncio.Event()

    async def exchange(*_args):
        access.lease_db.is_live_resource_lease_active.return_value = False
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    oauth.handler.exchange_code_for_tokens.side_effect = exchange
    with pytest.raises(HTTPException) as exc:
        await asyncio.wait_for(
            integrations.complete_oauth(
                "github",
                integrations.OAuthCompleteRequest(code="code", state_token="state"),
                auth,
            ),
            1,
        )
    assert exc.value.status_code == 403
    assert cancelled.is_set()
    oauth.manager.create.assert_not_awaited()
    assert not access.lease_active


@pytest.mark.asyncio
async def test_oauth_provider_errors_do_not_expose_tokens(access, oauth, auth, caplog):
    oauth.handler.exchange_code_for_tokens.side_effect = RuntimeError("secret-token")
    with pytest.raises(HTTPException) as exc:
        await integrations.complete_oauth(
            "github",
            integrations.OAuthCompleteRequest(code="code", state_token="state"),
            auth,
        )
    assert exc.value.status_code == 400
    assert "secret-token" not in exc.value.detail
    assert "secret-token" not in caplog.text
    oauth.manager.create.assert_not_awaited()
