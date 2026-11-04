"""Private-org credential boundaries with PostgreSQL and Redis, fake providers only."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from autogpt_libs.auth import RequestContext, get_request_context, get_user_id
from fastapi import FastAPI, Request
from pydantic import SecretStr

from backend.api.features.integrations import router as routes
from backend.data import db, user
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.credential_scope import credential_scope
from backend.integrations.creds_manager import IntegrationCredentialsManager

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(loop_scope="session")
async def owner():
    # The client is shared with the rest of the session: leave it as found.
    connected_here = not db.is_connected()
    await db.connect()
    owner_id = str(uuid.uuid4())
    await db.prisma.user.create(
        data={"id": owner_id, "email": f"{owner_id}@example.test"}
    )
    orgs = []
    try:
        for name in ["personal", "business"]:
            org = await db.prisma.organization.create(
                data={
                    "name": name,
                    "slug": f"{name}-{owner_id}",
                    "isPersonal": name == "personal",
                    "Members": {"create": {"userId": owner_id, "isOwner": True}},
                }
            )
            orgs.append(org.id)
        yield owner_id, *orgs
    finally:
        for org_id in orgs:
            await db.prisma.organization.delete(where={"id": org_id})
        await db.prisma.user.delete(where={"id": owner_id})
        if connected_here:
            await db.disconnect()


def key(title: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        provider="github", api_key=SecretStr("fake-token"), title=title
    )


@pytest_asyncio.fixture(loop_scope="session")
async def client(owner, monkeypatch):
    owner_id, personal, business = owner
    app = FastAPI()
    app.include_router(routes.router)

    async def context(request: Request):
        org_id = request.headers.get("X-Org-Id", personal)
        assert org_id in (personal, business)
        return RequestContext(
            owner_id, org_id, None, True, False, False, False, False, "ACTIVE"
        )

    app.dependency_overrides[get_request_context] = context
    # Dev's paywall dependency reads the signed-in user from the token, and this
    # owner has no plan: sign them in and treat them as paid, as the API
    # tests' conftest does.
    app.dependency_overrides[get_user_id] = lambda: owner_id
    monkeypatch.setattr(
        "backend.copilot.rate_limit.is_user_paywalled", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(routes, "is_user_paywalled", AsyncMock(return_value=False))
    monkeypatch.setattr(routes, "_ensure_managed_credentials_bounded", AsyncMock())
    monkeypatch.setattr(
        routes.product_analytics, "track_integration_connected", lambda **_: None
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def test_same_owner_crud_and_explicit_id_acquisition(owner, client):
    owner_id, personal, business = owner
    manager = IntegrationCredentialsManager()
    a, b = key("A"), key("B")
    for org, cred in [(personal, a), (business, b)]:
        response = await client.post(
            "/github/credentials",
            headers={"X-Org-Id": org},
            json=cred.model_dump(mode="json"),
        )
        assert response.status_code == 201, response.text
    for org, expected in [(personal, a.id), (business, b.id)]:
        response = await client.get("/github/credentials", headers={"X-Org-Id": org})
        assert response.status_code == 200, response.text
        assert [item["id"] for item in response.json()] == [expected]
    assert [c.id for c in await user.get_user_credentials(owner_id)] == [a.id]
    with pytest.raises(ValueError):
        await manager.scoped(personal).acquire_lease(owner_id, b.id)
    with pytest.raises(ValueError):
        await manager.scoped(personal).update(
            owner_id, b.model_copy(update={"title": "wrong"})
        )
    await manager.scoped(business).update(
        owner_id, b.model_copy(update={"title": "new"})
    )
    assert (await manager.scoped(business).get(owner_id, b.id)).title == "new"
    denied = await client.delete(
        f"/github/credentials/{b.id}", headers={"X-Org-Id": personal}
    )
    assert denied.status_code == 404
    deleted = await client.delete(
        f"/github/credentials/{b.id}", headers={"X-Org-Id": business}
    )
    assert deleted.status_code == 200, deleted.text
    assert (await manager.scoped(personal).get(owner_id, a.id)).id == a.id
    assert await manager.scoped(business).get(owner_id, b.id) is None


async def test_oauth_callback_after_switch_preserves_origin(owner, client, monkeypatch):
    owner_id, personal, business = owner
    manager = IntegrationCredentialsManager()
    token, _ = await manager.scoped(personal).store.store_state_token(
        owner_id, "github", ["repo"]
    )
    credential = OAuth2Credentials(
        provider="github", access_token=SecretStr("fake-oauth"), scopes=["repo"]
    )
    handler = SimpleNamespace(
        client_secret="",
        exchange_code_for_tokens=AsyncMock(return_value=credential),
        handle_default_scopes=lambda scopes: scopes,
    )
    monkeypatch.setattr(routes, "_get_provider_oauth_handler", lambda *_: handler)
    response = await client.post(
        "/github/callback",
        headers={"X-Org-Id": business},
        json={"code": "fake-code", "state_token": token},
    )
    assert response.status_code == 200, response.text
    assert (
        await manager.scoped(personal).get(owner_id, credential.id)
    ).id == credential.id
    assert await manager.scoped(business).get(owner_id, credential.id) is None


async def test_full_replace_is_local_and_foreign_id_is_rejected(owner):
    owner_id, personal, business = owner
    a, b, new = key("A"), key("B"), key("new")
    await user.set_user_credentials(owner_id, [a], personal)
    await user.set_user_credentials(owner_id, [b], business)
    await user.set_user_credentials(owner_id, [new], business)
    assert [c.id for c in await user.get_user_credentials(owner_id, personal)] == [a.id]
    assert [c.id for c in await user.get_user_credentials(owner_id, business)] == [
        new.id
    ]
    from prisma.errors import UniqueViolationError

    with pytest.raises(UniqueViolationError):
        await user.set_user_credentials(owner_id, [new], personal)
    assert [c.id for c in await user.get_user_credentials(owner_id, personal)] == [a.id]


async def test_worker_uses_saved_scope_after_switch(owner, monkeypatch):
    from backend.executor import manager as execution

    owner_id, personal, business = owner
    a, b = key("A"), key("B")
    await user.set_user_credentials(owner_id, [a], personal)
    await user.set_user_credentials(owner_id, [b], business)
    manager = IntegrationCredentialsManager()

    async def run(*args):
        lease = await manager.acquire_lease(owner_id, a.id)
        await lease.release()
        with pytest.raises(ValueError):
            await manager.acquire_lease(owner_id, b.id)
        yield "output", "saved-org"

    monkeypatch.setattr(execution, "_execute_node_authorized", run)
    monkeypatch.setattr(execution, "_revalidate_owner_credentials", AsyncMock())
    monkeypatch.setattr(execution, "get_db_async_client", lambda: None)
    saved = SimpleNamespace(
        execution_context=SimpleNamespace(
            credentials_grant_id=None, organization_id=personal
        ),
        user_id=owner_id,
        graph_id="graph",
        graph_version=1,
    )
    with credential_scope(business):
        assert [value async for value in execution.execute_node(None, saved, None)] == [
            ("output", "saved-org")
        ]
        assert [
            c.id for c in await manager.store.get_creds_by_provider(owner_id, "github")
        ] == [b.id]


async def test_device_callback_after_switch_preserves_origin(
    owner, client, monkeypatch
):
    from backend.integrations.oauth.device_base import DeviceAuthPollResult

    owner_id, personal, business = owner
    manager = IntegrationCredentialsManager()
    token, _ = await manager.scoped(personal).store.store_state_token(
        owner_id,
        "github",
        ["repo"],
        state_metadata={"device_code": "fake-device", "interval": 1},
    )
    credential = OAuth2Credentials(
        provider="github", access_token=SecretStr("fake-device-token"), scopes=["repo"]
    )
    handler = SimpleNamespace(
        poll_for_tokens=AsyncMock(
            return_value=DeviceAuthPollResult(status="approved", credentials=credential)
        ),
        revoke_tokens=AsyncMock(),
    )
    monkeypatch.setattr(routes, "_get_device_auth_handler", lambda *_: handler)
    response = await client.post(
        "/github/device-auth/poll",
        headers={"X-Org-Id": business},
        json={"state_token": token},
    )
    assert response.status_code == 200, response.text
    assert response.json()["credentials"]["id"] == credential.id
    assert await manager.scoped(personal).exists(owner_id, credential.id)
    assert not await manager.scoped(business).exists(owner_id, credential.id)
    handler.revoke_tokens.assert_not_called()


async def test_lease_checkpoint_and_delete_remain_bound_after_switch(owner):
    owner_id, personal, business = owner
    manager = IntegrationCredentialsManager()
    a = OAuth2Credentials(
        provider="codex",
        access_token=SecretStr("fake"),
        scopes=[],
        refresh_strategy="provider_runtime",
    )
    b = key("B")
    await user.set_user_credentials(owner_id, [a], personal)
    await user.set_user_credentials(owner_id, [b], business)
    with credential_scope(personal):
        lease = await manager.acquire_lease(owner_id, a.id)
    try:
        with credential_scope(business):
            await lease.checkpoint(a.model_copy(update={"title": "checkpoint"}))
            assert (
                await manager.scoped(personal).store.get_creds_by_id(owner_id, a.id)
            ).title == "checkpoint"
            await lease.delete()
            assert await manager.exists(owner_id, b.id)
    finally:
        await lease.release()
    assert not await manager.scoped(personal).exists(owner_id, a.id)


async def test_foreign_oauth_snapshot_cannot_refresh(owner):
    owner_id, personal, business = owner
    credential = OAuth2Credentials(
        provider="github", access_token=SecretStr("fake"), scopes=[]
    )
    await user.set_user_credentials(owner_id, [credential], personal)
    with pytest.raises(ValueError):
        await IntegrationCredentialsManager().scoped(business).refresh_if_needed(
            owner_id, credential
        )


async def test_token_and_mcp_discovery_caches_do_not_cross_orgs(owner, monkeypatch):
    from backend.blocks.mcp.helpers import auto_lookup_mcp_credential
    from backend.copilot import integration_creds

    owner_id, personal, business = owner
    monkeypatch.setattr(
        integration_creds, "_ensure_cache_invalidation_listener", lambda: None
    )
    a = key("A").model_copy(update={"api_key": SecretStr("fake-a")})
    b = key("B").model_copy(update={"api_key": SecretStr("fake-b")})
    mcp_a = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("fake-mcp-a"),
        scopes=[],
        metadata={"mcp_server_url": "https://mcp.example.test/mcp"},
    )
    mcp_b = mcp_a.model_copy(
        update={"id": str(uuid.uuid4()), "access_token": SecretStr("fake-mcp-b")}
    )
    await user.set_user_credentials(owner_id, [a, mcp_a], personal)
    await user.set_user_credentials(owner_id, [b, mcp_b], business)
    for org, expected, mcp_expected in [
        (personal, "fake-a", mcp_a),
        (business, "fake-b", mcp_b),
        (personal, "fake-a", mcp_a),
    ]:
        with credential_scope(org):
            assert (
                await integration_creds.get_provider_token(
                    owner_id, "github", allowed_ids=None
                )
                == expected
            )
            found = await auto_lookup_mcp_credential(
                owner_id, "https://mcp.example.test/mcp"
            )
            assert found.id == mcp_expected.id
    with credential_scope(business):
        assert (
            await integration_creds.get_provider_token(
                owner_id, "github", credential_id=a.id, allowed_ids=None
            )
            is None
        )
    integration_creds.invalidate_user_provider_cache(owner_id, "github")


async def test_mcp_callback_replaces_only_origin_server_credentials(
    owner, client, monkeypatch
):
    from backend.api.features.mcp import routes as mcp

    owner_id, personal, business = owner
    manager = IntegrationCredentialsManager()
    a = OAuth2Credentials(
        provider="mcp",
        access_token=SecretStr("fake-mcp-a"),
        scopes=[],
        metadata={"mcp_server_url": "https://mcp.example.test/mcp"},
    )
    b = a.model_copy(update={"id": str(uuid.uuid4())})
    await user.set_user_credentials(owner_id, [a], personal)
    await user.set_user_credentials(owner_id, [b], business)
    token, _ = await manager.scoped(personal).store.store_state_token(
        owner_id,
        "mcp",
        [],
        state_metadata={
            "client_id": "fake-client",
            "authorize_url": "https://mcp.example.test/auth",
            "token_url": "https://mcp.example.test/token",
            "server_url": "https://mcp.example.test/mcp",
        },
    )
    replacement = a.model_copy(update={"id": str(uuid.uuid4())})
    monkeypatch.setattr(mcp.settings.config, "frontend_base_url", "http://test")
    monkeypatch.setattr(
        mcp,
        "MCPOAuthHandler",
        lambda **_: SimpleNamespace(
            exchange_code_for_tokens=AsyncMock(return_value=replacement)
        ),
    )
    client._transport.app.include_router(mcp.router, prefix="/mcp")
    response = await client.post(
        "/mcp/oauth/callback",
        headers={"X-Org-Id": business},
        json={"state_token": token, "code": "fake-code"},
    )
    assert response.status_code == 200, response.text
    assert not await manager.scoped(personal).exists(owner_id, a.id)
    assert await manager.scoped(personal).exists(owner_id, replacement.id)
    assert await manager.scoped(business).exists(owner_id, b.id)


async def test_saved_account_transport_preference_is_ignored_in_other_org(
    owner, monkeypatch
):
    from backend.copilot import transports
    from backend.copilot.transports_test import _codex_credentials

    owner_id, personal, business = owner
    credential = _codex_credentials()
    await user.set_user_credentials(owner_id, [credential], personal)
    monkeypatch.setattr(
        transports, "has_codex_access_for_discovery", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(transports, "is_deployment_chat_available", lambda: True)
    monkeypatch.setattr(
        transports,
        "user_db",
        lambda: SimpleNamespace(
            get_user_default_chat_route=AsyncMock(return_value=("codex", credential.id))
        ),
    )
    with credential_scope(personal):
        assert await transports.resolve_default_chat_route(owner_id) == (
            "codex",
            credential.id,
        )
    with credential_scope(business):
        assert await transports.resolve_default_chat_route(owner_id) == (
            "platform",
            None,
        )
        assert await transports.resolve_default_chat_route(
            owner_id, organization_id=None
        ) == ("codex", credential.id)
        assert await transports.resolve_default_chat_route(
            owner_id, organization_id=personal
        ) == ("codex", credential.id)
        assert all(
            t.credential_id != credential.id
            for t in await transports.get_chat_transports(owner_id)
        )


async def test_managed_provisioning_and_legacy_data_are_org_local(owner, monkeypatch):
    from unittest.mock import MagicMock

    from backend.data.model import UserIntegrations
    from backend.integrations import managed_credentials

    owner_id, personal, business = owner
    integrations = UserIntegrations()
    integrations.managed_credentials.ayrshare_profile_key = SecretStr(
        "fake-legacy-profile"
    )
    await user.update_user_integrations(owner_id, integrations)
    store = IntegrationCredentialsManager().store
    assert (
        await store.scoped(personal).get_user_integrations(owner_id)
    ).managed_credentials.ayrshare_profile_key == SecretStr("fake-legacy-profile")
    assert (
        await store.scoped(business).get_user_integrations(owner_id)
    ).managed_credentials.ayrshare_profile_key is None
    provider = MagicMock(auto_provision=True, provider_name="github")
    provider.is_available = AsyncMock(return_value=True)
    provider.post_provision = AsyncMock()
    provider.provision = AsyncMock(
        side_effect=lambda *_: key("managed").model_copy(update={"is_managed": True})
    )
    monkeypatch.setattr(managed_credentials, "_PROVIDERS", {"github": provider})
    for org in [personal, business, personal, business]:
        await managed_credentials.ensure_managed_credentials(
            owner_id, store.scoped(org)
        )
    assert provider.provision.await_count == 2
    a = await store.scoped(personal).get_creds_by_provider(owner_id, "github")
    b = await store.scoped(business).get_creds_by_provider(owner_id, "github")
    assert a[0].id != b[0].id


async def test_agentmail_managed_resources_use_distinct_org_identity(
    owner, monkeypatch
):
    import agentmail

    from backend.integrations.managed_providers.agentmail import (
        AgentMailManagedProvider,
    )

    owner_id, personal, business = owner
    pods = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(pod_id="fake-pod")),
        api_keys=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(api_key="fake-api-key"))
        ),
    )
    monkeypatch.setattr(
        agentmail, "AsyncAgentMail", lambda **_: SimpleNamespace(pods=pods)
    )
    provider = AgentMailManagedProvider()
    store = IntegrationCredentialsManager().store
    a = await provider.provision(owner_id, store.scoped(personal))
    b = await provider.provision(owner_id, store.scoped(business))
    assert a.metadata["client_id"] == owner_id
    assert b.metadata["client_id"] == f"{owner_id}:{business}"
    assert (
        pods.create.await_args_list[0].kwargs["client_id"]
        != pods.create.await_args_list[1].kwargs["client_id"]
    )


async def test_picker_token_and_capabilities_are_org_local(owner, client):
    from backend.copilot.capabilities.resolve import load_connection_state

    owner_id, personal, business = owner
    oauth = OAuth2Credentials(
        provider="google",
        access_token=SecretStr("fake-picker"),
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    a = key("A").model_copy(update={"provider": "private_a"})
    b = key("B").model_copy(update={"provider": "private_b"})
    await user.set_user_credentials(owner_id, [a, oauth], personal)
    await user.set_user_credentials(owner_id, [b], business)
    for org, status in [(personal, 200), (business, 404)]:
        response = await client.post(
            f"/google/credentials/{oauth.id}/picker-token", headers={"X-Org-Id": org}
        )
        assert response.status_code == status, response.text
    manager = IntegrationCredentialsManager()
    with credential_scope(personal):
        getter = manager.cached_getter(owner_id)
        assert "private_a" in (await load_connection_state(owner_id)).providers
    with credential_scope(business):
        state = await load_connection_state(owner_id)
        assert "private_a" not in state.providers and "private_b" in state.providers
        assert (await getter(a.id)).id == a.id
        assert await getter(b.id) is None


async def test_graph_matcher_uses_explicit_saved_org_and_personal_override(owner):
    from backend.copilot.tools.utils import match_user_credentials_to_graph
    from backend.data.model import CredentialsFieldInfo

    owner_id, personal, business = owner
    a, b = key("A"), key("B")
    await user.set_user_credentials(owner_id, [a], personal)
    await user.set_user_credentials(owner_id, [b], business)
    graph = SimpleNamespace(
        id="workflow",
        regular_credentials_inputs={
            "github": (
                CredentialsFieldInfo(
                    credentials_provider=frozenset({"github"}),
                    credentials_types=frozenset({"api_key"}),
                ),
                None,
                None,
            )
        },
    )
    with credential_scope(personal):
        matched, missing = await match_user_credentials_to_graph(
            owner_id, graph, organization_id=business
        )
        assert not missing
        assert matched["github"].id == b.id
    with credential_scope(business):
        matched, _ = await match_user_credentials_to_graph(owner_id, graph)
        assert matched["github"].id == b.id
        matched, _ = await match_user_credentials_to_graph(
            owner_id, graph, organization_id=None
        )
        assert matched["github"].id == a.id


async def test_idle_refresh_reloads_and_persists_each_saved_org(owner, monkeypatch):
    import time

    owner_id, personal, business = owner
    manager = IntegrationCredentialsManager()
    old = int(time.time()) - 20 * 86_400
    credentials = [
        OAuth2Credentials(
            provider="github",
            access_token=SecretStr("old-access"),
            refresh_token=SecretStr("old-refresh"),
            access_token_expires_at=old,
            scopes=["repo"],
        )
        for _ in range(2)
    ]
    for org, credential in zip((personal, business), credentials):
        await user.set_user_credentials(owner_id, [credential], org)

    async def refresh(credential):
        return credential.model_copy(
            update={
                "access_token": SecretStr(f"new-{credential.id}"),
                "access_token_expires_at": int(time.time()) + 3_600,
            }
        )

    handler = SimpleNamespace(
        needs_refresh=lambda credential: credential.access_token_expires_at <= old,
        refresh_tokens=AsyncMock(side_effect=refresh),
    )
    monkeypatch.setattr(manager, "_get_oauth_handler", AsyncMock(return_value=handler))
    with credential_scope(business):
        await manager.refresh_idle_credentials()
    assert handler.refresh_tokens.await_count == 2
    for org, credential in zip((personal, business), credentials):
        stored = await user.get_user_credentials(owner_id, org)
        assert [item.id for item in stored] == [credential.id]
        assert stored[0].access_token.get_secret_value() == f"new-{credential.id}"


async def test_briefing_chat_ignores_ambient_business_connection(owner, monkeypatch):
    from backend.copilot import db as chat_db
    from backend.copilot import transports
    from backend.copilot.transports_test import _codex_credentials

    owner_id, personal, business = owner
    credentials = [_codex_credentials(str(uuid.uuid4())) for _ in range(2)]
    monkeypatch.setattr(
        transports, "has_codex_access_for_discovery", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(transports, "is_deployment_chat_available", lambda: True)
    for org, credential in zip((personal, business), credentials):
        await user.set_user_credentials(owner_id, [credential], org)
    await user.set_user_default_chat_route(owner_id, "codex", credentials[0].id)
    with credential_scope(business):
        session_id = await chat_db.append_plain_session_message(
            owner_id, "Private personal briefing", str(uuid.uuid4())
        )
    assert session_id is not None
    session = await chat_db.get_chat_session_metadata(session_id)
    assert session.organization_id is None
    assert session.metadata.llm_credential_id == credentials[0].id
    assert session.metadata.llm_auth_provider == "codex"


async def test_idle_refresh_skips_revoked_organization_membership(owner, monkeypatch):
    import time

    owner_id, _personal, business = owner
    manager = IntegrationCredentialsManager()
    credential = OAuth2Credentials(
        provider="github",
        access_token=SecretStr("old"),
        refresh_token=SecretStr("fake-refresh"),
        access_token_expires_at=int(time.time()) - 20 * 86_400,
        scopes=[],
    )
    await user.set_user_credentials(owner_id, [credential], business)
    await db.prisma.orgmember.update(
        where={"orgId_userId": {"orgId": business, "userId": owner_id}},
        data={"status": "SUSPENDED"},
    )
    refresh = AsyncMock()
    monkeypatch.setattr(manager, "refresh_if_needed", refresh)
    await manager.refresh_idle_credentials()
    refresh.assert_not_awaited()
    assert (await user.get_user_credentials(owner_id, business))[0] == credential
