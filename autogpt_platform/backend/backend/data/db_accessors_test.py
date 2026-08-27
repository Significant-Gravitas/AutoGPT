"""Tests for the connected-or-RPC database accessor helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.orgs import db as orgs_module
from backend.data import bot_installs as bot_installs_module
from backend.data import db_accessors


def test_orgs_db_uses_direct_module_when_connected():
    with patch("backend.data.db_accessors.db.is_connected", return_value=True):
        assert db_accessors.orgs_db() is orgs_module


def test_orgs_db_falls_back_to_database_manager_client():
    # Services without their own Prisma connection (PlatformLinkingManager)
    # must route through the DatabaseManager's centralized pool.
    client = MagicMock()
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ),
    ):
        assert db_accessors.orgs_db() is client


def test_bot_installs_db_uses_direct_module_when_connected():
    with patch("backend.data.db_accessors.db.is_connected", return_value=True):
        assert db_accessors.bot_installs_db() is bot_installs_module


def test_bot_installs_db_falls_back_to_database_manager_client():
    # The copilot-bot bridge pod has no Prisma connection — Slack's
    # per-workspace token lookups must route through the DatabaseManager.
    client = MagicMock()
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ),
    ):
        assert db_accessors.bot_installs_db() is client


@pytest.mark.asyncio
async def test_exact_chat_session_scope_accepts_matching_session() -> None:
    session = MagicMock(user_id="user-1", organization_id="org-1", team_id="team-1")
    client = MagicMock()
    client.get_chat_session_metadata = AsyncMock(return_value=session)
    with patch("backend.data.db_accessors.chat_db", return_value=client):
        await db_accessors.require_exact_chat_session_scope(
            "session-1", "user-1", "org-1", "team-1"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "organization_id", "team_id"),
    [
        ("attacker", "org-1", "team-1"),
        ("user-1", "other-org", "team-1"),
        ("user-1", "org-1", "other-team"),
        ("user-1", None, None),
    ],
)
async def test_exact_chat_session_scope_rejects_mismatch(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
) -> None:
    session = MagicMock(user_id="user-1", organization_id="org-1", team_id="team-1")
    client = MagicMock()
    client.get_chat_session_metadata = AsyncMock(return_value=session)
    with (
        patch("backend.data.db_accessors.chat_db", return_value=client),
        pytest.raises(db_accessors.LiveResourceAccessRevoked),
    ):
        await db_accessors.require_exact_chat_session_scope(
            "session-1", user_id, organization_id, team_id
        )


@pytest.mark.asyncio
async def test_live_resource_lease_reuses_exact_active_guard() -> None:
    client = MagicMock()
    client.acquire_live_resource_lease = AsyncMock(return_value="lease-1")
    client.release_live_resource_lease = AsyncMock(return_value=True)
    client.is_live_resource_lease_active = AsyncMock(return_value=True)

    with patch("backend.data.db_accessors.credit_db", return_value=client):
        async with db_accessors.live_resource_lease(
            "user-1", "org-1", "team-1", "execute"
        ) as outer:
            async with db_accessors.live_resource_lease(
                "user-1", "org-1", "team-1", "execute"
            ) as nested:
                assert nested is outer

    client.acquire_live_resource_lease.assert_awaited_once_with(
        "user-1", "org-1", "team-1", "execute"
    )
    client.release_live_resource_lease.assert_awaited_once_with("lease-1")


@pytest.mark.asyncio
async def test_transferred_guard_is_reused_without_acquiring_another_lease() -> None:
    client = MagicMock()
    client.is_live_resource_lease_active = AsyncMock(return_value=True)
    guard = db_accessors.LiveResourceLeaseGuard(client, "lease-1")

    async def nested_action() -> None:
        async with db_accessors.live_resource_lease(
            "user-1", "org-1", "team-1", "execute"
        ) as nested:
            assert nested is guard

    with patch("backend.data.db_accessors.credit_db") as credit_db:
        await db_accessors.run_with_live_resource_lease_guard(
            guard,
            user_id="user-1",
            organization_id="org-1",
            team_id="team-1",
            access="execute",
            action=nested_action(),
        )

    credit_db.assert_not_called()
