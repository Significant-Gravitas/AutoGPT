"""Tests for the connected-or-RPC database accessor helpers."""

from unittest.mock import MagicMock, patch

import pytest

from backend.api.features.orgs import db as orgs_module
from backend.data import bot_installs as bot_installs_module
from backend.data import db_accessors, onboarding_audience, onboarding_role
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient


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


@pytest.mark.parametrize(
    "accessor, module, endpoint",
    [
        (db_accessors.onboarding_role_db, onboarding_role, "save_onboarding_role"),
        (
            db_accessors.onboarding_audience_db,
            onboarding_audience,
            "queue_onboarding_role",
        ),
    ],
)
def test_onboarding_accessors_preserve_connected_and_rpc_paths(
    accessor, module, endpoint
):
    assert endpoint in vars(DatabaseManager)
    assert endpoint in vars(DatabaseManagerAsyncClient)
    with patch("backend.data.db_accessors.db.is_connected", return_value=True):
        assert accessor() is module
    client = MagicMock(spec=DatabaseManagerAsyncClient)
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ),
    ):
        assert accessor() is client
