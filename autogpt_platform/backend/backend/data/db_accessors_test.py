"""Tests for the connected-or-RPC database accessor helpers."""

from unittest.mock import MagicMock, patch

from backend.api.features.orgs import db as orgs_module
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
