"""Tests for the connected-or-RPC database accessor helpers."""

from unittest.mock import MagicMock, patch

from backend.api.features.orgs import db as orgs_module
from backend.data import bot_installs as bot_installs_module
from backend.data import db_accessors
from backend.data import dream_pass as dream_pass_module


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


def test_dream_db_uses_direct_module_when_connected():
    with patch("backend.data.db_accessors.db.is_connected", return_value=True):
        assert db_accessors.dream_db() is dream_pass_module


def test_dream_db_falls_back_to_a_non_retrying_database_manager_client():
    # The scheduler (sync passes) and the batch executor (batch callbacks)
    # keep no Prisma connection, so the dream store's writes cross the RPC.
    # The store bounds each write itself; client retries would only keep
    # going on a request the store has already given up on.
    client = MagicMock()
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ) as get_client,
    ):
        assert db_accessors.dream_db() is client
    get_client.assert_called_once_with(should_retry=False)


def test_db_for_returns_the_module_itself_when_connected():
    with patch("backend.data.db_accessors.db.is_connected", return_value=True):
        assert db_accessors.db_for(dream_pass_module) is dream_pass_module


def test_db_for_asks_for_the_shared_retrying_client_with_no_arguments():
    # The client is cached per argument list: a default caller that passed
    # should_retry=True would get a second client and connection pool.
    client = MagicMock()
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ) as get_client,
    ):
        assert db_accessors.db_for(orgs_module) is client
    get_client.assert_called_once_with()


def test_db_for_can_ask_for_a_client_that_does_not_retry():
    client = MagicMock()
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ) as get_client,
    ):
        assert db_accessors.db_for(orgs_module, should_retry=False) is client
    get_client.assert_called_once_with(should_retry=False)
