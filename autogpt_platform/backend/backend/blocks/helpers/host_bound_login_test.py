from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.helpers.host_bound_login import login_may_go_to
from backend.data.credential_scope import ServerTarget
from backend.data.model import UserPasswordCredentials


def _login(scope: str | None) -> UserPasswordCredentials:
    return UserPasswordCredentials(
        provider="smtp",
        username=SecretStr("user"),
        password=SecretStr("pass"),
        scope=scope,
    )


def _db(owns_graph: bool) -> MagicMock:
    return MagicMock(is_graph_version_owner=AsyncMock(return_value=owns_graph))


_DB_CLIENT = "backend.blocks.helpers.host_bound_login.get_database_manager_async_client"
_RUN = {"user_id": "user-1", "graph_id": "graph-1", "graph_version": 3}
_OTHER_SERVER = ServerTarget(scheme="smtp", host="203.0.113.7", port=587)


@pytest.mark.asyncio
@pytest.mark.parametrize("owns_graph", [True, False])
async def test_login_without_scope_follows_graph_ownership(owns_graph: bool):
    db = _db(owns_graph)
    with patch(_DB_CLIENT, return_value=db):
        allowed = await login_may_go_to(_login(None), _OTHER_SERVER, **_RUN)
    assert allowed is owns_graph
    db.is_graph_version_owner.assert_awaited_once_with("graph-1", 3, "user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "run",
    [
        {"user_id": None, "graph_id": "graph-1", "graph_version": 3},
        {"user_id": "user-1", "graph_id": None, "graph_version": 3},
        {"user_id": "user-1", "graph_id": "graph-1", "graph_version": None},
    ],
)
async def test_login_without_scope_refused_outside_a_graph(run: dict):
    db = _db(True)
    with patch(_DB_CLIENT, return_value=db):
        assert not await login_may_go_to(_login(None), _OTHER_SERVER, **run)
    db.is_graph_version_owner.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["smtp://smtp.example.com", "", "smtp.example.com"])
async def test_login_with_scope_never_falls_back_to_ownership(scope: str):
    """Only a login with no scope is a legacy one; any scope, even an empty or
    malformed one, is checked as it is."""
    db = _db(True)
    with patch(_DB_CLIENT, return_value=db):
        assert not await login_may_go_to(_login(scope), _OTHER_SERVER, **_RUN)
    db.is_graph_version_owner.assert_not_awaited()


@pytest.mark.asyncio
async def test_login_with_scope_sent_to_its_server():
    db = _db(False)
    with patch(_DB_CLIENT, return_value=db):
        assert await login_may_go_to(
            _login("smtp://smtp.example.com"),
            ServerTarget(scheme="smtp", host="smtp.example.com", port=587),
            **_RUN,
        )
