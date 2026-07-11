"""Tests for IntegrationCredentialsStore's read resolution.

Locks in the wiring that makes credential LISTING and execution FETCH
resolve the user's accessible set (USER + TEAM + ORG) via
``get_accessible_credentials``. Because a credential id only appears in
``get_all_creds`` when the user has access, the id-based lookup used by the
executor (``creds_manager.get`` → ``get_creds_by_id``) inherits that access
check — which these tests exercise for both the allowed and denied paths.
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import IntegrationCredentialsStore


class _NullAsyncCM:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


async def _fake_locked(self, user_id):
    return _NullAsyncCM()


def _api_key(cred_id: str, provider: str = "github") -> APIKeyCredentials:
    return APIKeyCredentials(
        id=cred_id, provider=provider, api_key=SecretStr("sk"), title="t"
    )


@pytest.fixture
def store(mocker):
    s = IntegrationCredentialsStore()
    # Bypass the Redis-backed per-user lock; these tests exercise resolution.
    mocker.patch.object(
        IntegrationCredentialsStore, "locked_user_integrations", _fake_locked
    )
    fake_db = MagicMock()
    mocker.patch.object(
        IntegrationCredentialsStore,
        "db_manager",
        new_callable=PropertyMock,
        return_value=fake_db,
    )
    return s, fake_db


@pytest.mark.asyncio
async def test_get_all_creds_resolves_accessible_set(store):
    s, fake_db = store
    fake_db.get_accessible_credentials = AsyncMock(
        return_value=[_api_key("u"), _api_key("t", provider="notion")]
    )

    result = await s.get_all_creds("user-1")

    ids = {c.id for c in result}
    assert {"u", "t"} <= ids  # plus system creds (e.g. ollama)
    fake_db.get_accessible_credentials.assert_awaited_once_with(user_id="user-1")


@pytest.mark.asyncio
async def test_get_creds_by_id_allows_accessible_team_cred(store):
    """Executor fetch-by-id succeeds for a team credential the user can use."""
    s, fake_db = store
    fake_db.get_accessible_credentials = AsyncMock(
        return_value=[_api_key("t", provider="notion")]
    )

    got = await s.get_creds_by_id("user-1", "t")

    assert got is not None
    assert got.id == "t"


@pytest.mark.asyncio
async def test_get_creds_by_id_denies_inaccessible_cred(store):
    """Executor fetch-by-id returns None for a credential outside the user's
    accessible set — a non-member (or departed member) cannot fetch it."""
    s, fake_db = store
    fake_db.get_accessible_credentials = AsyncMock(return_value=[_api_key("u")])

    got = await s.get_creds_by_id("user-1", "other-team-cred")

    assert got is None
