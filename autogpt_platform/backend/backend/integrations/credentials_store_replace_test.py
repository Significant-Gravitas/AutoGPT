"""A full-list credential replace, seen by a concurrent reader, on the real database.

Credential reads take no lock (#15389), so a reader can run in the middle of a
replace. Committed row by row, the replace showed that reader the new
credential next to the one it supersedes: discovery offered both, and token
selection could pick the one about to be revoked. The reader must see the old
list or the new one, never a mix.
"""

import asyncio
import uuid

import pytest
from prisma.actions import IntegrationCredentialActions
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.data.user import get_or_create_user
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.util.test import SpinTestServer

_STALL = 5.0


async def _user() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"creds-replace-{suffix}@example.com",
            "name": "Credentials Owner",
        }
    )
    return user.id


def _api_key(provider: str, *, managed: bool = False) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=str(uuid.uuid4()),
        provider=provider,
        api_key=SecretStr(f"sk-{uuid.uuid4().hex}"),
        title=provider,
        is_managed=managed,
    )


async def _persisted_ids(
    store: IntegrationCredentialsStore, user_id: str, ours: set[str]
) -> set[str]:
    """The ids *user_id* is offered, out of the ones this test created."""
    return {c.id for c in await store.get_all_creds(user_id)} & ours


async def _wait_for(event: asyncio.Event, what: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=_STALL)
    except TimeoutError:
        pytest.fail(f"timed out after {_STALL}s waiting for {what}")


@pytest.mark.asyncio(loop_scope="session")
async def test_readers_during_a_replace_see_the_old_list_or_the_new_one(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    store = IntegrationCredentialsStore()
    user_id = await _user()
    unrelated = _api_key("github")
    superseded = _api_key("openai", managed=True)
    replacement = _api_key("openai", managed=True)
    await store.add_creds(user_id, unrelated)
    await store.add_managed_credential(user_id, superseded)

    ours = {unrelated.id, superseded.id, replacement.id}
    before = {unrelated.id, superseded.id}
    after = {unrelated.id, replacement.id}
    assert await _persisted_ids(store, user_id, ours) == before

    # Hold the replace between creating the new row and revoking the old one.
    created = asyncio.Event()
    release = asyncio.Event()
    create = IntegrationCredentialActions.create

    async def create_then_hold(self, *args, **kwargs):
        row = await create(self, *args, **kwargs)
        created.set()
        await release.wait()
        return row

    monkeypatch.setattr(IntegrationCredentialActions, "create", create_then_hold)

    writer = asyncio.create_task(
        store.add_managed_credential(user_id, replacement),
    )
    try:
        await _wait_for(created, "the replace to create its new row")
        seen = await asyncio.wait_for(
            asyncio.gather(*(_persisted_ids(store, user_id, ours) for _ in range(5))),
            timeout=_STALL,
        )
    finally:
        release.set()
        await asyncio.wait_for(writer, timeout=_STALL)

    assert all(ids == before for ids in seen), (
        f"a reader saw a mixed list mid-replace: {seen}; "
        f"expected only {before} until the replace commits"
    )
    assert await _persisted_ids(store, user_id, ours) == after


@pytest.mark.asyncio(loop_scope="session")
async def test_a_replace_that_fails_partway_leaves_the_old_list(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    store = IntegrationCredentialsStore()
    user_id = await _user()
    unrelated = _api_key("github")
    superseded = _api_key("openai", managed=True)
    replacement = _api_key("openai", managed=True)
    await store.add_creds(user_id, unrelated)
    await store.add_managed_credential(user_id, superseded)
    ours = {unrelated.id, superseded.id, replacement.id}

    update = IntegrationCredentialActions.update

    async def fail_to_revoke(self, *args, **kwargs):
        if kwargs.get("data") == {"status": "revoked"}:
            raise RuntimeError("revocation failed")
        return await update(self, *args, **kwargs)

    monkeypatch.setattr(IntegrationCredentialActions, "update", fail_to_revoke)

    with pytest.raises(RuntimeError, match="revocation failed"):
        await store.add_managed_credential(user_id, replacement)

    assert await _persisted_ids(store, user_id, ours) == {unrelated.id, superseded.id}
