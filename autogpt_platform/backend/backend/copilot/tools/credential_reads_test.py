"""An expert session reads the account's credentials and the expert's grants
side by side, with the behaviour of reading them one after the other: the same
filter, a failure raised in the same order, and the grants seeded (a write)
only once the credentials have been read."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.copilot.tools.expert_scope import resolve_credential_scope
from backend.copilot.tools.utils import (
    check_user_has_required_credentials,
    get_user_credentials,
    match_user_credentials_to_graph,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsFieldInfo,
    CredentialsMetaInput,
)
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.integrations.providers import ProviderName
from backend.util.testing import Rendezvous


def _key(credential_id: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=credential_id, provider="github", title=credential_id, api_key=SecretStr("k")
    )


_ACCOUNT = [_key("ungranted-cred"), _key("granted-cred")]


@contextmanager
def _account(get_all_creds: AsyncMock, experts: MagicMock) -> Iterator[None]:
    """The account's credentials and the experts store, through every module
    that reads them, so a read in the wrong place fails instead of hanging."""
    with (
        patch.object(IntegrationCredentialsStore, "get_all_creds", get_all_creds),
        patch("backend.data.db_accessors.experts_db", return_value=experts),
        patch("backend.copilot.tools.expert_scope.experts_db", return_value=experts),
    ):
        yield


def _experts(seeded: AsyncMock, allowed: list[str] | None = None) -> MagicMock:
    experts = MagicMock()
    experts.seeded_expert_credential_ids = seeded
    experts.expert_allowed_credential_ids = AsyncMock(return_value=allowed or [])
    return experts


async def test_an_experts_grants_are_read_beside_the_credentials():
    meet = Rendezvous(2)
    experts = _experts(AsyncMock(side_effect=meet.returning(["granted-cred"])))
    with _account(AsyncMock(side_effect=meet.returning(_ACCOUNT)), experts):
        credentials = await get_user_credentials("user-1", "expert-a")

    assert [c.id for c in credentials] == ["granted-cred"]
    experts.expert_allowed_credential_ids.assert_not_awaited()


async def test_the_setup_cards_snapshot_reads_them_side_by_side_too():
    meet = Rendezvous(2)
    experts = _experts(AsyncMock(side_effect=meet.returning(["granted-cred"])))
    with _account(AsyncMock(side_effect=meet.returning(_ACCOUNT)), experts):
        scope = await resolve_credential_scope("user-1", "expert-a")

    assert scope.allowed_ids == {"granted-cred"}
    assert [c.id for c in scope.available] == ["granted-cred"]


async def test_an_unseeded_expert_is_seeded_only_once_the_credentials_are_read():
    order: list[str] = []

    async def read_credentials(_user_id):
        await asyncio.sleep(0)
        order.append("credentials read")
        return _ACCOUNT

    experts = _experts(AsyncMock(return_value=None), allowed=["granted-cred"])
    experts.expert_allowed_credential_ids.side_effect = lambda *_: (
        order.append("seed") or ["granted-cred"]
    )
    with _account(AsyncMock(side_effect=read_credentials), experts):
        credentials = await get_user_credentials("user-1", "expert-a")

    assert order == ["credentials read", "seed"]
    assert [c.id for c in credentials] == ["granted-cred"]
    experts.expert_allowed_credential_ids.assert_awaited_once_with("user-1", "expert-a")


async def test_a_failed_credentials_read_seeds_nothing():
    experts = _experts(AsyncMock(return_value=None))
    with (
        _account(AsyncMock(side_effect=RuntimeError("store down")), experts),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await get_user_credentials("user-1", "expert-a")

    experts.expert_allowed_credential_ids.assert_not_awaited()


async def test_a_failed_credentials_read_is_raised_before_a_failed_grants_read():
    """The grants read fails first, but read one after the other the
    credentials came first, so theirs is the error the caller sees."""

    async def credentials_fail_later(_user_id):
        for _ in range(3):
            await asyncio.sleep(0)
        raise RuntimeError("store down")

    experts = _experts(AsyncMock(side_effect=LookupError("expert gone")))
    with (
        _account(AsyncMock(side_effect=credentials_fail_later), experts),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await get_user_credentials("user-1", "expert-a")


async def test_a_plain_session_reads_no_grants():
    experts = _experts(AsyncMock(return_value=["granted-cred"]))
    with _account(AsyncMock(return_value=_ACCOUNT), experts):
        credentials = await get_user_credentials("user-1")

    assert [c.id for c in credentials] == ["ungranted-cred", "granted-cred"]
    experts.seeded_expert_credential_ids.assert_not_called()
    experts.expert_allowed_credential_ids.assert_not_called()


@pytest.mark.parametrize("granted", [["granted-cred"], []], ids=["granted", "not"])
async def test_a_workflow_in_an_expert_session_matches_only_a_granted_credential(
    granted,
):
    field = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": ["api_key"],
            "is_auto_credential": False,
        },
        by_alias=True,
    )
    graph = MagicMock(id="test-graph")
    graph.regular_credentials_inputs = {
        "github_api_key": (field, {("node-1", "credentials")}, True)
    }
    experts = _experts(AsyncMock(return_value=granted), allowed=granted)
    with _account(AsyncMock(return_value=_ACCOUNT), experts):
        matched, missing = await match_user_credentials_to_graph(
            "user-1", graph, expert_id="expert-a"
        )

    if granted:
        assert matched["github_api_key"].id == "granted-cred" and missing == []
    else:
        assert matched == {} and len(missing) == 1


@pytest.mark.parametrize("granted", [["granted-cred"], []], ids=["granted", "not"])
async def test_a_required_credential_counts_only_once_the_expert_is_granted_it(
    granted,
):
    required = [
        CredentialsMetaInput(
            id="github", provider=ProviderName.GITHUB, type="api_key", title="GitHub"
        )
    ]
    experts = _experts(AsyncMock(return_value=granted), allowed=granted)
    with _account(AsyncMock(return_value=[_key("granted-cred")]), experts):
        missing = await check_user_has_required_credentials(
            "user-1", required, expert_id="expert-a"
        )

    assert len(missing) == (0 if granted else 1)
