"""Tests for per-expert credential grants and the enforcement they drive."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from pydantic import SecretStr

from backend.api.features.experts.credentials import (
    _derive_from_workflows,
    _to_refs,
    expert_allowed_credential_ids,
    filter_credentials_for_expert,
    settle_credential_seed,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsMetaInput,
    OAuth2Credentials,
)
from backend.executor.utils import _enforce_expert_credential_scope
from backend.util.exceptions import ExpertNotFoundError


def _api_key_credential(credential_id: str, provider: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=credential_id,
        provider=provider,
        api_key=SecretStr("key"),
        title=f"{provider} key",
    )


class _Grant:
    def __init__(self, credential_id: str, provider: str) -> None:
        self.credentialId = credential_id
        self.provider = provider


def test_refs_drop_grants_whose_credential_was_deleted():
    """A grant pointing at a deleted credential is inert, not renderable.

    Rendering it would claim the expert has access it does not have — the id
    resolves to nothing at enforcement time.
    """
    linkedin = _api_key_credential("cred-linkedin", "linkedin")
    grants = [_Grant("cred-linkedin", "linkedin"), _Grant("cred-gone", "notion")]

    refs = _to_refs(grants, [linkedin])  # type: ignore[arg-type]

    assert [ref.credential_id for ref in refs] == ["cred-linkedin"]
    assert refs[0].provider == "linkedin"


def test_refs_read_title_from_the_live_credential():
    """Titles come from the credential, not the grant, so renames propagate."""
    credential = _api_key_credential("cred-1", "notion")
    credential.title = "Work Notion"

    refs = _to_refs([_Grant("cred-1", "notion")], [credential])  # type: ignore[arg-type]

    assert refs[0].title == "Work Notion"


def test_filter_drops_ungranted_credentials():
    granted = _api_key_credential("cred-granted", "linkedin")
    ungranted = _api_key_credential("cred-ungranted", "notion")

    kept = filter_credentials_for_expert([granted, ungranted], {"cred-granted"})

    assert [c.id for c in kept] == ["cred-granted"]


def test_filter_always_keeps_system_credentials(mocker: pytest_mock.MockFixture):
    """Platform LLM keys carry no grant; filtering them would stop every expert
    from running a single LLM block."""
    mocker.patch(
        "backend.api.features.experts.credentials.is_system_credential",
        side_effect=lambda credential_id: credential_id == "cred-system",
    )
    system = _api_key_credential("cred-system", "openai")
    ungranted = _api_key_credential("cred-ungranted", "notion")

    kept = filter_credentials_for_expert([system, ungranted], set())

    assert [c.id for c in kept] == ["cred-system"]


def test_filter_with_no_grants_keeps_nothing_user_owned():
    """Deny-by-default: an expert with an empty allow-list reaches nothing."""
    credential = _api_key_credential("cred-1", "linkedin")

    assert filter_credentials_for_expert([credential], set()) == []


@pytest.mark.asyncio
async def test_derivation_reports_incomplete_when_a_workflow_fails(
    mocker: pytest_mock.MockFixture,
):
    """A transient graph-load failure must leave the seed pending.

    Stamping here would freeze an under-seeded allow-list in place, and
    enforcement reads that as "reaches nothing" — blocking every run the
    expert has until someone re-granted by hand.
    """
    mocker.patch("backend.data.graph.get_graph", side_effect=RuntimeError("db is down"))
    expert = SimpleNamespace(
        id="expert-1",
        Workflows=[
            SimpleNamespace(
                id="wf-1",
                LibraryAgent=SimpleNamespace(agentGraphId="g1", agentGraphVersion=1),
            )
        ],
    )

    derived, is_complete = await _derive_from_workflows("user-1", expert)  # type: ignore[arg-type]

    assert derived == {}
    assert is_complete is False


@pytest.mark.asyncio
async def test_a_workflow_whose_graph_is_gone_still_counts_as_complete(
    mocker: pytest_mock.MockFixture,
):
    """A missing graph resolves to nothing and always will — retrying it every
    read would cost a graph load per header render forever."""
    mocker.patch("backend.data.graph.get_graph", return_value=None)
    expert = SimpleNamespace(
        id="expert-1",
        Workflows=[
            SimpleNamespace(
                id="wf-1",
                LibraryAgent=SimpleNamespace(agentGraphId="g1", agentGraphVersion=1),
            )
        ],
    )

    derived, is_complete = await _derive_from_workflows("user-1", expert)  # type: ignore[arg-type]

    assert derived == {}
    assert is_complete is True


@pytest.mark.asyncio
async def test_an_expert_with_no_workflows_seeds_completely():
    expert = SimpleNamespace(id="expert-1", Workflows=[])

    derived, is_complete = await _derive_from_workflows("user-1", expert)  # type: ignore[arg-type]

    assert derived == {}
    assert is_complete is True


@pytest.mark.asyncio
async def test_execution_is_rejected_when_a_credential_was_not_granted(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.executor.utils.get_experts_db",
        return_value=AsyncMock(
            expert_allowed_credential_ids=AsyncMock(return_value=["cred-granted"])
        ),
    )

    with pytest.raises(ValueError, match="cred-ungranted"):
        await _enforce_expert_credential_scope(
            "user-1",
            "expert-1",
            {
                "field": CredentialsMetaInput(
                    id="cred-ungranted", provider="notion", type="api_key"
                )
            },
        )


@pytest.mark.asyncio
async def test_execution_is_allowed_when_every_credential_was_granted(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.executor.utils.get_experts_db",
        return_value=AsyncMock(
            expert_allowed_credential_ids=AsyncMock(return_value=["cred-granted"])
        ),
    )

    await _enforce_expert_credential_scope(
        "user-1",
        "expert-1",
        {
            "field": CredentialsMetaInput(
                id="cred-granted", provider="linkedin", type="api_key"
            )
        },
    )


@pytest.mark.asyncio
async def test_a_run_with_no_credentials_skips_the_lookup_entirely(
    mocker: pytest_mock.MockFixture,
):
    accessor = mocker.patch("backend.executor.utils.get_experts_db")

    await _enforce_expert_credential_scope("user-1", "expert-1", None)

    accessor.assert_not_called()


@pytest.mark.asyncio
async def test_derivation_includes_picker_credentials_the_user_owns(
    mocker: pytest_mock.MockFixture,
):
    """A ``_credentials_id`` embedded in a picker field is enforced at run time,
    so the seed must offer it or every Drive-backed expert workflow fails."""
    node = SimpleNamespace(
        block=SimpleNamespace(
            input_schema=SimpleNamespace(
                get_auto_credentials_fields=lambda: {
                    "credentials": {
                        "field_name": "spreadsheet",
                        "config": {"provider": "google", "type": "oauth2"},
                    }
                }
            )
        ),
        input_default={
            "spreadsheet": {"_credentials_id": "cred-drive", "name": "sheet"},
            "other": {"_credentials_id": "cred-foreign", "name": "not mine"},
        },
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        return_value=SimpleNamespace(
            nodes=[], sub_graphs=[SimpleNamespace(nodes=[node])]
        ),
    )
    mocker.patch(
        "backend.copilot.tools.utils.match_user_credentials_to_graph",
        AsyncMock(return_value=({}, {})),
    )
    mocker.patch(
        "backend.api.features.experts.credentials._user_credentials",
        AsyncMock(return_value=[_api_key_credential("cred-drive", "google")]),
    )
    expert = SimpleNamespace(
        id="expert-1",
        Workflows=[
            SimpleNamespace(
                id="wf-1",
                LibraryAgent=SimpleNamespace(agentGraphId="g1", agentGraphVersion=1),
            )
        ],
    )

    derived, is_complete = await _derive_from_workflows("user-1", expert)  # type: ignore[arg-type]

    assert derived == {"cred-drive": "google"}
    assert is_complete is True


@pytest.mark.asyncio
async def test_a_failed_credential_read_leaves_the_seed_pending(
    mocker: pytest_mock.MockFixture,
):
    node = SimpleNamespace(
        block=SimpleNamespace(
            input_schema=SimpleNamespace(
                get_auto_credentials_fields=lambda: {
                    "credentials": {
                        "field_name": "sheet",
                        "config": {"provider": "google"},
                    }
                }
            )
        ),
        input_default={"sheet": {"_credentials_id": "cred-drive"}},
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        return_value=SimpleNamespace(nodes=[node], sub_graphs=[]),
    )
    mocker.patch(
        "backend.copilot.tools.utils.match_user_credentials_to_graph",
        AsyncMock(return_value=({}, {})),
    )
    mocker.patch(
        "backend.api.features.experts.credentials._user_credentials",
        AsyncMock(side_effect=RuntimeError("redis is down")),
    )
    expert = SimpleNamespace(
        id="expert-1",
        Workflows=[
            SimpleNamespace(
                id="wf-1",
                LibraryAgent=SimpleNamespace(agentGraphId="g1", agentGraphVersion=1),
            )
        ],
    )

    derived, is_complete = await _derive_from_workflows("user-1", expert)  # type: ignore[arg-type]

    assert derived == {}
    assert is_complete is False


@pytest.mark.asyncio
async def test_settling_the_seed_stamps_even_when_derivation_was_incomplete(
    mocker: pytest_mock.MockFixture,
):
    expert = SimpleNamespace(id="expert-1", Workflows=[], credentialsSeededAt=None)
    mocker.patch(
        "backend.api.features.experts.credentials._owned_expert",
        AsyncMock(return_value=expert),
    )
    seed = mocker.patch(
        "backend.api.features.experts.credentials._seed_if_needed", AsyncMock()
    )
    stamp = mocker.patch(
        "backend.api.features.experts.credentials._stamp_seeded", AsyncMock()
    )

    await settle_credential_seed("user-1", "expert-1")

    seed.assert_awaited_once_with("user-1", expert)
    stamp.assert_awaited_once_with("expert-1")


_CREDENTIALS = "backend.api.features.experts.credentials"


@pytest.mark.asyncio
async def test_the_grants_are_read_beside_the_ownership_check(
    mocker: pytest_mock.MockFixture,
):
    """Each read returns only once both are in flight, so reading the grants
    after the ownership check times out."""
    started: list[str] = []
    both = asyncio.Event()

    def in_flight(name: str, value):
        async def read(*_args, **_kwargs):
            started.append(name)
            if len(started) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), timeout=1)
            return value

        return read

    seeded = SimpleNamespace(
        id="expert-1", credentialsSeededAt=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    mocker.patch(
        f"{_CREDENTIALS}._owned_expert",
        AsyncMock(side_effect=in_flight("expert", seeded)),
    )
    grants = mocker.patch(
        f"{_CREDENTIALS}._grants",
        AsyncMock(side_effect=in_flight("grants", [_Grant("cred-a", "github")])),
    )
    seed = mocker.patch(f"{_CREDENTIALS}._seed_if_needed", AsyncMock())

    assert await expert_allowed_credential_ids("user-1", "expert-1") == ["cred-a"]
    assert sorted(started) == ["expert", "grants"]
    seed.assert_not_awaited()
    assert grants.await_count == 1


@pytest.mark.asyncio
async def test_a_seed_stamped_while_the_grants_were_read_has_them_read_again(
    mocker: pytest_mock.MockFixture,
):
    """Another call seeded the expert during these reads: its grants landed
    after this grants read started, its stamp before the expert read ended."""
    just_seeded = SimpleNamespace(
        id="expert-1", credentialsSeededAt=datetime.now(timezone.utc)
    )
    mocker.patch(f"{_CREDENTIALS}._owned_expert", AsyncMock(return_value=just_seeded))
    grants = mocker.patch(
        f"{_CREDENTIALS}._grants",
        AsyncMock(side_effect=[[], [_Grant("cred-seeded", "github")]]),
    )
    seed = mocker.patch(f"{_CREDENTIALS}._seed_if_needed", AsyncMock())

    assert await expert_allowed_credential_ids("user-1", "expert-1") == ["cred-seeded"]
    seed.assert_not_awaited()
    assert grants.await_count == 2


@pytest.mark.asyncio
async def test_an_expert_seeded_by_the_read_gets_the_grants_the_seed_wrote(
    mocker: pytest_mock.MockFixture,
):
    unseeded = SimpleNamespace(id="expert-1", credentialsSeededAt=None)
    mocker.patch(f"{_CREDENTIALS}._owned_expert", AsyncMock(return_value=unseeded))
    grants = mocker.patch(
        f"{_CREDENTIALS}._grants",
        AsyncMock(side_effect=[[], [_Grant("cred-seeded", "github")]]),
    )
    seed = mocker.patch(f"{_CREDENTIALS}._seed_if_needed", AsyncMock())

    ids = await expert_allowed_credential_ids("user-1", "expert-1")

    assert ids == ["cred-seeded"]
    seed.assert_awaited_once_with("user-1", unseeded)
    assert grants.await_count == 2


@pytest.mark.asyncio
async def test_an_expert_the_user_does_not_own_grants_nothing(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        f"{_CREDENTIALS}._owned_expert",
        AsyncMock(side_effect=ExpertNotFoundError("Expert #expert-1 not found")),
    )
    mocker.patch(
        f"{_CREDENTIALS}._grants", AsyncMock(return_value=[_Grant("cred-a", "x")])
    )

    with pytest.raises(ExpertNotFoundError):
        await expert_allowed_credential_ids("user-1", "expert-1")


def test_refs_carry_the_service_behind_an_mcp_credential():
    from backend.integrations.mcp_catalog import get_mcp_catalog

    linear = next(e for e in get_mcp_catalog() if e.name == "mcp_linear")
    credential = OAuth2Credentials(
        id="cred-mcp",
        provider="mcp",
        title="MCP: mcp.linear.app",
        access_token=SecretStr("t"),
        scopes=[],
        metadata={"mcp_server_url": linear.mcp_server.server_url},
    )

    refs = _to_refs([_Grant("cred-mcp", "mcp")], [credential])  # type: ignore[arg-type]

    assert refs[0].service == "linear"
    assert refs[0].service_name == "Linear"
    assert refs[0].service_icon == linear.mcp_server.icon_id


def test_refs_carry_the_provider_as_service_for_a_block_credential():
    refs = _to_refs(  # type: ignore[arg-type]
        [_Grant("cred-1", "notion")], [_api_key_credential("cred-1", "notion")]
    )
    assert refs[0].service == "notion"
    assert refs[0].service_icon == "notion"
