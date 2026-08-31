"""Tests for per-expert credential grants and the enforcement they drive."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from pydantic import SecretStr

from backend.api.features.experts.credentials import (
    _derive_from_workflows,
    _to_refs,
    filter_credentials_for_expert,
)
from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.executor.utils import _enforce_expert_credential_scope


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
