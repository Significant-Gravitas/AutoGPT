"""Tests for the Team page's setup card: what each pending workflow is missing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from pydantic import SecretStr

from backend.api.features.experts import setup
from backend.data.model import APIKeyCredentials, CredentialsFieldInfo


def _api_key_credential(credential_id: str, provider: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=credential_id,
        provider=provider,
        api_key=SecretStr("key"),
        title=f"{provider} key",
    )


def _field(provider: str, required: bool = True):
    info = CredentialsFieldInfo(
        credentials_provider=frozenset({provider}),
        credentials_types=frozenset({"api_key"}),
    )
    return (info, set(), required)


def _workflow(**overrides) -> SimpleNamespace:
    values = dict(
        id="wf-1",
        scheduleCron="0 9 * * 1",
        scheduleId=None,
        libraryAgentId="lib-1",
        LibraryAgent=SimpleNamespace(
            agentGraphId="g1",
            agentGraphVersion=1,
            name=None,
            AgentGraph=SimpleNamespace(name="SEO Audit"),
        ),
        StoreListingVersion=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _expert(*workflows: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        id="expert-1", name="Maria", avatarUrl=None, Workflows=list(workflows)
    )


def _arrange(
    mocker: pytest_mock.MockFixture,
    *,
    experts: list[SimpleNamespace],
    credentials: list[APIKeyCredentials],
    allowed: list[str],
    fields: dict,
) -> None:
    expert_client = mocker.MagicMock()
    expert_client.find_many = AsyncMock(return_value=experts)
    mocker.patch.object(
        setup.prisma.models.Expert, "prisma", return_value=expert_client
    )
    mocker.patch.object(
        setup, "_user_credentials", new=AsyncMock(return_value=credentials)
    )
    mocker.patch.object(
        setup, "expert_allowed_credential_ids", new=AsyncMock(return_value=allowed)
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        new=AsyncMock(return_value=SimpleNamespace(regular_credentials_inputs=fields)),
    )


@pytest.mark.asyncio
async def test_lists_nothing_when_every_workflow_has_its_schedule(
    mocker: pytest_mock.MockFixture,
):
    _arrange(
        mocker,
        experts=[_expert(_workflow(scheduleId="sched-1"))],
        credentials=[],
        allowed=[],
        fields={"notion": _field("notion")},
    )

    assert await setup.list_setup_items("user-1") == []


@pytest.mark.asyncio
async def test_connect_when_the_user_has_no_matching_credential(
    mocker: pytest_mock.MockFixture,
):
    _arrange(
        mocker,
        experts=[_expert(_workflow())],
        credentials=[],
        allowed=[],
        fields={"notion": _field("notion")},
    )

    [item] = await setup.list_setup_items("user-1")

    assert item.resolution == "connect"
    assert item.providers == ["notion"]
    assert item.credential_id is None
    assert item.workflow_name == "SEO Audit"
    assert item.expert_name == "Maria"


@pytest.mark.asyncio
async def test_allow_when_a_credential_exists_but_the_expert_may_not_use_it(
    mocker: pytest_mock.MockFixture,
):
    """The fix is a grant, not a new connection, so the row carries the id."""
    _arrange(
        mocker,
        experts=[_expert(_workflow())],
        credentials=[_api_key_credential("cred-notion", "notion")],
        allowed=[],
        fields={"notion": _field("notion")},
    )

    [item] = await setup.list_setup_items("user-1")

    assert item.resolution == "allow"
    assert item.credential_id == "cred-notion"


@pytest.mark.asyncio
async def test_a_reachable_credential_leaves_only_the_missing_schedule(
    mocker: pytest_mock.MockFixture,
):
    """Nothing credential-shaped is missing, so the user gets the workflow."""
    _arrange(
        mocker,
        experts=[_expert(_workflow())],
        credentials=[_api_key_credential("cred-notion", "notion")],
        allowed=["cred-notion"],
        fields={"notion": _field("notion")},
    )

    [item] = await setup.list_setup_items("user-1")

    assert item.resolution == "workflow"
    assert item.providers == []
    assert item.library_agent_id == "lib-1"


@pytest.mark.asyncio
async def test_optional_credentials_never_block(mocker: pytest_mock.MockFixture):
    """The scheduler skips a node whose optional credential is unset."""
    _arrange(
        mocker,
        experts=[_expert(_workflow())],
        credentials=[],
        allowed=[],
        fields={"notion": _field("notion", required=False)},
    )

    [item] = await setup.list_setup_items("user-1")

    assert item.resolution == "workflow"


@pytest.mark.asyncio
async def test_a_graph_that_fails_to_load_still_reports_the_workflow(
    mocker: pytest_mock.MockFixture,
):
    _arrange(
        mocker,
        experts=[_expert(_workflow())],
        credentials=[],
        allowed=[],
        fields={},
    )
    mocker.patch(
        "backend.data.graph.get_graph",
        new=AsyncMock(side_effect=RuntimeError("db is down")),
    )

    [item] = await setup.list_setup_items("user-1")

    assert item.resolution == "workflow"
