"""Flag-off memory remains private without exposing shared-tier capabilities."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.prompting import get_graphiti_supplement
from backend.copilot.sdk.tool_adapter import _build_input_schema
from backend.copilot.tools import get_available_tools
from backend.copilot.tools.graphiti_forget import (
    MemoryForgetConfirmTool,
    MemoryForgetSearchTool,
)
from backend.copilot.tools.graphiti_store import MemoryStoreTool
from backend.copilot.tools.models import ErrorResponse, MemoryStoreResponse
from backend.util.feature_flag import Flag

from . import ingest, rollout, tiers
from .client import derive_memory_group_id


@pytest.fixture(autouse=True)
def _rollout_off(monkeypatch: pytest.MonkeyPatch, _enable_shared_memory_rollout):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")


@pytest.mark.asyncio
async def test_unavailable_flag_defaults_to_disabled():
    with patch.object(
        rollout, "is_feature_enabled", AsyncMock(return_value=False)
    ) as flag:
        assert not await rollout.shared_memory_enabled("user-1")
        flag.assert_awaited_once_with(Flag.SHOW_ORG_SETTINGS, "user-1", default=False)
        flag.reset_mock()
        assert not await rollout.shared_memory_enabled(None)
        flag.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_id", [None, "expert-1"])
async def test_off_warm_and_default_search_keep_only_personal_memory(expert_id):
    with (
        patch.object(tiers, "is_org_member", AsyncMock()) as org_access,
        patch.object(tiers, "get_user_team_ids", AsyncMock()) as team_access,
    ):
        warm = await tiers.resolve_warm_targets(
            "user-1", "org-personal", "team-personal", expert_id=expert_id
        )
        search = await tiers.resolve_search_targets(
            "user-1",
            "org-personal",
            "all",
            session_team_id="team-personal",
            expert_id=expert_id,
        )
    assert warm == search
    assert len(warm) == 1
    assert warm[0].tier == tiers.MemoryTier.personal
    assert warm[0].group_id == derive_memory_group_id("user-1", expert_id)
    org_access.assert_not_awaited()
    team_access.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["org", "team"])
async def test_off_rejects_explicit_shared_search_and_store(tier):
    with pytest.raises(tiers.TierError, match="not enabled"):
        await tiers.resolve_search_targets("user-1", "org-1", tier)
    session = ChatSession.new(
        "user-1", dry_run=False, organization_id="org-1", team_id="team-1"
    )
    with (
        patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.graphiti_store.enqueue_episode", AsyncMock()
        ) as enqueue,
    ):
        result = await MemoryStoreTool()._execute(
            "user-1", session, name="Policy", content="A shared policy", tier=tier
        )
    assert isinstance(result, ErrorResponse)
    assert "not enabled" in result.message
    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_off_keeps_personal_store_in_personal_organization():
    session = ChatSession.new(
        "user-1", dry_run=False, organization_id="org-personal", team_id="team-personal"
    )
    with (
        patch(
            "backend.copilot.tools.graphiti_store.is_enabled_for_user",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.tools.graphiti_store.enqueue_episode",
            AsyncMock(return_value=True),
        ) as enqueue,
    ):
        result = await MemoryStoreTool()._execute(
            "user-1", session, name="Preference", content="I prefer tea"
        )
    assert isinstance(result, MemoryStoreResponse)
    assert enqueue.await_args.kwargs["group_id"] is None


@pytest.mark.asyncio
async def test_off_rejects_shared_enqueue_and_drops_already_queued_work():
    with patch.object(
        ingest, "_enqueue_payload", AsyncMock(return_value=True)
    ) as enqueue:
        assert not await ingest.enqueue_episode(
            "user-1",
            "session-1",
            name="Shared",
            episode_body="Policy",
            group_id="org_org-1",
            organization_id="org-1",
        )
        enqueue.assert_not_awaited()
        assert await ingest.enqueue_episode(
            "user-1", "session-1", name="Private", episode_body="Preference"
        )
        assert enqueue.await_args.args[1] == "user_user-1"
    with patch.object(ingest, "get_graphiti_client", AsyncMock()) as client:
        await ingest._process_ingestion_payload(
            "user-1",
            "org_org-1",
            {
                "group_id": "org_org-1",
                "_resource_scope": {
                    "organization_id": "org-1",
                    "team_id": None,
                },
            },
        )
    client.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [MemoryForgetSearchTool(), MemoryForgetConfirmTool()])
async def test_off_shared_forget_does_not_offer_admin_features(tool):
    with patch(
        "backend.copilot.tools.graphiti_forget.is_enabled_for_user",
        AsyncMock(return_value=True),
    ):
        result = await tool._execute(
            "user-1", ChatSession.new("user-1", dry_run=False), tier="org"
        )
    assert isinstance(result, ErrorResponse)
    assert "not enabled" in result.message


def test_baseline_and_sdk_hide_shared_arguments_without_mutating_enabled_schema():
    enabled = get_available_tools(shared_memory=True)
    disabled = get_available_tools()
    for schema in disabled:
        name = schema["function"]["name"]
        if not name.startswith("memory_"):
            continue
        properties = schema["function"]["parameters"]["properties"]
        assert "tier" not in properties
        assert "team_id" not in properties
    store = MemoryStoreTool()
    assert "tier" not in _build_input_schema(store)["properties"]
    assert "tier" in _build_input_schema(store, shared_memory=True)["properties"]
    assert "tier" in store.parameters["properties"]
    assert any(
        "tier" in schema["function"]["parameters"]["properties"]
        for schema in enabled
        if schema["function"]["name"] == "memory_store"
    )


def test_prompt_only_teaches_shared_memory_when_enabled():
    assert "MEMORY TIERS" not in get_graphiti_supplement()
    assert 'tier="team"' not in get_graphiti_supplement()
    assert "memory_search" in get_graphiti_supplement()
    assert "MEMORY TIERS" in get_graphiti_supplement(shared_memory=True)
