"""Tests for EditAgentTool's builder-session guard.

We cover only the pre-flight validation that lives entirely inside
``_execute`` — the rest of the pipeline (fetching the existing agent,
fix+validate+save) is exercised by the agent-generation pipeline tests.
"""

import pytest

from backend.copilot.model import ChatSessionMetadata
from backend.copilot.tools.edit_agent import (
    EditAgentTool,
    _changed_trigger_config_fields,
)
from backend.copilot.tools.models import ErrorResponse

from ._test_data import make_session

_USER_ID = "test-user-edit-agent-guard"

_EDIT_MODULE = "backend.copilot.tools.edit_agent"
_WEBHOOK_BID = "webhook-trigger-block"


class _FakeSchema:
    # Mirrors a webhook trigger block: two config fields + one credentials field.
    model_fields = {
        "repo": object(),
        "events": object(),
        "payload_credentials": object(),
    }


class _FakeBlock:
    input_schema = _FakeSchema


def _patch_block_lookup(mocker) -> None:
    """Make the trigger-config helper see ``_WEBHOOK_BID`` as a webhook block
    with config fields ``repo``/``events`` (credentials excluded)."""
    mocker.patch(f"{_EDIT_MODULE}.get_webhook_block_ids", return_value=[_WEBHOOK_BID])
    mocker.patch(f"{_EDIT_MODULE}.get_block", return_value=_FakeBlock())


@pytest.fixture
def tool() -> EditAgentTool:
    return EditAgentTool()


@pytest.mark.asyncio
async def test_builder_session_rejects_foreign_agent_id(
    tool: EditAgentTool,
) -> None:
    """A builder-bound session cannot edit a different agent."""
    session = make_session(_USER_ID)
    session.metadata = ChatSessionMetadata(builder_graph_id="graph-bound")

    result = await tool._execute(
        user_id=_USER_ID,
        session=session,
        agent_id="graph-other",
        agent_json={"nodes": [{"id": "n1"}], "links": []},
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "builder_session_graph_mismatch"


@pytest.mark.asyncio
async def test_builder_session_defaults_missing_agent_id(
    tool: EditAgentTool,
    mocker,
) -> None:
    """Omitting ``agent_id`` in a builder session defaults to the bound graph."""
    session = make_session(_USER_ID)
    session.metadata = ChatSessionMetadata(builder_graph_id="graph-bound")

    # Stop the pipeline after the guard — we only care that the guard
    # accepted the default and moved on to the "does the agent exist"
    # lookup.  Returning ``None`` here turns into an ``agent_not_found``
    # error that proves the guard passed.
    mocker.patch(
        "backend.copilot.tools.edit_agent.get_agent_as_json",
        return_value=None,
    )

    result = await tool._execute(
        user_id=_USER_ID,
        session=session,
        agent_id="",  # intentionally empty
        agent_json={"nodes": [{"id": "n1"}], "links": []},
    )

    assert isinstance(result, ErrorResponse)
    # The guard defaulted to "graph-bound" and asked get_agent_as_json
    # for it.  The important signal is that we did NOT see the
    # builder_session_graph_mismatch or missing_agent_id errors.
    assert result.error != "builder_session_graph_mismatch"
    assert result.error != "missing_agent_id"


@pytest.mark.asyncio
async def test_non_builder_session_keeps_missing_agent_id_error(
    tool: EditAgentTool,
) -> None:
    """Outside the builder, omitting ``agent_id`` still errors with the
    plain ``missing_agent_id`` code — the builder guard does not widen
    the contract for non-builder sessions."""
    session = make_session(_USER_ID)

    result = await tool._execute(
        user_id=_USER_ID,
        session=session,
        agent_id="",
        agent_json={"nodes": [{"id": "n1"}], "links": []},
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_id"


# --- Trigger-config guard (Option A) --------------------------------------


def _trigger_node(repo: str, events: list[str]) -> dict:
    return {
        "id": "trigger-1",
        "block_id": _WEBHOOK_BID,
        "input_default": {"repo": repo, "events": events},
        "metadata": {},
    }


def test_changed_trigger_config_is_detected(mocker) -> None:
    _patch_block_lookup(mocker)
    current = [_trigger_node("owner/repo", ["opened"])]
    new = [_trigger_node("owner/OTHER", ["opened"])]
    assert _changed_trigger_config_fields(current, new) == ["repo"]


def test_unchanged_trigger_config_is_allowed(mocker) -> None:
    _patch_block_lookup(mocker)
    current = [_trigger_node("owner/repo", ["opened"])]
    new = [_trigger_node("owner/repo", ["opened"])]
    assert _changed_trigger_config_fields(current, new) == []


def test_removed_trigger_node_is_not_guarded(mocker) -> None:
    """Dropping the trigger node entirely is a structural edit, not a config
    tweak — the guard must not block it."""
    _patch_block_lookup(mocker)
    current = [_trigger_node("owner/repo", ["opened"])]
    new = [{"id": "other", "block_id": "some-block", "input_default": {}}]
    assert _changed_trigger_config_fields(current, new) == []


def test_agent_without_trigger_is_not_guarded(mocker) -> None:
    _patch_block_lookup(mocker)
    nodes = [{"id": "n1", "block_id": "some-block", "input_default": {"x": 1}}]
    assert _changed_trigger_config_fields(nodes, nodes) == []


def test_credentials_changes_are_ignored(mocker) -> None:
    """Changing the trigger node's credential field is not a trigger-config
    change (credentials are handled separately)."""
    _patch_block_lookup(mocker)
    current = [
        {
            "id": "trigger-1",
            "block_id": _WEBHOOK_BID,
            "input_default": {
                "repo": "owner/repo",
                "events": ["opened"],
                "payload_credentials": {"id": "cred-a"},
            },
        }
    ]
    new = [
        {
            "id": "trigger-1",
            "block_id": _WEBHOOK_BID,
            "input_default": {
                "repo": "owner/repo",
                "events": ["opened"],
                "payload_credentials": {"id": "cred-b"},
            },
        }
    ]
    assert _changed_trigger_config_fields(current, new) == []


@pytest.mark.asyncio
async def test_execute_blocks_trigger_config_edit(tool, mocker) -> None:
    """End-to-end: editing the trigger node's config through edit_agent is
    rejected with an actionable pointer to setup_agent_webhook_trigger."""
    _patch_block_lookup(mocker)
    session = make_session(_USER_ID)
    # builder_graph_id satisfies the guide gate; agent_id matches it so the
    # builder guard passes too.
    session.metadata = ChatSessionMetadata(builder_graph_id="graph-x")

    mocker.patch(
        f"{_EDIT_MODULE}.get_agent_as_json",
        return_value={
            "id": "graph-x",
            "version": 3,
            "nodes": [_trigger_node("owner/repo", ["opened"])],
        },
    )

    result = await tool._execute(
        user_id=_USER_ID,
        session=session,
        agent_id="graph-x",
        agent_json={"nodes": [_trigger_node("owner/CHANGED", ["opened"])], "links": []},
    )

    assert isinstance(result, ErrorResponse)
    assert result.error == "trigger_config_edit_blocked"
    assert "setup_agent_webhook_trigger" in result.message


@pytest.mark.asyncio
async def test_execute_allows_non_trigger_edit(tool, mocker) -> None:
    """An edit that leaves trigger config untouched flows past the guard into
    the validate-and-save pipeline."""
    _patch_block_lookup(mocker)
    session = make_session(_USER_ID)
    session.metadata = ChatSessionMetadata(builder_graph_id="graph-x")

    mocker.patch(
        f"{_EDIT_MODULE}.get_agent_as_json",
        return_value={
            "id": "graph-x",
            "version": 3,
            "nodes": [_trigger_node("owner/repo", ["opened"])],
        },
    )
    mocker.patch(f"{_EDIT_MODULE}.fetch_library_agents", return_value=[])
    sentinel = ErrorResponse(message="pipeline-reached", error="sentinel")
    saved = mocker.patch(f"{_EDIT_MODULE}.fix_validate_and_save", return_value=sentinel)

    result = await tool._execute(
        user_id=_USER_ID,
        session=session,
        agent_id="graph-x",
        # Same trigger config; only an unrelated node changes.
        agent_json={
            "nodes": [
                _trigger_node("owner/repo", ["opened"]),
                {"id": "n2", "block_id": "other", "input_default": {"x": 2}},
            ],
            "links": [],
        },
    )

    assert result is sentinel
    saved.assert_called_once()
