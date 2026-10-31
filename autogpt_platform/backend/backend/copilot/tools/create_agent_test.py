"""Tests for CreateAgentTool."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.create_agent import CreateAgentTool
from backend.copilot.tools.models import AgentPreviewResponse, ErrorResponse

from ._test_data import make_session

_TEST_USER_ID = "test-user-create-agent"
_PIPELINE = "backend.copilot.tools.agent_generator.pipeline"
_TOOL = "backend.copilot.tools.create_agent"


@pytest.fixture
def tool():
    return CreateAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── Input validation tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_agent_json_returns_error(tool, session):
    """Missing agent_json returns ErrorResponse."""
    result = await tool._execute(user_id=_TEST_USER_ID, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_json"


# ── Local mode tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_mode_empty_nodes_returns_error(tool, session):
    """Local mode with no nodes returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_json={"nodes": [], "links": []},
    )
    assert isinstance(result, ErrorResponse)
    assert "no nodes" in result.message.lower()


@pytest.mark.asyncio
async def test_local_mode_preview(tool, session):
    """Local mode with save=false returns AgentPreviewResponse."""
    agent_json = _agent(name="Test Agent", description="A test agent")

    with _patched_pipeline(agent_json):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
            save=False,
        )

    assert isinstance(result, AgentPreviewResponse)
    assert result.agent_name == "Test Agent"
    assert result.node_count == 1


@pytest.mark.asyncio
async def test_local_mode_validation_failure(tool, session):
    """Local mode returns ErrorResponse when validation fails after fixing."""
    agent_json = _agent({"block_id": "bad-block", "metadata": {}})

    with _patched_pipeline(agent_json, errors=["Block 'bad-block' not found"]):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "validation_failed"
    assert result.details is not None
    assert "Block 'bad-block' not found" in result.details["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_hidden", [True, False])
async def test_is_hidden_passes_through_to_pipeline(tool, session, is_hidden):
    """The is_hidden param on create_agent must reach
    fix_validate_and_save unchanged so trigger agents land in the DB
    hidden. Regression for an earlier review finding that the kwarg
    plumbing was untested."""
    agent_json = _agent(name="Trigger Watcher")

    with _patched_save() as mock_fix_validate:
        await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
            is_hidden=is_hidden,
        )

    mock_fix_validate.assert_awaited_once()
    assert mock_fix_validate.call_args.kwargs["is_hidden"] is is_hidden


# ── Library similarity gate tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_create_agent_gated_until_library_check(tool):
    """Without prior find_library_agent, create_agent must refuse."""
    session = make_session(_TEST_USER_ID, library_check=False)

    agent_json = _agent(name="Test Agent")

    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
        agent_json=agent_json,
    )

    assert isinstance(result, ErrorResponse)
    assert "find_library_agent" in result.message
    assert "library_check_ack" in result.message


@pytest.mark.asyncio
async def test_create_agent_skipped_with_ack(tool):
    """library_check_ack=True bypasses the gate even when find_library_agent
    was never called this session."""
    session = make_session(_TEST_USER_ID, library_check=False)

    agent_json = _agent(name="Test Agent", description="A test agent")

    with _patched_save() as mock_fix_validate:
        await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
            library_check_ack=True,
        )

    mock_fix_validate.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_agent_skipped_in_builder_context(tool):
    """Builder-bound sessions bypass the library gate (no goal to match)."""
    session = make_session(_TEST_USER_ID, library_check=False)
    session.metadata.builder_graph_id = "some-graph-id"

    agent_json = _agent(name="Test Agent", description="A test agent")

    with _patched_save() as mock_fix_validate:
        await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
        )

    mock_fix_validate.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_mode_no_auth_returns_error(tool, session):
    """Local mode with save=true and no user returns ErrorResponse."""
    agent_json = _agent({"metadata": {}})

    with _patched_pipeline(agent_json):
        result = await tool._execute(
            user_id=None,
            session=session,
            agent_json=agent_json,
            save=True,
        )

    assert isinstance(result, ErrorResponse)
    assert "logged in" in result.message.lower()


def _agent(nodes_extra: dict | None = None, **fields) -> dict:
    """A one-node agent JSON: nodes_extra overrides the node, fields go on top."""
    node = {
        "id": "node-1",
        "block_id": "block-1",
        "input_default": {},
        "metadata": {"position": {"x": 0, "y": 0}},
    }
    if nodes_extra:
        node.update(nodes_extra)
    return {**fields, "nodes": [node], "links": []}


@contextmanager
def _patched_pipeline(fixed_agent: dict, errors: list[str] | None = None):
    """The pipeline's fixer returns fixed_agent with no fixes; its validator
    passes unless given errors."""
    errors = errors or []
    fixer = MagicMock()
    fixer.apply_all_fixes.return_value = fixed_agent
    fixer.get_fixes_applied.return_value = []
    validator = MagicMock()
    validator.validate.return_value = (False, errors[0]) if errors else (True, None)
    validator.errors = errors
    with (
        patch(f"{_PIPELINE}.get_blocks_as_dicts", return_value=[]),
        patch(f"{_PIPELINE}.AgentFixer", return_value=fixer),
        patch(f"{_PIPELINE}.AgentValidator", return_value=validator),
    ):
        yield


@contextmanager
def _patched_save():
    """Stub the library lookup and the pipeline; yields the pipeline's mock."""
    with (
        patch(f"{_TOOL}.fetch_library_agents", new=AsyncMock(return_value=[])),
        patch(
            f"{_TOOL}.fix_validate_and_save", new=AsyncMock(return_value=MagicMock())
        ) as fix_validate,
    ):
        yield fix_validate
