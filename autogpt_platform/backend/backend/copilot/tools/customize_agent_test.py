"""Tests for CustomizeAgentTool local mode."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from backend.copilot.tools.customize_agent import CustomizeAgentTool
from backend.copilot.tools.models import AgentPreviewResponse, ErrorResponse

from ._test_data import make_session

_TEST_USER_ID = "test-user-customize-agent"
_PIPELINE = "backend.copilot.tools.agent_generator.pipeline"


@pytest.fixture
def tool():
    return CustomizeAgentTool()


@pytest.fixture
def session():
    return make_session(_TEST_USER_ID)


# ── Input validation tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_agent_json_returns_error(tool, session):
    """Missing agent_json returns ErrorResponse."""
    result = await tool._execute(
        user_id=_TEST_USER_ID,
        session=session,
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_agent_json"


# ── Local mode tests (agent_json provided) ───────────────────────────────


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
    agent_json = _agent(name="Customized Agent", description="A customized agent")

    with _patched_pipeline(agent_json):
        result = await tool._execute(
            user_id=_TEST_USER_ID,
            session=session,
            agent_json=agent_json,
            save=False,
        )

    assert isinstance(result, AgentPreviewResponse)
    assert result.agent_name == "Customized Agent"
    assert result.node_count == 1


@pytest.mark.asyncio
async def test_local_mode_validation_failure(tool, session):
    """Local mode returns ErrorResponse when validation fails."""
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
