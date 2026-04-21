"""Tests for EditAgentTool's builder-session guard.

We cover only the pre-flight validation that lives entirely inside
``_execute`` — the rest of the pipeline (fetching the existing agent,
fix+validate+save) is exercised by the agent-generation pipeline tests.
"""

import pytest

from backend.copilot.model import ChatSessionMetadata
from backend.copilot.tools.edit_agent import EditAgentTool
from backend.copilot.tools.models import ErrorResponse

from ._test_data import make_session

_USER_ID = "test-user-edit-agent-guard"


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
