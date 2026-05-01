import datetime

import prisma.enums
import prisma.models
import pytest

from . import model as library_model


def _make_library_agent(
    *,
    graph_id: str = "g1",
    executions: list | None = None,
) -> prisma.models.LibraryAgent:
    return prisma.models.LibraryAgent(
        id="la1",
        userId="u1",
        agentGraphId=graph_id,
        settings="{}",  # type: ignore
        agentGraphVersion=1,
        isCreatedByUser=True,
        isDeleted=False,
        isArchived=False,
        createdAt=datetime.datetime.now(),
        updatedAt=datetime.datetime.now(),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        AgentGraph=prisma.models.AgentGraph(
            id=graph_id,
            version=1,
            name="Agent",
            description="Desc",
            userId="u1",
            isActive=True,
            createdAt=datetime.datetime.now(),
            Executions=executions,
        ),
    )


def test_from_db_execution_count_override_covers_success_rate():
    """Covers execution_count_override is not None branch and executions/count > 0 block."""
    now = datetime.datetime.now(datetime.timezone.utc)
    exec1 = prisma.models.AgentGraphExecution(
        id="exec-1",
        agentGraphId="g1",
        agentGraphVersion=1,
        userId="u1",
        executionStatus=prisma.enums.AgentExecutionStatus.COMPLETED,
        createdAt=now,
        updatedAt=now,
        isDeleted=False,
        isShared=False,
    )
    agent = _make_library_agent(executions=[exec1])

    result = library_model.LibraryAgent.from_db(agent, execution_count_override=1)

    assert result.execution_count == 1
    assert result.success_rate is not None
    assert result.success_rate == 100.0


@pytest.mark.asyncio
async def test_agent_preset_from_db(test_user_id: str):
    # Create mock DB agent
    db_agent = prisma.models.AgentPreset(
        id="test-agent-123",
        createdAt=datetime.datetime.now(),
        updatedAt=datetime.datetime.now(),
        agentGraphId="agent-123",
        agentGraphVersion=1,
        name="Test Agent",
        description="Test agent description",
        isActive=True,
        userId=test_user_id,
        isDeleted=False,
        InputPresets=[
            prisma.models.AgentNodeExecutionInputOutput.model_validate(
                {
                    "id": "input-123",
                    "time": datetime.datetime.now(),
                    "name": "input1",
                    "data": '{"type": "string", "value": "test value"}',
                }
            )
        ],
    )

    # Convert to LibraryAgentPreset
    agent = library_model.LibraryAgentPreset.from_db(db_agent)

    assert agent.id == "test-agent-123"
    assert agent.graph_version == 1
    assert agent.is_active is True
    assert agent.name == "Test Agent"
    assert agent.description == "Test agent description"
    assert agent.inputs == {"input1": {"type": "string", "value": "test value"}}
