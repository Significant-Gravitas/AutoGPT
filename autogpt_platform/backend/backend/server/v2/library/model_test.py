import datetime

import prisma.fields
import prisma.models
import pytest

import backend.server.v2.library.model as library_model


@pytest.mark.asyncio
async def test_agent_preset_from_db():
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
        userId="test-user-123",
        isDeleted=False,
        InputPresets=[
            prisma.models.AgentNodeExecutionInputOutput(
                id="input-123",
                time=datetime.datetime.now(),
                name="input1",
                data=prisma.Json({"type": "string", "value": "test value"}),
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
