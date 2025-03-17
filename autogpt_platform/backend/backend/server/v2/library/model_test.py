import datetime

import prisma.fields
import prisma.models
import pytest

import backend.server.v2.library.model as library_model
from backend.util import json


@pytest.mark.asyncio
async def test_agent_preset_from_db():
    # Create mock DB agent
    db_agent = prisma.models.AgentPreset(
        id="test-agent-123",
        createdAt=datetime.datetime.now(),
        updatedAt=datetime.datetime.now(),
        agentId="agent-123",
        agentVersion=1,
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
                data=json.dumps({"type": "string", "value": "test value"}),  # type: ignore
            )
        ],
    )

    # Convert to LibraryAgentPreset
    agent = library_model.LibraryAgentPreset.from_db(db_agent)

    assert agent.id == "test-agent-123"
    assert agent.agent_version == 1
    assert agent.is_active is True
    assert agent.name == "Test Agent"
    assert agent.description == "Test agent description"
    assert agent.inputs == {"input1": {"type": "string", "value": "test value"}}
