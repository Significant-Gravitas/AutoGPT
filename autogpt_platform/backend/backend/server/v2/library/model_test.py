import backend.server.v2.library.model


def test_library_agent():
    agent = backend.server.v2.library.model.LibraryAgent(
        agent_id="test-agent-123",
        agent_version=1,
        name="Test Agent",
        description="Test description",
        isCreatedByUser=False,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.agent_id == "test-agent-123"
    assert agent.agent_version == 1
    assert agent.name == "Test Agent"
    assert agent.description == "Test description"
    assert agent.isCreatedByUser is False
    assert agent.input_schema == {"type": "object", "properties": {}}
    assert agent.output_schema == {"type": "object", "properties": {}}


def test_library_agent_with_user_created():
    agent = backend.server.v2.library.model.LibraryAgent(
        agent_id="user-agent-456",
        agent_version=2,
        name="User Created Agent",
        description="An agent created by the user",
        isCreatedByUser=True,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.agent_id == "user-agent-456"
    assert agent.agent_version == 2
    assert agent.name == "User Created Agent"
    assert agent.description == "An agent created by the user"
    assert agent.isCreatedByUser is True
    assert agent.input_schema == {"type": "object", "properties": {}}
    assert agent.output_schema == {"type": "object", "properties": {}}
