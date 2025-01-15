import backend.server.v2.library.model


def test_library_agent():
    agent = backend.server.v2.library.model.LibraryAgent(
        id="test-agent-123",
        version=1,
        is_active=True,
        name="Test Agent",
        description="Test description",
        isCreatedByUser=False,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.id == "test-agent-123"
    assert agent.version == 1
    assert agent.is_active is True
    assert agent.name == "Test Agent"
    assert agent.description == "Test description"
    assert agent.isCreatedByUser is False
    assert agent.input_schema == {"type": "object", "properties": {}}
    assert agent.output_schema == {"type": "object", "properties": {}}


def test_library_agent_with_user_created():
    agent = backend.server.v2.library.model.LibraryAgent(
        id="user-agent-456",
        version=2,
        is_active=True,
        name="User Created Agent",
        description="An agent created by the user",
        isCreatedByUser=True,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    assert agent.id == "user-agent-456"
    assert agent.version == 2
    assert agent.is_active is True
    assert agent.name == "User Created Agent"
    assert agent.description == "An agent created by the user"
    assert agent.isCreatedByUser is True
    assert agent.input_schema == {"type": "object", "properties": {}}
    assert agent.output_schema == {"type": "object", "properties": {}}
