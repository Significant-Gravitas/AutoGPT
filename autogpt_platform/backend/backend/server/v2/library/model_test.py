from datetime import datetime

import backend.data.block
import backend.server.model
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


def test_library_agent_preset():
    preset = backend.server.v2.library.model.LibraryAgentPreset(
        id="preset-123",
        name="Test Preset",
        description="Test preset description",
        agent_id="test-agent-123",
        agent_version=1,
        is_active=True,
        inputs={
            "input1": backend.data.block.BlockInput(
                name="input1",
                data={"type": "string", "value": "test value"},
            )
        },
        updated_at=datetime.now(),
    )
    assert preset.id == "preset-123"
    assert preset.name == "Test Preset"
    assert preset.description == "Test preset description"
    assert preset.agent_id == "test-agent-123"
    assert preset.agent_version == 1
    assert preset.is_active is True
    assert preset.inputs == {"input1": "test value"}


def test_library_agent_preset_response():
    preset = backend.server.v2.library.model.LibraryAgentPreset(
        id="preset-123",
        name="Test Preset",
        description="Test preset description",
        agent_id="test-agent-123",
        agent_version=1,
        is_active=True,
        inputs={
            "input1": backend.data.block.BlockInput(
                name="input1",
                data={"type": "string", "value": "test value"},
            )
        },
        updated_at=datetime.now(),
    )

    pagination = backend.server.model.Pagination(
        total_items=1, total_pages=1, current_page=1, page_size=10
    )

    response = backend.server.v2.library.model.LibraryAgentPresetResponse(
        presets=[preset], pagination=pagination
    )

    assert len(response.presets) == 1
    assert response.presets[0].id == "preset-123"
    assert response.pagination.total_items == 1
    assert response.pagination.total_pages == 1
    assert response.pagination.current_page == 1
    assert response.pagination.page_size == 10


def test_create_library_agent_preset_request():
    request = backend.server.v2.library.model.CreateLibraryAgentPresetRequest(
        name="New Preset",
        description="New preset description",
        agent_id="agent-123",
        agent_version=1,
        is_active=True,
        inputs={
            "input1": backend.data.block.BlockInput(
                name="input1",
                data={"type": "string", "value": "test value"},
            )
        },
    )

    assert request.name == "New Preset"
    assert request.description == "New preset description"
    assert request.agent_id == "agent-123"
    assert request.agent_version == 1
    assert request.is_active is True
    assert request.inputs == {"input1": "test value"}
