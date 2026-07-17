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
        isHidden=False,
        createdAt=datetime.datetime.now(),
        updatedAt=datetime.datetime.now(),
        isFavorite=False,
        useGraphIsActiveVersion=True,
        visibility=prisma.enums.ResourceVisibility.PRIVATE,
        AgentGraph=prisma.models.AgentGraph(
            id=graph_id,
            version=1,
            name="Agent",
            description="Desc",
            userId="u1",
            isActive=True,
            createdAt=datetime.datetime.now(),
            visibility=prisma.enums.ResourceVisibility.PRIVATE,
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
        visibility=prisma.enums.ResourceVisibility.PRIVATE,
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
        visibility=prisma.enums.ResourceVisibility.PRIVATE,
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


def test_preset_serialization_redacts_webhook_signing_material():
    """A serialized preset must never leak the webhook's signing secret or
    provider webhook id (GHSA-4m2w-qfr5-9f3v), while still exposing the ingress
    URL and non-sensitive metadata the frontend needs."""
    from backend.data.integrations import Webhook
    from backend.integrations.providers import ProviderName

    webhook = Webhook(
        id="wh-1",
        user_id="u1",
        provider=ProviderName.GITHUB,
        credentials_id="cred-1",
        webhook_type="repo",
        resource="owner/repo",
        events=["pull_request"],
        config={},
        secret="super-secret-signing-key",
        provider_webhook_id="gh-provider-123",
    )
    preset = library_model.LibraryAgentPreset(
        id="preset-1",
        user_id="u1",
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        graph_id="g1",
        graph_version=1,
        inputs={},
        credentials={},
        name="p",
        description="",
        is_active=True,
        webhook_id="wh-1",
        webhook=webhook,
    )

    dumped = preset.model_dump(mode="json")
    assert dumped["webhook"] is not None
    assert "secret" not in dumped["webhook"]
    assert "provider_webhook_id" not in dumped["webhook"]
    # Non-sensitive fields callers rely on are preserved.
    assert dumped["webhook"]["id"] == "wh-1"
    assert dumped["webhook"]["url"]

    # The secret must not leak through JSON string serialization either.
    serialized = preset.model_dump_json()
    assert "super-secret-signing-key" not in serialized
    assert "gh-provider-123" not in serialized
