from unittest.mock import AsyncMock, MagicMock

import pytest

from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.message_agent_user import Emiter, MessageAgentUser, Questions
from AFAAS.lib.message_common import AFAASMessage, AFAASMessageStack
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
)


# Test Instance Creation
@pytest.mark.asyncio
async def test_message_agent_user_creation():
    message_agent_user = MessageAgentUser(
        emitter=Emiter.USER.value,
        user_id="user123",
        agent_id="agent123",
        message="Sample message",
    )
    assert message_agent_user.emitter == Emiter.USER.value
    assert message_agent_user.user_id == "user123"
    assert message_agent_user.agent_id == "agent123"
    assert message_agent_user.message == "Sample message"
    assert isinstance(message_agent_user.message_id, str)


# Test UUID Generation
@pytest.mark.asyncio
async def test_uuid_generation():
    id1 = MessageAgentUser.generate_uuid()
    id2 = MessageAgentUser.generate_uuid()
    assert id1.startswith("MAU")
    assert id2.startswith("MAU")
    assert id1 != id2


# Test Load Method
@pytest.mark.asyncio
async def test_load_method(default_task: Task):
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.agent_id = "agent123"
    mock_table = MagicMock()
    mock_agent.db.get_table.return_value = mock_table
    mock_table.list.return_value = AsyncMock(
        return_value=[AFAASMessage(message_id="MESSAGE123")]
    )

    result = await default_task.agent.message_agent_user.load(
        default_task.agent, MessageAgentUser
    )
    assert isinstance(result, AFAASMessageStack)
    assert isinstance(result._messages, dict)


@pytest.mark.asyncio
async def test_load_method_integration(default_task: Task):
    agent = default_task.agent
    message1 = MessageAgentUser(
        emitter=Emiter.USER,
        user_id=agent.user_id,
        agent_id=agent.agent_id,
        message="Sample message 1",
    )
    message2 = MessageAgentUser(
        emitter=Emiter.AGENT,
        user_id=agent.user_id,
        agent_id=agent.agent_id,
        message="Sample message 2",
    )

    loaded_messages_before_creation = await agent.message_agent_user.load(
        agent, MessageAgentUser
    )
    assert len(agent.message_agent_user) == 0
    agent.message_agent_user = AFAASMessageStack(db=agent.db)

    assert len(agent.message_agent_user) == 0
    await agent.message_agent_user.db_create(message=message1)
    await agent.message_agent_user.db_create(message=message2)

    assert len(agent.message_agent_user) == 2
    loaded_messages = await agent.message_agent_user.load(agent, MessageAgentUser)

    # Validate that the loaded messages match what was inserted
    assert len(loaded_messages) == len(loaded_messages_before_creation) + 2
    assert any(v.message == "Sample message 1" for k, v in loaded_messages)
    assert any(v.message == "Sample message 2" for k, v in loaded_messages)
