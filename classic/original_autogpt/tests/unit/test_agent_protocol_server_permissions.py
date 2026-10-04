from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt.app.agent_protocol_server import AgentProtocolServer

from forge.agent_protocol.models import StepRequestBody, TaskRequestBody
from forge.permissions import (
    ApprovalScope,
    CommandPermissionManager,
    PermissionCheckResult,
)
from forge.utils.exceptions import AgentFinished


def test_permission_manager_is_noninteractive_and_default_deny(tmp_path):
    server = AgentProtocolServer.__new__(AgentProtocolServer)
    server.app_config = MagicMock()
    server.app_config.workspace = tmp_path
    server.app_config.app_data_dir = tmp_path / ".autogpt"

    manager = server._get_permission_manager("AutoGPT-task-1")
    result = manager.check_command("execute_shell", {"command_line": "echo hello"})

    assert isinstance(manager, CommandPermissionManager)
    assert manager.prompt_fn is None
    assert not result.allowed
    assert result.scope is ApprovalScope.DENY


@pytest.mark.asyncio
async def test_create_task_passes_permission_manager():
    server = AgentProtocolServer.__new__(AgentProtocolServer)
    task = SimpleNamespace(
        task_id="task-1",
        input="Inspect the workspace",
        additional_input={},
    )
    server.db = MagicMock()
    server.db.create_task = AsyncMock(return_value=task)
    server.app_config = MagicMock()
    server.file_storage = MagicMock()
    server._get_task_llm_provider = MagicMock(return_value=MagicMock())
    manager = MagicMock(spec=CommandPermissionManager)
    server._get_permission_manager = MagicMock(return_value=manager)
    agent = MagicMock()
    agent.file_manager.save_state = AsyncMock()

    with patch(
        "autogpt.app.agent_protocol_server.create_agent", return_value=agent
    ) as create_agent:
        result = await server.create_task(
            TaskRequestBody(input="Inspect the workspace")
        )

    assert result is task
    server._get_permission_manager.assert_called_once_with("AutoGPT-task-1")
    assert create_agent.call_args.kwargs["permission_manager"] is manager


@pytest.mark.asyncio
async def test_execute_step_passes_permission_manager():
    server = AgentProtocolServer.__new__(AgentProtocolServer)
    task = SimpleNamespace(
        task_id="task-1",
        input="Inspect the workspace",
        additional_input={},
    )
    state = MagicMock()
    server.get_task = AsyncMock(return_value=task)
    server.agent_manager = MagicMock()
    server.agent_manager.load_agent_state = MagicMock(return_value=state)
    server.app_config = MagicMock()
    server.file_storage = MagicMock()
    server._get_task_llm_provider = MagicMock(return_value=MagicMock())
    manager = MagicMock(spec=CommandPermissionManager)
    server._get_permission_manager = MagicMock(return_value=manager)
    completed_step = MagicMock()
    server.db = MagicMock()
    server.db.create_step = AsyncMock(return_value=SimpleNamespace(step_id="step-1"))
    server.db.update_step = AsyncMock(return_value=completed_step)
    agent = MagicMock()
    agent.event_history = []
    agent.propose_action = AsyncMock(side_effect=RuntimeError("stop after wiring"))

    with patch(
        "autogpt.app.agent_protocol_server.configure_agent_with_state",
        return_value=agent,
    ) as configure_agent:
        result = await server.execute_step("task-1", StepRequestBody(input=""))

    assert result is completed_step
    server._get_permission_manager.assert_called_once_with("AutoGPT-task-1")
    assert configure_agent.call_args.kwargs["permission_manager"] is manager


@pytest.mark.asyncio
async def test_execute_step_enforces_ask_user_permission():
    server = AgentProtocolServer.__new__(AgentProtocolServer)
    task = SimpleNamespace(
        task_id="task-1",
        input="Inspect the workspace",
        additional_input={},
    )
    proposal = MagicMock()
    proposal.use_tool.name = "ask_user"
    proposal.use_tool.arguments = {"question": "Which folder?"}
    state = MagicMock()
    server.get_task = AsyncMock(return_value=task)
    server.agent_manager = MagicMock()
    server.agent_manager.load_agent_state = MagicMock(return_value=state)
    server.app_config = MagicMock()
    server.file_storage = MagicMock()
    server._get_task_llm_provider = MagicMock(return_value=MagicMock())
    manager = MagicMock(spec=CommandPermissionManager)
    manager.check_command.return_value = PermissionCheckResult(
        allowed=False, scope=ApprovalScope.DENY
    )
    server._get_permission_manager = MagicMock(return_value=manager)
    completed_step = MagicMock()
    server.db = MagicMock()
    server.db.create_step = AsyncMock(return_value=SimpleNamespace(step_id="step-1"))
    server.db.update_step = AsyncMock(return_value=completed_step)
    agent = MagicMock()
    agent.permission_manager = manager
    agent.event_history.current_episode = SimpleNamespace(result=None, action=proposal)
    agent.do_not_execute = AsyncMock(return_value=MagicMock())
    agent.propose_action = AsyncMock(side_effect=RuntimeError("stop after check"))

    with patch(
        "autogpt.app.agent_protocol_server.configure_agent_with_state",
        return_value=agent,
    ):
        result = await server.execute_step(
            "task-1", StepRequestBody(input="The reports folder")
        )

    assert result is completed_step
    manager.check_command.assert_called_once_with(
        "ask_user", {"question": "Which folder?"}
    )
    agent.do_not_execute.assert_awaited_once()
    agent.event_history.register_result.assert_not_called()


@pytest.mark.asyncio
async def test_denied_finish_step_is_not_marked_final():
    server = AgentProtocolServer.__new__(AgentProtocolServer)
    task = SimpleNamespace(
        task_id="task-1",
        input="Inspect the workspace",
        additional_input={},
    )
    proposal = MagicMock()
    proposal.use_tool.name = "finish"
    proposal.use_tool.arguments = {"reason": "Done"}
    server.get_task = AsyncMock(return_value=task)
    server.agent_manager = MagicMock()
    server.app_config = MagicMock()
    server.file_storage = MagicMock()
    server._get_task_llm_provider = MagicMock(return_value=MagicMock())
    server._get_permission_manager = MagicMock(
        return_value=MagicMock(spec=CommandPermissionManager)
    )
    server.db = MagicMock()
    server.db.create_step = AsyncMock(return_value=SimpleNamespace(step_id="step-1"))
    server.db.update_step = AsyncMock(return_value=MagicMock())
    agent = MagicMock()
    agent.event_history.current_episode = SimpleNamespace(result=None, action=proposal)
    agent.execute = AsyncMock(return_value=MagicMock())
    agent.propose_action = AsyncMock(side_effect=RuntimeError("stop after denial"))

    with patch(
        "autogpt.app.agent_protocol_server.configure_agent_with_state",
        return_value=agent,
    ):
        await server.execute_step("task-1", StepRequestBody(input=""))

    assert server.db.create_step.call_args.kwargs["is_last"] is False


@pytest.mark.asyncio
async def test_successful_finish_step_is_marked_final():
    server = AgentProtocolServer.__new__(AgentProtocolServer)
    task = SimpleNamespace(
        task_id="task-1",
        input="Inspect the workspace",
        additional_input={},
    )
    proposal = MagicMock()
    proposal.use_tool.name = "finish"
    proposal.use_tool.arguments = {"reason": "Done"}
    server.get_task = AsyncMock(return_value=task)
    server.agent_manager = MagicMock()
    server.app_config = MagicMock()
    server.file_storage = MagicMock()
    llm_provider = MagicMock()
    llm_provider.get_incurred_cost.return_value = 0
    server._get_task_llm_provider = MagicMock(return_value=llm_provider)
    server._get_permission_manager = MagicMock(
        return_value=MagicMock(spec=CommandPermissionManager)
    )
    completed_step = MagicMock()
    server.db = MagicMock()
    server.db.create_step = AsyncMock(return_value=SimpleNamespace(step_id="step-1"))
    server.db.update_step = AsyncMock(side_effect=[MagicMock(), completed_step])
    agent = MagicMock()
    agent.event_history.current_episode = SimpleNamespace(result=None, action=proposal)
    agent.execute = AsyncMock(side_effect=AgentFinished("Done"))
    agent.file_manager.save_state = AsyncMock()

    with patch(
        "autogpt.app.agent_protocol_server.configure_agent_with_state",
        return_value=agent,
    ):
        result = await server.execute_step("task-1", StepRequestBody(input=""))

    assert result is completed_step
    assert server.db.create_step.call_args.kwargs["is_last"] is False
    assert server.db.update_step.await_args_list[1].kwargs["is_last"] is True
