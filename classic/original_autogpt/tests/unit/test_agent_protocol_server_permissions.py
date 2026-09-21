from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt.app.agent_protocol_server import AgentProtocolServer

from forge.agent_protocol.models import StepRequestBody, TaskRequestBody
from forge.permissions import ApprovalScope, CommandPermissionManager


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
