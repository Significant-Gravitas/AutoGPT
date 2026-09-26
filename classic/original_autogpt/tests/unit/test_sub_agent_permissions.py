from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt.agent_factory.default_factory import DefaultAgentFactory
from autogpt.agents.agent import Agent
from autogpt.agents.prompt_strategies.base import BaseMultiStepPromptStrategy

from forge.agent.execution_context import SubAgentHandle
from forge.utils.exceptions import AgentFinished


def test_root_execution_context_passes_permission_manager_to_factory():
    agent = MagicMock(spec=Agent)
    agent.permission_manager = MagicMock()
    llm_provider = MagicMock()
    file_storage = MagicMock()
    app_config = MagicMock()

    with (
        patch(
            "autogpt.agent_factory.default_factory.DefaultAgentFactory"
        ) as factory_class,
        patch("autogpt.agents.agent.ExecutionContext") as context_class,
    ):
        Agent._create_root_execution_context(
            agent,
            llm_provider=llm_provider,
            file_storage=file_storage,
            app_config=app_config,
        )

    factory_class.assert_called_once_with(app_config, agent.permission_manager)
    context_class.assert_called_once_with(
        llm_provider=llm_provider,
        file_storage=file_storage,
        agent_factory=factory_class.return_value,
        parent_agent_id=None,
        depth=0,
        _app_config=app_config,
    )


def test_default_factory_passes_permission_manager_to_child_agent():
    app_config = MagicMock()
    permission_manager = MagicMock()
    context = MagicMock()
    agent_state = MagicMock()

    factory = DefaultAgentFactory(app_config, permission_manager)
    factory._create_agent_state = MagicMock(return_value=agent_state)

    with patch("autogpt.agent_factory.default_factory.Agent") as agent_class:
        factory.create_agent("child-id", "child task", context)

    assert agent_class.call_args.kwargs["permission_manager"] is permission_manager


@pytest.mark.asyncio
async def test_sub_agent_finish_runs_permission_check():
    strategy = MagicMock()
    strategy._execution_context = None
    strategy.logger = MagicMock()
    proposal = MagicMock()
    proposal.use_tool.name = "finish"
    proposal.use_tool.arguments = {"reason": "Done"}
    agent = MagicMock()
    agent.propose_action = AsyncMock(return_value=proposal)
    agent.execute = AsyncMock(side_effect=AgentFinished("Done"))
    handle = SubAgentHandle(agent_id="child", task="child task")

    result = await BaseMultiStepPromptStrategy._run_agent_loop(
        strategy, agent, max_cycles=1, handle=handle
    )

    assert result == "Done"
    agent.execute.assert_awaited_once_with(proposal)


@pytest.mark.asyncio
async def test_denied_sub_agent_finish_does_not_complete():
    strategy = MagicMock()
    strategy._execution_context = None
    strategy.logger = MagicMock()
    proposal = MagicMock()
    proposal.use_tool.name = "finish"
    proposal.use_tool.arguments = {"reason": "Done"}
    agent = MagicMock()
    agent.propose_action = AsyncMock(return_value=proposal)
    agent.execute = AsyncMock(return_value=MagicMock())
    handle = SubAgentHandle(agent_id="child", task="child task")

    result = await BaseMultiStepPromptStrategy._run_agent_loop(
        strategy, agent, max_cycles=1, handle=handle
    )

    assert result is None
    agent.execute.assert_awaited_once_with(proposal)
