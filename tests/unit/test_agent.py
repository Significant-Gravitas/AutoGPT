from unittest.mock import MagicMock, Mock, patch

import pytest

from autogpt.agent import Agent
from autogpt.config import AIConfig
from autogpt.config.config import Config


@pytest.fixture
def agent(config: Config):
    ai_name = "Test AI"
    memory = MagicMock()
    next_action_count = 0
    command_registry = MagicMock()
    ai_config = AIConfig(ai_name=ai_name)
    system_prompt = "System prompt"
    triggering_prompt = "Triggering prompt"
    workspace_directory = "workspace_directory"

    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        next_action_count=next_action_count,
        command_registry=command_registry,
        ai_config=ai_config,
        config=config,
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
        workspace_directory=workspace_directory,
    )
    return agent


def test_agent_initialization(agent: Agent):
    assert agent.ai_name == "Test AI"
    assert agent.history.messages == []
    assert agent.next_action_count == 0
    assert agent.system_prompt == "System prompt"
    assert agent.triggering_prompt == "Triggering prompt"


PATHLIKE_COMMAND_ARGS = (
    ({"directory": ""}, {"directory": "root_directory"}),
    ({"directory": "/"}, {"directory": "root_directory"}),
    ({"directory": "dir"}, {"directory": "resolved_dir"}),
    ({"filename": "file.txt"}, {"filename": "resolved_file.txt"}),
    ({"clone_path": "clone"}, {"clone_path": "resolved_clone"}),
)


@pytest.mark.parametrize("args,expected", PATHLIKE_COMMAND_ARGS)
def test_resolve_pathlike_command_args(agent: Agent, args, expected):
    mock_workspace = Mock()
    mock_workspace.root = "root_directory"
    mock_workspace.get_path.side_effect = lambda path: f"resolved_{path}"
    with patch.object(agent, "workspace", mock_workspace):
        assert agent._resolve_pathlike_command_args(args) == expected


@patch("autogpt.agent.agent.create_chat_completion")
def test_agent_self_feedback(fake_completion, agent):
    agent.config.ai_role = "test-role"
    _ = agent.get_self_feedback(
        {
            "reasoning": "test reasoning",
            "plan": "test plan",
            "thoughts": "test thought",
            "criticism": "test criticism",
        },
        "test-llm-model",
    )
    data, model = fake_completion.call_args[0]
    assert data[0]["role"] == "user"
    assert model == "test-llm-model"
    content = data[0]["content"].splitlines()
    assert "from an AI agent with the role of test-role." in content[0]
    assert content[-4:] == [
        "test thought",
        "test reasoning",
        "test plan",
        "test criticism",
    ]
