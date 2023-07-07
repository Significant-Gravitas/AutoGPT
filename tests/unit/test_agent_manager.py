import pytest

from autogpt.agent.agent_manager import AgentManager
from autogpt.llm import ChatModelResponse
from autogpt.llm.chat import create_chat_completion
from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS


@pytest.fixture
def agent_manager(config):
    # Hack, real gross. Singletons are not good times.
    yield AgentManager(config)
    del AgentManager._instances[AgentManager]


@pytest.fixture
def task():
    return "translate English to French"


@pytest.fixture
def prompt():
    return "Translate the following English text to French: 'Hello, how are you?'"


@pytest.fixture
def model():
    return "gpt-3.5-turbo"


@pytest.fixture(autouse=True)
def mock_create_chat_completion(mocker, config):
    mock_create_chat_completion = mocker.patch(
        "autogpt.agent.agent_manager.create_chat_completion",
        wraps=create_chat_completion,
    )
    mock_create_chat_completion.return_value = ChatModelResponse(
        model_info=OPEN_AI_CHAT_MODELS[config.fast_llm],
        content="irrelevant",
        function_call={},
    )
    return mock_create_chat_completion


def test_create_agent(agent_manager: AgentManager, task, prompt, model):
    key, agent_reply = agent_manager.create_agent(task, prompt, model)
    assert isinstance(key, int)
    assert isinstance(agent_reply, str)
    assert key in agent_manager.agents


def test_message_agent(agent_manager: AgentManager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    user_message = "Please translate 'Good morning' to French."
    agent_reply = agent_manager.message_agent(key, user_message)
    assert isinstance(agent_reply, str)


def test_list_agents(agent_manager: AgentManager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    agents_list = agent_manager.list_agents()
    assert isinstance(agents_list, list)
    assert (key, task) in agents_list


def test_delete_agent(agent_manager: AgentManager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    success = agent_manager.delete_agent(key)
    assert success
    assert key not in agent_manager.agents
