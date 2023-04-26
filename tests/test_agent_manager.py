from typing import List

import pytest

from autogpt.agent.agent_manager import AgentManager
from tests.utils import requires_api_key


@pytest.fixture
def agent_manager():
    # Hack, real gross. Singletons are not good times.
    if AgentManager in AgentManager._instances:
        del AgentManager._instances[AgentManager]
    return AgentManager()


@pytest.fixture
def task():
    return "translate English to French"


@pytest.fixture
def prompt():
    return "Translate the following English text to French: 'Hello, how are you?'"


@pytest.fixture
def model():
    return "gpt-3.5-turbo"


@requires_api_key("OPENAI_API_KEY")
def test_create_agent(agent_manager, task, prompt, model):
    key, agent_reply = agent_manager.create_agent(task, prompt, model)
    assert isinstance(key, int)
    assert isinstance(agent_reply, str)
    assert key in agent_manager.agents


@requires_api_key("OPENAI_API_KEY")
def test_message_agent(agent_manager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    user_message = "Please translate 'Good morning' to French."
    agent_reply = agent_manager.message_agent(key, user_message)
    assert isinstance(agent_reply, str)


@requires_api_key("OPENAI_API_KEY")
def test_list_agents(agent_manager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    agents_list = agent_manager.list_agents()
    assert isinstance(agents_list, list)
    assert (key, task) in agents_list


@requires_api_key("OPENAI_API_KEY")
def test_delete_agent(agent_manager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    success = agent_manager.delete_agent(key)
    assert success
    assert key not in agent_manager.agents
