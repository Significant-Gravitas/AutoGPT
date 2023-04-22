import pytest
from autogpt.agent.agent_manager import AgentManager
from autogpt.config.config import Config
from autogpt.llm_utils import create_chat_completion
from autogpt.types.openai import Message
# In test_agents.py

@pytest.fixture
def agent_manager():
    manager = AgentManager()
    manager.reset()
    return manager


def test_create_agent(agent_manager):
    task = "Test Task"
    prompt = "Test Prompt"
    model = "gpt-3.5-turbo"

    agent_key, agent_reply = agent_manager.create_agent(task, prompt, model)

    assert agent_key == 0
    assert agent_reply != ""

def test_message_agent(agent_manager):
    task = "Test Task"
    prompt = "Test Prompt"
    model = "gpt-3.5-turbo"
    message = "Test Message"

    agent_key, _ = agent_manager.create_agent(task, prompt, model)
    agent_reply = agent_manager.message_agent(agent_key, message)

    assert agent_reply != ""

def test_list_agents(agent_manager):
    task = "Test Task"
    prompt = "Test Prompt"
    model = "gpt-3.5-turbo"

    agent_key, _ = agent_manager.create_agent(task, prompt, model)
    agent_list = agent_manager.list_agents()

    assert agent_list == [(agent_key, task)]

def test_delete_agent(agent_manager):
    task = "Test Task"
    prompt = "Test Prompt"
    model = "gpt-3.5-turbo"

    agent_key, _ = agent_manager.create_agent(task, prompt, model)
    delete_success = agent_manager.delete_agent(agent_key)

    assert delete_success == True
    agent_list = agent_manager.list_agents()
    assert agent_list == []

# Add more test cases as needed
